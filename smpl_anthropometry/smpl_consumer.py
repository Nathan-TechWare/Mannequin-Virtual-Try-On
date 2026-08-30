"""
RabbitMQ consumer for SMPL-Anthropometry.

Listens for `mesh.ready` messages from AI-Tailor, runs the same
measurement logic as measure_my_mesh.py (height correction + full
measurement pass), saves measurement.json into the job's folder, and
publishes `measurements.ready` for Anny to pick up.

Kept as a separate file from measure_my_mesh.py on purpose -- that
script still works standalone with its own input() prompts if this
pipeline version has issues.
"""

import pika
import json
import os
import trimesh
import numpy as np
import torch
from measure import MeasureBody

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")


def process_message(ch, method, properties, body):
    payload = json.loads(body)
    job_id = payload.get('job_id')

    try:
        mesh_path = payload['mesh_path']
        gender = payload['gender']
        actual_height_cm = payload['height_cm']
        weight_kg = payload.get('weight_kg')
        age_years = payload.get('age')
        print(f"[INFO] Received mesh.ready for job {job_id} "
              f"(gender={gender}, height={actual_height_cm}cm, weight={weight_kg}kg, age={age_years}years)")

        # ---- Load the mesh AI-Tailor generated for this job ----
        mesh = trimesh.load(mesh_path)
        verts = torch.tensor(np.array(mesh.vertices, dtype=np.float32))
        print(f'Loaded mesh with {len(verts)} vertices')

        # ---- First pass: measure the mesh as-is to get its raw height ----
        measurer = MeasureBody('smplx')
        measurer.from_verts(verts=verts)
        measurer.measure(['height'])
        measured_height_cm = measurer.measurements['height']
        print(f'Measured height (pre-correction): {measured_height_cm:.2f} cm')

        # Guard against a degenerate mesh producing a zero/near-zero height,
        # which would blow up the scale_factor below.
        if measured_height_cm < 1.0:
            raise ValueError(
                f"implausible measured height {measured_height_cm:.2f} cm -- "
                f"mesh may be degenerate or in wrong units")

        # ---- Rescale all vertices so mesh height matches actual user height ----
        # This corrects every other measurement proportionally, not just height.
        scale_factor = actual_height_cm / measured_height_cm
        verts_corrected = verts * scale_factor

        # ---- Re-measure the corrected mesh ----
        measurer = MeasureBody('smplx')
        measurer.from_verts(verts=verts_corrected)
        measurement_names = measurer.all_possible_measurements
        measurer.measure(measurement_names)

        print('\n=== Measurements (cm) ===')
        for name, value in measurer.measurements.items():
            print(f'{name}: {value:.2f} cm')

        rounded_measurements = {name: round(float(value), 2)
                                for name, value in measurer.measurements.items()}

        output = {
            "height_cm": round(actual_height_cm, 2),
            "gender": gender,
            "weight_kg": weight_kg,
            "age_years": age_years,
            "measurements": rounded_measurements,
        }

        # ---- Save into the same job folder as the mesh, not cwd ----
        # mesh_path is absolute (AI-Tailor writes it that way), so this
        # resolves correctly regardless of where this consumer launched from.
        job_dir = os.path.dirname(mesh_path)
        measurements_path = os.path.join(job_dir, "measurement.json")
        with open(measurements_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\n[INFO] Saved {measurements_path}')

        # ---- Pass it on to Anny ----
        out_payload = {
            "job_id": job_id,
            "measurements_path": measurements_path,
            "status": "ready",
        }
        ch.basic_publish(
            exchange='pipeline',
            routing_key='measurements.ready',
            body=json.dumps(out_payload),
            properties=pika.BasicProperties(delivery_mode=2,
                                            content_type='application/json'),
        )
        print(f"[INFO] Published measurements.ready for job {job_id}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"[ERROR] failed job {job_id}: {e}")
        # requeue=False so a bad mesh doesn't become an infinite crash loop;
        # add a dead-letter queue before production if you want these kept
        # for inspection/retry.
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def start_consumer(host=RABBITMQ_HOST):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
    channel = connection.channel()

    channel.exchange_declare(exchange='pipeline', exchange_type='topic', durable=True)
    channel.queue_declare(queue='smpl_anthro_queue', durable=True)
    channel.queue_bind(exchange='pipeline', queue='smpl_anthro_queue', routing_key='mesh.ready')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='smpl_anthro_queue', on_message_callback=process_message)

    print("[INFO] Waiting for mesh.ready messages...")
    channel.start_consuming()


if __name__ == "__main__":
    start_consumer()
