"""
RabbitMQ consumer for anny_app. Listens for `measurements.ready` from
SMPL-Anthropometry, and writes that job's measurements into the single
MEASUREMENTS_PATH app.py already reads via _reload_state_from_disk().

Runs as a separate process from the Flask app, so it can't touch
app.py's in-memory state directly -- it hands off via the file on
disk, stamped with job_id, and app.py's /job_status endpoint notices
the match and reloads.
"""
import pika
import json
import os

from config import MEASUREMENTS_PATH

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")


def process_message(ch, method, properties, body):
    payload = json.loads(body)
    job_id = payload.get('job_id')
    try:
        with open(payload['measurements_path'], 'r') as f:
            job_data = json.load(f)
        job_data['job_id'] = job_id

        os.makedirs(os.path.dirname(MEASUREMENTS_PATH), exist_ok=True)
        with open(MEASUREMENTS_PATH, 'w') as f:
            json.dump(job_data, f, indent=2)

        print(f"[anny_consumer] wrote measurements for job {job_id}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"[anny_consumer] failed job {job_id}: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def start_consumer(host=RABBITMQ_HOST):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
    channel = connection.channel()
    channel.exchange_declare(exchange='pipeline', exchange_type='topic', durable=True)
    channel.queue_declare(queue='anny_queue', durable=True)
    channel.queue_bind(exchange='pipeline', queue='anny_queue', routing_key='measurements.ready')
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='anny_queue', on_message_callback=process_message)
    print("[anny_consumer] waiting for measurements.ready...")
    channel.start_consuming()


if __name__ == "__main__":
    start_consumer()
