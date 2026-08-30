"""
server.py — AI-Tailor HTTP API

Accepts the 3 user photos (front / left / right), saves them to a local
per-job folder, and publishes a job onto ai_tailor_queue. The actual
MediaPipe + SMPL-X fitting happens in ai_tailor_consumer.py, which reads
the same job folder and publishes mesh.ready onward.

Demo scope: local disk storage (jobs/{job_id}/), no S3, no job-status DB.
Publisher confirms are on so a 200 response actually means "RabbitMQ has
the message", not just "we tried to send it".

Paths are read from a .env sitting next to this file and stored as
ABSOLUTE paths in the queue message, so the consumer resolves them
correctly no matter which folder it launches from.

Run:
    uvicorn server:app --reload
    (and separately: python ai_tailor_consumer.py)
"""

import json
import os
import uuid
import shutil
from pathlib import Path

import pika
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env from the folder this file lives in, regardless of cwd.
load_dotenv(Path(__file__).parent / ".env")

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
JOBS_DIR = os.environ.get("JOBS_DIR", "jobs")
QUEUE_NAME = "ai_tailor_queue"

app = FastAPI(title="AI-Tailor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_channel():
    """Fresh connection + confirm-enabled channel per request.

    Simple and safe for a demo. For production you'd hold a long-lived
    connection, but per-request avoids
    stale-connection surprises when the API sits idle between demos.
    """
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.confirm_delivery()  # publisher confirms
    return connection, channel


def save_upload(job_dir, name, upload):
    # abspath so the path stored in the queue message is absolute -- the
    # consumer reads paths from the message, so this is what makes it
    # launch-directory independent.
    path = os.path.abspath(os.path.join(job_dir, name))
    upload.file.seek(0)                       # ensure we're at the start
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)    # streams the whole file
    return path


@app.post("/upload")
async def upload_photos(
    front: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    gender: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    age: int = Form(...),
):
    job_id = str(uuid.uuid4())
    job_dir = os.path.abspath(os.path.join(JOBS_DIR, job_id))
    os.makedirs(job_dir, exist_ok=True)

    image_paths = {
        "front": save_upload(job_dir, "front.jpg", front),
        "left": save_upload(job_dir, "left.jpg", left),
        "right": save_upload(job_dir, "right.jpg", right),
    }

    payload = {
        "job_id": job_id,
        "image_paths": image_paths,
        "gender": gender,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "age": age,
    }

    connection, channel = get_channel()
    try:
        # With confirm_delivery on, a failed confirm raises
        # pika.exceptions.UnroutableError / NackError instead of silently
        # dropping -- so we only reach the return below on a real ack.
        channel.basic_publish(
            exchange="",  # default exchange routes by queue name
            routing_key=QUEUE_NAME,
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=2),  # persistent
        )
    except pika.exceptions.AMQPError as e:
        raise HTTPException(status_code=503, detail=f"Failed to queue job: {e}")
    finally:
        connection.close()

    return {"job_id": job_id, "status": "queued"}


@app.get("/health")
async def health():
    return {"status": "ok"}
