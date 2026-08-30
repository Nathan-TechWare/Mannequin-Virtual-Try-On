# Mannequin-Virtual-Try-On

A three-screen web application that takes photos of a person, extracts body
measurements, and generates a personalised 3D body mesh for virtual try-on.

The three processing stages run as **independent services connected by a
RabbitMQ message queue** — the web app hands a job to the queue and polls for
the result, rather than running the stages inline.

---

## How it works

1. **Screen 1 — Upload:** User uploads three photos (front, left side, right
   side) and enters height, weight, gender, and age.
2. **Screen 2 — Measurements:** The extracted body measurements are displayed
   (read-only).
3. **Screen 3 — 3D Mesh:** A parametric 3D body mesh is rendered in the browser
   with fine-tuning sliders. This is the only step that runs the Anny model.

### The pipeline (queue-based)

Each stage is its own long-running process. They communicate through a RabbitMQ
`pipeline` topic exchange — a stage consumes a message, does its work, and
publishes the next message for the stage downstream:

```
Browser (Screen 1)
   │  POST /process_upload
   ▼
anny_app (app.py, Flask :5000)
   │  POST /upload
   ▼
ai_tailor API (server.py, FastAPI :8000)
   │  publishes  →  ai_tailor_queue
   ▼
ai_tailor_consumer.py     MediaPipe pose + SMPL-X fit → mesh.obj
   │  publishes  mesh.ready  →  smpl_anthro_queue
   ▼
smpl_consumer.py          measures the mesh → measurement.json
   │  publishes  measurements.ready  →  anny_queue
   ▼
anny_consumer.py          writes measurements (stamped with job_id) to disk
   │
   ▼
Screen 1 polls /job_status/<job_id> → redirects to Screen 2 when ready
```

- **AI-Tailor** — estimates body shape from photos using MediaPipe pose
  detection and SMPL-X optimisation
- **SMPL-Anthropometry** — extracts body measurements from the SMPL-X mesh
- **Anny** — maps measurements to a parametric body model and serves the web UI

Because the stages are decoupled by the queue, a job submitted while a consumer
is momentarily down simply waits in that consumer's queue until it comes back
up.

---

## Requirements

- Python 3.11
- Git
- Docker Desktop (for RabbitMQ)

> **Environments:** each of the three folders (`ai_tailor`,
> `smpl_anthropometry`, `anny_app`) has its **own** virtual environment. They
> are set up separately in the steps below.

---

## Project structure

```
Mannequin-Virtual-Try-On/
  ai_tailor/                    ← AI-Tailor pipeline
    server.py                   ← FastAPI upload API (port 8000)
    ai_tailor_consumer.py       ← queue consumer: photos → SMPL-X mesh
    config.py
    requirements_pipeline.txt
    .env                        ← JOBS_DIR, SMPLX_MODEL_PATH, RABBITMQ_HOST
    models/                     ← SMPL-X model files go here (see Step 4)
    uploads/                    ← per-job folders, created automatically

  smpl_anthropometry/           ← SMPL-Anthropometry pipeline
    smpl_consumer.py            ← queue consumer: mesh → measurements
    measure.py
    measurement_definitions.py
    landmark_definitions.py
    joint_definitions.py
    config.py
    requirements_pipeline.txt

  anny_app/                     ← Anny web application
    app.py                      ← Flask server + UI (port 5000)
    anny_consumer.py            ← queue consumer: measurements → disk
    config.py
    requirements_anny.txt
    static/                     ← 3D viewer, height picker, Three.js, overlays
    templates/
      screen1_upload.html       ← Photo upload form (Screen 1)
      screen2_measurements.html ← Measurements display (Screen 2)

  infra/
    rabbitmq/
      rabbitmq-setup.py         ← declares the exchange + queue bindings
    terraform/                  ← cloud resources (not used yet)
```

---

## Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/Nathan-TechWare/Mannequin-Virtual-Try-On.git
cd Mannequin-Virtual-Try-On
```

---

### Step 2 — Edit the config files

Each folder has a `config.py`. Set the paths to match your machine — edit the
lines marked `# EDIT THIS` in each.

**Windows example:**
```python
AI_TAILOR_DIR = r'C:\Users\YourName\Projects\Mannequin-Virtual-Try-On\ai_tailor'
SMPL_DIR      = r'C:\Users\YourName\Projects\Mannequin-Virtual-Try-On\smpl_anthropometry'
```

**Mac/Linux example:**
```python
AI_TAILOR_DIR = '/Users/yourname/Projects/Mannequin-Virtual-Try-On/ai_tailor'
SMPL_DIR      = '/Users/yourname/Projects/Mannequin-Virtual-Try-On/smpl_anthropometry'
```

Also check `ai_tailor/.env` — it holds the paths the queue uses:
```
JOBS_DIR=C:/Users/YourName/Projects/Mannequin-Virtual-Try-On/ai_tailor/uploads/jobs
SMPLX_MODEL_PATH=C:/Users/YourName/Projects/Mannequin-Virtual-Try-On/ai_tailor/models
RABBITMQ_HOST=localhost
```

`smpl_anthropometry/.env` holds the host name for the Anthropometry queue:
```
RABBITMQ_HOST=localhost
```

---

### Step 3 — Create the three virtual environments

Each folder gets its own `.venv`, using Python 3.11.

```powershell
# AI-Tailor
cd ai_tailor
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install wheel setuptools
pip install -r requirements_pipeline.txt
deactivate
cd ..

# SMPL-Anthropometry
cd smpl_anthropometry
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_pipeline.txt
deactivate
cd ..

# Anny
cd anny_app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_anny.txt
deactivate
cd ..
```

> **chumpy install note:** if `chumpy` fails to build with a `pip`/`bdist_wheel`
> error, install it without build isolation first, then re-run the requirements:
> ```powershell
> pip install wheel setuptools
> pip install --no-build-isolation chumpy==0.70
> pip install -r requirements_pipeline.txt
> ```

> The `anny` package downloads its model weights (~500MB) automatically on
> first run — an internet connection is required the first time.

---

### Step 4 — Download SMPL-X model files

The SMPL-X body model files are required by AI-Tailor but cannot be included in
this repository due to licensing restrictions.

1. Register (free) at: https://smpl-x.is.tue.mpg.de/
2. Download the **SMPL-X** model (the `.npz` files)
3. Place them inside `ai_tailor/models/` so the folder looks like:

```
ai_tailor/models/
  smplx/
    SMPLX_MALE.npz
    SMPLX_FEMALE.npz
    SMPLX_NEUTRAL.npz
```

4. Place them inside `smpl_anthropometry/data/` so the folder looks like:

```
smpl_anthropometry/data/
  smpl/
    smpl_body_parts_2_faces.json
  smplx/
    SMPLX_MALE.pkl
    SMPLX_FEMALE.pkl
    SMPLX_NEUTRAL.pkl
```

### Step 5 — Set up the RabbitMQ topology

With the broker running (Step 0), run the infrastructure setup script once to
declare the `pipeline` exchange and bind the queues:

```powershell
cd infra\rabbitmq
python rabbitmq-setup.py
cd ..\..
```

This creates the `pipeline` topic exchange and wires the routing keys:
`mesh.ready` → `smpl_anthro_queue`, `measurements.ready` → `anny_queue`. You
should see `[INFO] Exchange and queues set up.` when it completes.

Run this after starting RabbitMQ but before starting the consumers.

---

## Running the application

The app runs as **five processes plus RabbitMQ**. RabbitMQ must be up first;
then start the five in order (later stages should be listening before a job is
submitted). Each process needs its folder's `.venv` activated.

### 0. RabbitMQ (must be up before anything else)

First time only — pull and start the broker:
```powershell
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.13-management
```

After that first run you don't need the command again — the `rabbitmq`
container appears in **Docker Desktop → Containers**, and you can just press the
**play button** next to it to start it back up (same name and settings each
time). `-d` runs it in the background, so there's no terminal to keep open.

- Management UI: http://localhost:15672 (guest / guest)
- Queues to expect once things connect: `ai_tailor_queue`,
  `smpl_anthro_queue`, `anny_queue`

**Topology setup** — the repo has an `infra/rabbitmq/rabbitmq-setup.py` script
that declares the `pipeline` exchange and the queue bindings (`mesh.ready` →
`smpl_anthro_queue`, `measurements.ready` → `anny_queue`). Run it once after the
broker starts to stand up the full topology explicitly:
```powershell
cd infra\rabbitmq
python rabbitmq-setup.py
cd ..\..
```
This isn't strictly required — each consumer also declares its own queue and
binding on startup — but it's the clean way to set everything up in one shot.
(The `infra/terraform/` folder is for cloud resources we'll set up another time
— ignore it for now.)

### 1. AI-Tailor API — `server.py` (port 8000)

```powershell
cd ai_tailor
.\.venv\Scripts\Activate.ps1
uvicorn server:app --reload
```

### 2. AI-Tailor consumer — `ai_tailor_consumer.py`

```powershell
cd ai_tailor
.\.venv\Scripts\Activate.ps1
python ai_tailor_consumer.py
```
Wait for: `waiting for jobs on ai_tailor_queue...`

### 3. SMPL-Anthropometry consumer — `smpl_consumer.py`

```powershell
cd smpl_anthropometry
.\.venv\Scripts\Activate.ps1
python smpl_consumer.py
```
Wait for: `Waiting for mesh.ready messages...`

### 4. Anny consumer — `anny_consumer.py`

```powershell
cd anny_app
.\.venv\Scripts\Activate.ps1
python anny_consumer.py
```
Wait for: `waiting for measurements.ready...`

### 5. Anny web app — `app.py` (port 5000) ← the one you browse to

```powershell
cd anny_app
.\.venv\Scripts\Activate.ps1
python app.py
```
Auto-opens `http://127.0.0.1:5000` (Screen 1). **Start this last.**

---

## First run

1. Open `http://127.0.0.1:5000`
2. Upload three photos — front facing, left side, right side
3. Enter height (feet/inches), weight (kg), gender, and age
4. Click **Continue** — processing takes 1–2 minutes (CPU-based optimisation).
   Screen 1 polls until the job is done, then redirects.
5. Review your measurements on Screen 2
6. Click **Continue to 3D Mesh** to view and fine-tune your avatar

---

## Troubleshooting

**Everything fails with `pika.exceptions.AMQPConnectionError`**
- RabbitMQ isn't running. Start the container (Step 0) and confirm the
  management UI loads at http://localhost:15672.

**Screen 1 hangs on "processing" forever**
- Check that `anny_consumer.py` is actually running and draining `anny_queue`
  (visible in the management UI). That stage stamps `job_id` into the file that
  `/job_status` polls for — if it isn't running, the redirect never fires.
- Check the management UI: if messages are stuck in a queue with no consumer,
  the process for that stage isn't up.

**"AI-Tailor pipeline failed" / job nacked in ai_tailor_consumer**
- Check the SMPL-X model files are in `ai_tailor/models/smplx/`
- Check the paths in `config.py` and `ai_tailor/.env` are correct and absolute

**"We couldn't detect a person in your photo"**
- Use clear, well-lit photos; plain background; full body head-to-toe

**`ModuleNotFoundError` on startup (fastapi / uvicorn / dotenv / etc.)**
- A dependency is missing from that folder's requirements file. Activate the
  folder's `.venv` and `pip install` the named package, then add it to the
  requirements file.

**`chumpy` won't install**
- See the chumpy note under Step 3 (`--no-build-isolation`).

**Anny model weights not downloading**
- Ensure an internet connection on first run; weights cache at `~/.cache/anny/`.

---

## Dependencies overview

| Service | Key packages |
|---------|-------------|
| AI-Tailor (`server.py`, `ai_tailor_consumer.py`) | fastapi, uvicorn, pika, torch, smplx, mediapipe, opencv-python, scipy, trimesh, python-dotenv |
| SMPL-Anthropometry (`smpl_consumer.py`) | pika, trimesh, torch, numpy |
| Anny (`app.py`, `anny_consumer.py`) | flask, pika, requests, torch, trimesh, anny, pillow |

---

## Notes & known limitations

- Processing time is 1–2 minutes per upload on CPU. A CUDA-capable GPU reduces
  this significantly.
- **One job at a time for now.** Results are written to a single shared
  measurements file, so two jobs processed close together will overwrite each
  other (last-write-wins). Fine for solo use; a per-user / per-job storage
  layout (`{user_id}/{job_id}/`) is the planned fix.
- Uploaded photos are saved per-job under `ai_tailor/uploads/` and are never
  deleted automatically.
- Consumers use `requeue=False` on failure, so a bad job is dropped rather than
  retried. A dead-letter queue would be the production fix for keeping failed
  jobs for inspection.
- The 3D mesh viewer requires WebGL. Chrome and Edge are recommended.
