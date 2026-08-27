# Mannequin-Virtual-Try-On

A three-screen web application that takes photos of a person, extracts body measurements, and generates a personalised 3D body mesh for virtual try-on.

---

## How it works

1. **Screen 1 — Upload:** User uploads three photos (front, left side, right side) and enters height, weight, gender, and age.
2. **Screen 2 — Measurements:** The system processes the photos and displays extracted body measurements.
3. **Screen 3 — 3D Mesh:** A parametric 3D body mesh is rendered in the browser with fine-tuning sliders.

Three pipelines run sequentially behind the scenes:
- **AI-Tailor** — estimates body shape from photos using MediaPipe pose detection and SMPL-X optimisation
- **SMPL-Anthropometry** — extracts body measurements from the SMPL-X mesh
- **Anny** — maps measurements to a parametric body model and serves the web UI

---

## Requirements

- Python 3.11
- Git

> **Note:** AI-Tailor and SMPL-Anthropometry use one Python environment. The Anny web app uses a separate virtual environment. Both are set up in the steps below.

---

## Project structure

After cloning, your folder structure should look like this:

```
Mannequin-Virtual-Try-On/
  ai_tailor/                  ← AI-Tailor pipeline
    pipeline_v6.py
    config.py
    requirements_pipeline.txt
    models/                   ← SMPL-X model files go here (see Step 3)
    uploads/                  ← Created automatically on first run

  smpl_anthropometry/         ← SMPL-Anthropometry pipeline
    measure_my_mesh.py        ← SMPL measurement extraction script
    measure.py
    measurement_definitions.py
    config.py
    requirements_pipeline.txt

  anny_app/                   ← Anny web application
    app.py                    ← Flask server — main entry point, run this
    config.py
    requirements_anny.txt
    static/
      index.html              ← Anny 3D mesh viewer (Screen 3)
      drum-picker.css         ← Height picker styles
      drum-picker.js          ← Height picker logic
      screen3_overlay.css     ← Screen 3 nav bar + slider enhancements
      screen3_overlay.js      ← Screen 3 nav bar + slider enhancements
      three.min.js            ← Three.js 3D renderer
      GLTFLoader.js           ← Three.js GLTF loader
      OrbitControls.js        ← Three.js camera controls
    templates/
      screen1_upload.html     ← Photo upload form (Screen 1)
      screen2_measurements.html ← Measurements display (Screen 2)
```

---

## Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Mannequin-Virtual-Try-On.git
cd Mannequin-Virtual-Try-On
```

---

### Step 2 — Edit the config files

Each pipeline has a `config.py` file. You need to set the paths to match your machine. Open each one and edit the two lines marked `# EDIT THIS`.

**`ai_tailor/config.py`:**
```python
AI_TAILOR_DIR = '/absolute/path/to/Mannequin-Virtual-Try-On/ai_tailor'   # EDIT THIS
SMPL_DIR      = '/absolute/path/to/Mannequin-Virtual-Try-On/smpl_anthropometry'  # EDIT THIS
```

**`smpl_anthropometry/config.py`:**
```python
AI_TAILOR_DIR = '/absolute/path/to/Mannequin-Virtual-Try-On/ai_tailor'   # EDIT THIS
SMPL_DIR      = '/absolute/path/to/Mannequin-Virtual-Try-On/smpl_anthropometry'  # EDIT THIS
```

**`anny_app/config.py`:**
```python
AI_TAILOR_DIR = '/absolute/path/to/Mannequin-Virtual-Try-On/ai_tailor'   # EDIT THIS
SMPL_DIR      = '/absolute/path/to/Mannequin-Virtual-Try-On/smpl_anthropometry'  # EDIT THIS
```

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

> Edit all three config files — one in each folder. They all need the same two path values.

---

### Step 3 — Download SMPL-X model files

The SMPL-X body model files are required by AI-Tailor but cannot be included in this repository due to licensing restrictions.

1. Register (free) at: https://smpl-x.is.tue.mpg.de/
2. Download the **SMPL-X** model (look for the `.npz` files)
3. Place the downloaded files inside `ai_tailor/models/` so the folder looks like:

```
ai_tailor/models/
  smplx/
    SMPLX_MALE.npz
    SMPLX_FEMALE.npz
    SMPLX_NEUTRAL.npz
```

---

### Step 4 — Set up the pipeline environment (AI-Tailor + SMPL-Anthropometry)

This environment is used by both `pipeline_v6.py` and `measure_my_mesh.py`.

**Windows:**
```
python -m pip install -r ai_tailor/requirements_pipeline.txt
```

**Mac/Linux:**
```
pip3 install -r ai_tailor/requirements_pipeline.txt
```

> If you prefer a virtual environment for the pipelines, create one first:
> ```
> python -m venv pipeline_env
> # Windows: pipeline_env\Scripts\activate
> # Mac/Linux: source pipeline_env/bin/activate
> pip install -r ai_tailor/requirements_pipeline.txt
> ```

---

### Step 5 — Set up the Anny environment

The Anny web app requires a separate virtual environment.

**Windows:**
```
cd anny_app
python -m venv anny_env
anny_env\Scripts\activate
pip install -r requirements_anny.txt
```

**Mac/Linux:**
```
cd anny_app
python3 -m venv anny_env
source anny_env/bin/activate
pip install -r requirements_anny.txt
```

> The `anny` package will download its model weights (~500MB) automatically on first run. An internet connection is required the first time.

---

### Step 6 — Create the uploads folder

The uploads folder is not included in the repository. Create it manually:

**Windows:**
```
mkdir ai_tailor\uploads
```

**Mac/Linux:**
```
mkdir ai_tailor/uploads
```

---

## Running the application

### Windows

```
cd anny_app
anny_env\Scripts\activate
python app.py
```

### Mac/Linux

```
cd anny_app
source anny_env/bin/activate
python app.py
```

The browser will open automatically at `http://127.0.0.1:5000`.

> **Important:** The pipeline environment (Step 4) must be on your system PATH so that `app.py` can call `python pipeline_v6.py` and `python measure_my_mesh.py` as subprocesses. If you used a virtual environment for the pipelines, activate it in a separate terminal before starting the app, or adjust the subprocess calls in `app.py` to point to the pipeline environment's Python executable.

---

## First run

1. Open `http://127.0.0.1:5000` in your browser
2. Upload three photos — front facing, left side, right side
3. Enter height (feet/inches), weight (kg), gender, and age
4. Click **Continue** — processing takes 1-2 minutes (CPU-based optimisation)
5. Review your measurements on Screen 2
6. Click **Continue to 3D Mesh** to view and fine-tune your avatar

---

## Troubleshooting

**"AI-Tailor pipeline failed"**
- Check that the SMPL-X model files are in `ai_tailor/models/smplx/`
- Check that the paths in all three `config.py` files are correct and absolute
- Check that the pipeline environment has all dependencies installed

**"We couldn't detect a person in your photo"**
- Use clear, well-lit photos
- Stand against a plain background if possible
- Make sure your full body is visible from head to toe

**"SMPL-Anthropometry pipeline failed"**
- Usually caused by AI-Tailor not completing successfully first
- Check that `ai_tailor/fitted_smplx_mesh.obj` exists after a run

**Anny model weights not downloading**
- Ensure you have an internet connection on first run
- Weights are cached after the first download at `~/.cache/anny/`

**Subprocess pipeline not found (Mac/Linux)**
- The `python` command may not exist — try changing `'python'` to `'python3'` in `app.py`'s subprocess calls

---

## Dependencies overview

| Pipeline | Key packages |
|----------|-------------|
| AI-Tailor | torch, smplx, mediapipe, opencv-python, scipy, trimesh |
| SMPL-Anthropometry | trimesh, torch, numpy |
| Anny web app | flask, torch, trimesh, anny, pillow, warp |

---

## Notes

- Processing time is 1-2 minutes per upload on CPU. A CUDA-capable GPU reduces this significantly.
- Uploaded photos are saved timestamped to `ai_tailor/uploads/` and are never deleted automatically.
- The 3D mesh viewer requires WebGL support in the browser. Chrome and Edge are recommended.
- `measurements.json` in `smpl_anthropometry/` is overwritten on each new upload.
