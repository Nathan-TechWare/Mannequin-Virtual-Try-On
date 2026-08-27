from flask import Flask, jsonify, send_file, send_from_directory, request, render_template, redirect, url_for, Response
import torch
import trimesh
import json
import anny
import numpy as np
import os
import threading
import io
import subprocess
from datetime import datetime
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AI_TAILOR_DIR, SMPL_DIR, UPLOADS_DIR, MEASUREMENTS_PATH, FACE_PARAMS_PATH

# Optional: HEIC support if pillow-heif is installed
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("[INFO] HEIC support enabled")
except ImportError:
    print("[INFO] pillow-heif not installed; HEIC uploads will fail (JPG/PNG still work)")

app = Flask(__name__, static_folder='static', template_folder='templates')

# ── Pipeline paths (external — not modified) ─────────────────
AI_TAILOR_SCRIPT      = 'pipeline_v6.py'
SMPL_SCRIPT           = 'measure_my_mesh.py'
measurements_path     = MEASUREMENTS_PATH
face_params_path      = FACE_PARAMS_PATH

os.makedirs(UPLOADS_DIR, exist_ok=True)

# ── Load measurements (graceful fallback if file missing) ────
def _load_measurements_or_default():
    try:
        with open(measurements_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[INFO] measurements.json not found/invalid ({e}). Using neutral defaults until first upload.")
        return {
            'gender': 'neutral',
            'age_years': 25.0,
            'height_cm': 170.0,
            'weight_kg': None,
            'measurements': {
                'height': 170.0, 'shoulder to crotch height': 65.0,
                'arm left length': 60.0, 'arm right length': 60.0,
                'inside leg height': 75.0, 'shoulder breadth': 42.0,
                'head circumference': 55.0, 'neck circumference': 38.0,
                'chest circumference': 95.0, 'waist circumference': 80.0,
                'hip circumference': 100.0, 'wrist right circumference': 17.0,
                'bicep right circumference': 30.0, 'forearm right circumference': 27.0,
                'thigh left circumference': 55.0, 'calf left circumference': 37.0,
                'ankle left circumference': 22.0,
            }
        }

data = _load_measurements_or_default()

face_params = {}
if os.path.exists(face_params_path):
    with open(face_params_path, 'r') as f:
        face_params = json.load(f)
    print(f'[INFO] Loaded {len(face_params)} face parameters')
else:
    print('[INFO] No face params found — using neutral face')

gender    = data['gender']
age_years = data.get('age_years', 25.0)
m         = data['measurements']

MEASUREMENT_BOUNDS = {
    'height': (140.0, 210.0), 'shoulder to crotch height': (45.0, 90.0),
    'arm left length': (40.0, 80.0), 'arm right length': (40.0, 80.0),
    'inside leg height': (50.0, 100.0), 'shoulder breadth': (25.0, 55.0),
    'head circumference': (40.0, 70.0), 'neck circumference': (25.0, 55.0),
    'chest circumference': (60.0, 140.0), 'waist circumference': (50.0, 130.0),
    'hip circumference': (65.0, 145.0), 'wrist right circumference': (10.0, 25.0),
    'bicep right circumference': (18.0, 55.0), 'forearm right circumference': (18.0, 45.0),
    'thigh left circumference': (35.0, 85.0), 'calf left circumference': (22.0, 55.0),
    'ankle left circumference': (15.0, 35.0),
}

validation_warnings = []
validation_errors   = []

def _run_validation():
    global validation_warnings, validation_errors
    validation_warnings = []
    validation_errors   = []
    for field, (lo, hi) in MEASUREMENT_BOUNDS.items():
        if field not in m:
            validation_warnings.append(f"MISSING: '{field}' not in measurements")
            continue
        val = m[field]
        if val < lo or val > hi:
            validation_errors.append(f"OUT OF RANGE: '{field}' = {val:.2f} cm (expected {lo}–{hi} cm)")

_run_validation()

if validation_warnings:
    print("[WARNING] Measurement validation warnings:")
    for w in validation_warnings: print(f"  {w}")
if validation_errors:
    print("[WARNING] Measurement validation errors — mesh may look unusual:")
    for e in validation_errors: print(f"  {e}")
else:
    print("[INFO] All measurements passed validation.")

# ── Mapping functions ─────────────────────────────────────────
def clamp01(x): return max(0.0, min(1.0, x))
def map_height(h): return clamp01((h - 140.0) / (200.0 - 140.0))
def map_age(a): return clamp01(0.5 + (a - 18.0) * (0.5 / 52.0))

def map_weight_from_kg(kg):
    """Direct kg -> phenotype 0-1. 40 kg = 0.0, 120 kg = 1.0. Preferred when user provides weight."""
    return clamp01((float(kg) - 40.0) / 80.0)

def map_weight(chest, waist, hip):
    """Fallback: infer weight phenotype from circumferences (used when user kg is unavailable)."""
    return clamp01(((chest + waist + hip) / 3.0 - 60.0) / 40.0)

def map_proportions_torso(s2c, leg):
    return clamp01((s2c / (s2c + leg) - 0.4) / 0.2)

def map_muscle(bicep, thigh, weight_p):
    b = clamp01((bicep - 24.0) / 18.0)
    t = clamp01((thigh - 40.0) / 30.0)
    return clamp01((b + t) / 2.0 - 0.4 * weight_p + 0.4)

def incr(val, lo, hi):
    mid = (lo + hi) / 2.0
    return max(-1.0, min(1.0, (val - mid) / (hi - lo) * 2.0))

def _recompute_initial_values():
    global weight_p, proportions_p, muscle_p, gender_val, INITIAL_PHENOTYPES, INITIAL_LOCAL

    # Prefer user-supplied kg (accurate) over circumference-inferred estimate
    user_weight_kg = data.get('weight_kg')
    if user_weight_kg is not None:
        weight_p = map_weight_from_kg(user_weight_kg)
    else:
        weight_p = map_weight(m['chest circumference'], m['waist circumference'], m['hip circumference'])

    proportions_p = map_proportions_torso(m['shoulder to crotch height'], m['inside leg height'])
    muscle_p      = map_muscle(m['bicep right circumference'], m['thigh left circumference'], weight_p)
    gender_val    = {'male': 0.0, 'female': 1.0, 'neutral': 0.5}.get(gender.strip().lower(), 0.5)

    INITIAL_PHENOTYPES = {
        'gender': gender_val, 'age': map_age(age_years),
        'height': map_height(data['height_cm']), 'weight': weight_p,
        'muscle': muscle_p, 'proportions': proportions_p,
        'cupsize': 0.5, 'firmness': 0.5,
        'african': 1/3, 'asian': 1/3, 'caucasian': 1/3,
    }

    _avg_arm = (m['arm left length'] + m['arm right length']) / 2.0

    INITIAL_LOCAL = {
        'measure-neck-circ-incr':        incr(m['neck circumference'],           25.0, 55.0),
        'measure-waist-circ-incr':       incr(m['waist circumference'],          50.0, 130.0),
        'measure-hips-circ-incr':        incr(m['hip circumference'],            65.0, 145.0),
        'measure-bust-circ-incr':        incr(m['chest circumference'],          60.0, 140.0),
        'measure-thigh-circ-incr':       incr(m['thigh left circumference'],     35.0, 85.0),
        'measure-calf-circ-incr':        incr(m['calf left circumference'],      22.0, 55.0),
        'measure-upperarm-circ-incr':    incr(m['bicep right circumference'],    18.0, 55.0),
        'measure-shoulder-dist-incr':    incr(m['shoulder breadth'],             25.0, 55.0),
        'measure-wrist-circ-incr':       incr(m['wrist right circumference'],    10.0, 25.0),
        'measure-ankle-circ-incr':       incr(m['ankle left circumference'],     15.0, 35.0),
        'measure-upperarm-length-incr':  incr(_avg_arm * 0.45,                  25.0, 40.0),
        'measure-lowerarm-length-incr':  incr(_avg_arm * 0.55,                  28.0, 42.0),
        'l-lowerarm-fat-incr':           incr(m['forearm right circumference'],  18.0, 38.0),
        'r-lowerarm-fat-incr':           incr(m['forearm right circumference'],  18.0, 38.0),
        'measure-upperleg-height-incr':  incr(m['inside leg height'] * 0.55,    35.0, 55.0),
        'measure-lowerleg-height-incr':  incr(m['inside leg height'] * 0.45,    28.0, 45.0),
        'measure-napetowaist-dist-incr': incr(m['shoulder to crotch height'] * 0.6, 32.0, 50.0),
    }

_recompute_initial_values()

# ── Load Anny once ────────────────────────────────────────────
print("Loading Anny model...")
anny_model = anny.Anny(all_phenotypes=True, local_changes="default", remove_unattached_vertices=True)
anny_model = anny_model.to(dtype=torch.float32, device=torch.device('cpu'))
anny_lock  = threading.Lock()
print("Anny model loaded.")

def run_anny(phenotypes, local_changes):
    with anny_lock:
        pk = {k: torch.tensor([float(v)]) for k, v in phenotypes.items()}
        lk = {k: torch.tensor([float(v)]) for k, v in local_changes.items()}
        out = anny_model(phenotype_kwargs=pk, local_changes_kwargs=lk)
        verts = out['vertices'][0].detach().cpu().numpy()
        faces = anny_model.faces.cpu().numpy()
        return verts, faces

def make_glb(verts, faces):
    verts = verts.copy()
    x = verts[:, 0].copy()
    y = verts[:, 1].copy()
    z = verts[:, 2].copy()
    verts[:, 0] = x
    verts[:, 1] = z
    verts[:, 2] = -y
    verts[:, 1] -= verts[:, 1].min()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=2)
    return mesh.export(file_type='glb')

def _reload_state_from_disk():
    global data, gender, age_years, m
    with open(measurements_path, 'r') as f:
        data = json.load(f)
    gender    = data['gender']
    age_years = data.get('age_years', 25.0)
    m         = data['measurements']
    _run_validation()
    _recompute_initial_values()


# ══════════════════════════════════════════════════════════════
# UI ROUTES (Screens 1, 2, 3)
# ══════════════════════════════════════════════════════════════

@app.route('/')
def screen1_upload():
    return render_template('screen1_upload.html')


@app.route('/measurements')
def screen2_measurements():
    return render_template('screen2_measurements.html')


@app.route('/mesh')
def screen3_mesh():
    """Serves existing Anny mesh viewer with overlay injected (nav bar + slider transforms).
    The original index.html on disk is NOT modified — injection happens per-request."""
    index_path = os.path.join(app.static_folder, 'index.html')
    with open(index_path, 'r', encoding='utf-8') as f:
        html = f.read()
    injection = (
        '<link rel="stylesheet" href="/static/screen3_overlay.css">\n'
        '<script src="/static/screen3_overlay.js" defer></script>\n'
    )
    if '</head>' in html:
        html = html.replace('</head>', injection + '</head>', 1)
    else:
        html = html.replace('</body>', injection + '</body>', 1)
    return Response(html, mimetype='text/html')


@app.route('/api/measurements')
def api_measurements():
    """Kept for backwards compat. Screen 2 uses /initial_values now."""
    return jsonify({
        'gender':       data['gender'],
        'age_years':    data.get('age_years', 25.0),
        'height_cm':    data['height_cm'],
        'weight_kg':    data.get('weight_kg'),
        'measurements': data['measurements'],
        'warnings':     validation_errors,
    })


def _friendly_error(pipeline_name, stderr_tail):
    print(f"[ERROR] {pipeline_name} failed:\n{stderr_tail}")
    if pipeline_name == 'AI-Tailor':
        return ("We couldn't process your photos. Please try clearer, well-lit photos where "
                "you're standing straight and facing the correct direction (front, left side, right side).")
    if pipeline_name == 'SMPL-Anthropometry':
        return ("We couldn't extract measurements from your photos. This is unusual — please try again "
                "with different photos.")
    return "Something went wrong processing your photos. Please try again."


@app.route('/process_upload', methods=['POST'])
def process_upload():
    try:
        # Read form fields
        height_cm = request.form.get('height_cm', '').strip()
        weight_kg = request.form.get('weight_kg', '').strip()
        gender_v  = request.form.get('gender', '').strip().lower()
        age_v     = request.form.get('age', '').strip()

        if gender_v not in ('male', 'female', 'neutral'):
            return jsonify({'error': 'Please select your gender.'}), 400
        try:
            height_int   = int(float(height_cm))
            age_float    = float(age_v)
            weight_float = float(weight_kg)
        except ValueError:
            return jsonify({'error': 'Height, weight, and age must be valid numbers.'}), 400
        if not (30 <= weight_float <= 200):
            return jsonify({'error': 'Weight must be between 30 and 200 kg.'}), 400
        if not (1 <= age_float <= 120):
            return jsonify({'error': 'Age must be between 1 and 120.'}), 400

        # Validate photos
        missing = [k for k in ('front', 'left', 'right')
                   if k not in request.files or request.files[k].filename == '']
        if missing:
            if len(missing) == 1:
                return jsonify({'error': f'Please upload the {missing[0]} photo.'}), 400
            return jsonify({'error': f'Please upload all photos. Missing: {", ".join(missing)}.'}), 400

        # Save photos with timestamp, convert to PNG
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        for key in ('front', 'left', 'right'):
            file = request.files[key]
            out_path = os.path.join(UPLOADS_DIR, f'{ts}_{key}.png')
            try:
                img = Image.open(file.stream).convert('RGB')
                img.save(out_path, 'PNG')
                print(f"[INFO] Saved upload: {out_path}")
            except Exception as e:
                return jsonify({
                    'error': f'Could not read your {key} photo. Please try a different file (JPG or PNG works best).'
                }), 400

        # Env vars for pipelines
        env = os.environ.copy()
        env['USER_HEIGHT_CM'] = str(height_int)
        env['USER_WEIGHT_KG'] = str(weight_float)
        env['USER_GENDER']    = gender_v
        env['USER_AGE']       = str(age_float)

        # Run AI-Tailor
        print("[INFO] Running AI-Tailor pipeline...")
        result = subprocess.run(['python', AI_TAILOR_SCRIPT],
            cwd=AI_TAILOR_DIR, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return jsonify({'error': _friendly_error('AI-Tailor', result.stderr[-2000:])}), 500
        print("[INFO] AI-Tailor done.")

        # Run SMPL-Anthropometry
        print("[INFO] Running SMPL-Anthropometry pipeline...")
        result = subprocess.run(['python', SMPL_SCRIPT],
            cwd=SMPL_DIR, env=env, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return jsonify({'error': _friendly_error('SMPL-Anthropometry', result.stderr[-2000:])}), 500
        print("[INFO] SMPL-Anthropometry done.")

        _reload_state_from_disk()
        print("[INFO] State reloaded.")

        return jsonify({'status': 'ok', 'redirect': url_for('screen2_measurements')})

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Processing took too long. Please try again.'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Something went wrong. Please try again.'}), 500


# ══════════════════════════════════════════════════════════════
# EXISTING MESH ROUTES (untouched; /initial_values expanded additively)
# ══════════════════════════════════════════════════════════════

@app.route('/initial_values')
def initial_values():
    """Includes age_years, weight_kg additively — existing Screen 3 code that reads
    only phenotypes/local_changes continues to work."""
    return jsonify({
        'phenotypes':    INITIAL_PHENOTYPES,
        'local_changes': INITIAL_LOCAL,
        'gender':        gender,
        'age_years':     age_years,
        'height_cm':     data['height_cm'],
        'weight_kg':     data.get('weight_kg'),
    })


@app.route('/reload_measurements', methods=['POST'])
def reload_measurements():
    _reload_state_from_disk()
    if validation_errors:
        print("[WARNING] Validation errors after reload:")
        for e in validation_errors: print(f"  {e}")
    else:
        print("[INFO] Reloaded measurements — all passed validation.")
    return jsonify({
        'status':        'reloaded',
        'gender':        gender,
        'age_years':     age_years,
        'height_cm':     data['height_cm'],
        'weight_kg':     data.get('weight_kg'),
        'phenotypes':    INITIAL_PHENOTYPES,
        'local_changes': INITIAL_LOCAL,
        'warnings':      validation_errors,
    })


@app.route('/generate', methods=['POST'])
def generate():
    try:
        body          = request.get_json(force=True) or {}
        phenotypes    = body.get('phenotypes',    INITIAL_PHENOTYPES)
        local_changes = body.get('local_changes', INITIAL_LOCAL)
        verts, faces = run_anny(phenotypes, local_changes)
        glb_bytes    = make_glb(verts, faces)
        print(f"[INFO] GLB generated: {len(glb_bytes)} bytes, magic={glb_bytes[:4]}")
        return Response(glb_bytes, mimetype='model/gltf-binary',
            headers={'Content-Disposition': 'inline; filename="body.glb"',
                     'Content-Length': str(len(glb_bytes))})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/export', methods=['POST'])
def export():
    body          = request.get_json()
    phenotypes    = body.get('phenotypes',    INITIAL_PHENOTYPES)
    local_changes = body.get('local_changes', INITIAL_LOCAL)
    verts, faces = run_anny(phenotypes, local_changes)
    verts_exp    = verts.copy()
    verts_exp[:, 1] *= -1
    verts_exp[:, 1] -= verts_exp[:, 1].min()
    mesh     = trimesh.Trimesh(vertices=verts_exp, faces=faces)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_body_{gender}_{int(data['height_cm'])}cm_{ts}.obj"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    mesh.export(filepath)
    return jsonify({'saved': filename})


if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=False, port=5000)
