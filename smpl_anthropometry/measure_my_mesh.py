import trimesh
import numpy as np
import torch
import json
import os
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import USER_INPUTS_PATH, MESH_OBJ_PATH, MEASUREMENTS_PATH

# Read weight from env var (set by Flask UI); fallback to None for standalone runs
_env_weight = os.environ.get('USER_WEIGHT_KG')
weight_kg = float(_env_weight) if _env_weight else None

# ── Read gender from AI-Tailor output ────────────────────────
user_inputs_path = USER_INPUTS_PATH
if os.path.exists(user_inputs_path):
    with open(user_inputs_path, 'r') as f:
        user_inputs = json.load(f)
    gender = user_inputs['gender']
    age_years = user_inputs.get('age_years', 25.0)
    print(f"[INFO] Loaded from user_inputs.json: gender={gender}, age={age_years}")
else:
    print("[WARNING] user_inputs.json not found, falling back to manual input")
    gender = input("Enter gender (male/female/neutral): ").strip().lower()

# ── Load mesh ─────────────────────────────────────────────────
mesh = trimesh.load(MESH_OBJ_PATH)
verts = np.array(mesh.vertices, dtype=np.float32)

# ── Manual height input ───────────────────────────────────────
_env_height = os.environ.get('USER_HEIGHT_CM')
if _env_height:
    actual_height_cm = float(_env_height)
    print(f"[INFO] Using height from env: {actual_height_cm} cm")
else:
    actual_height_cm = float(input("Enter actual height in cm: "))

# ── Apply height correction ───────────────────────────────────
current_height_cm = (verts[:, 1].max() - verts[:, 1].min()) * 100
print(f"[INFO] Mesh height before correction: {current_height_cm:.2f} cm")
correction = actual_height_cm / current_height_cm
verts = verts * correction
print(f"[INFO] Correction factor: {correction:.4f}")
print(f"[INFO] Mesh height after correction: {actual_height_cm:.2f} cm")

# ── Measure ───────────────────────────────────────────────────
verts_tensor = torch.tensor(verts, dtype=torch.float32)
print(f'Loaded mesh with {len(verts_tensor)} vertices')

measurer = MeasureBody('smplx')
measurer.from_verts(verts=verts_tensor)
measurement_names = measurer.all_possible_measurements
measurer.measure(measurement_names)
# ── Neck correction ──────────────────────────────────────────────────────
# SMPLX overestimates male neck due to Adam's apple landmark geometry
# Only corrects when neck exceeds realistic maximum for gender
NECK_EXPECTED = {'male': 39.0, 'female': 34.0, 'neutral': 36.0}
NECK_MAX      = {'male': 43.0, 'female': 38.0, 'neutral': 40.0}

neck_val      = measurer.measurements.get('neck circumference', 0)
neck_expected = NECK_EXPECTED.get(gender, 36.0)
neck_max      = NECK_MAX.get(gender, 40.0)

if neck_val > neck_max:
    correction_factor = neck_expected / neck_val
    corrected_neck    = neck_val * correction_factor
    print(f"[INFO] Neck correction applied: {neck_val:.2f}cm -> {corrected_neck:.2f}cm")
    measurer.measurements['neck circumference'] = corrected_neck
else:
    print(f"[INFO] Neck circumference {neck_val:.2f}cm within expected range — no correction applied")

measurer.label_measurements(STANDARD_LABELS)

print('\n=== Measurements (cm) ===')
for name, value in measurer.measurements.items():
    print(f'{name}: {value:.2f} cm')

print('\n=== Labeled Measurements ===')
for label, value in measurer.labeled_measurements.items():
    print(f'{label}: {value:.2f} cm')

# ── Save to JSON ──────────────────────────────────────────────
output = {
    "height_cm":    actual_height_cm,
    "gender":       gender,
    "age_years":    age_years,
    "weight_kg":    weight_kg,
    "measurements": {k: round(float(v), 2) for k, v in measurer.measurements.items()}
}
with open(MEASUREMENTS_PATH, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n[INFO] Saved measurements to {output_path}")