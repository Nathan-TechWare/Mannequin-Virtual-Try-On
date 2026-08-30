import os

# ── Edit these two lines for your machine ────────────────────
AI_TAILOR_DIR = r'C:\Users\Adminstrator\Downloads\AI-Tailor (1)\AI-Tailor\full_pipeline'
SMPL_DIR      = r'C:\Users\Adminstrator\Downloads\SMPL-Anthropometry\SMPL-Anthropometry'

# ── These are derived automatically — do not edit ────────────
ANNY_DIR          = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR       = os.path.join(AI_TAILOR_DIR, 'uploads')
MEASUREMENTS_PATH = os.path.join(SMPL_DIR, 'measurements.json')
FACE_PARAMS_PATH  = os.path.join(AI_TAILOR_DIR, 'face_params.json')
