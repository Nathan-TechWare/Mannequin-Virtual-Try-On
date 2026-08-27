import os

# ── Edit these two lines for your machine ────────────────────
AI_TAILOR_DIR = r'C:\Users\USER\Desktop\AI-Tailor\full_pipeline'
SMPL_DIR      = r'C:\Users\USER\Desktop\SMPL-Anthropometry'

# ── These are derived automatically — do not edit ────────────
MODEL_PATH        = os.path.join(AI_TAILOR_DIR, 'models')
UPLOADS_DIR       = os.path.join(AI_TAILOR_DIR, 'uploads')
USER_INPUTS_PATH  = os.path.join(AI_TAILOR_DIR, 'user_inputs.json')
MESH_OBJ_PATH     = os.path.join(AI_TAILOR_DIR, 'fitted_smplx_mesh.obj')