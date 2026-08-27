import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import smplx
import trimesh
from tqdm import trange
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AI_TAILOR_DIR, MODEL_PATH, UPLOADS_DIR, USER_INPUTS_PATH

# ========== CONFIG ==========
import glob
def _latest(prefix):
    files = sorted(glob.glob(os.path.join(UPLOADS_DIR, f'*_{prefix}.*')))
    return files[-1] if files else os.path.join(AI_TAILOR_DIR, 'dataset', f"female{'_' + prefix if prefix != 'front' else ''}.png")
IMAGES = {
    'front': _latest('front'),
    'left':  _latest('left'),
    'right': _latest('right'),
}
print(f"[INFO] Using images: {IMAGES}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BETAS = 30
LR = 3e-2

print(f"[INFO] Using device: {DEVICE}")

# ========== HANDLE NON-SQUARE IMAGES ==========
def normalize_image_aspect(img):
    h, w = img.shape[:2]
    return img, (h, w)

# ========== KEYPOINT EXTRACTION ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

all_keypoints = []
all_view_names = []
all_confidences = []
all_images_rgb = []
all_segmentation_masks = []

for view_name, img_path in IMAGES.items():
    print(f"\n[INFO] Processing {view_name} view: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not load {img_path}, skipping...")
        continue

    img_rgb, (h, w) = normalize_image_aspect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    aspect_ratio = w / h
    print(f"[INFO] Image size: {w}x{h}, aspect ratio: {aspect_ratio:.2f}")

    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        print(f"[WARNING] No pose detected in {view_name}, skipping...")
        continue

    if results.segmentation_mask is not None:
        all_segmentation_masks.append((results.segmentation_mask, img_rgb, view_name))

    if getattr(results, "pose_world_landmarks", None) is not None:
        landmarks = results.pose_world_landmarks.landmark
        kps = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        confidences = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
        # MediaPipe world landmarks: y is vertical, already metric-ish (normalized by torso)
        # Scale so that the vertical span matches a plausible human
        vert_span = np.max(kps[:, 1]) - np.min(kps[:, 1])
        scale_factor = 1.7 / (vert_span + 1e-6)
        kps = kps * scale_factor
    else:
        landmarks = results.pose_landmarks.landmark
        kps = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
        kps[:, 0] = kps[:, 0] / w * aspect_ratio
        kps[:, 1] = kps[:, 1] / h
        kps[:, 2] = kps[:, 2] / h
        kps = kps * 1.7
        confidences = np.array([lm.visibility for lm in landmarks], dtype=np.float32)

    all_keypoints.append(kps)
    all_view_names.append(view_name)
    all_confidences.append(confidences)
    all_images_rgb.append(img_rgb)
    print(f"[INFO] Extracted {len(kps)} keypoints from {view_name}, avg confidence: {confidences.mean():.3f}")

if len(all_keypoints) == 0:
    raise ValueError("No valid poses detected in any image!")

# ========== FIX 1: REAL-WORLD SCALE CALIBRATION ==========
def estimate_real_height_m(kps):
    """
    Estimate real height using hip-to-ankle (leg length = ~53% of height).
    Falls back to shoulder-to-ankle if hips are low confidence.
    """
    left_hip    = kps[23]
    left_ankle  = kps[27]
    right_hip   = kps[24]
    right_ankle = kps[28]

    leg_l = np.linalg.norm(left_ankle  - left_hip)
    leg_r = np.linalg.norm(right_ankle - right_hip)
    avg_leg = (leg_l + leg_r) / 2.0

    # Hip-to-ankle ≈ 53% of total height (standard anthropometry)
    estimated_height = avg_leg / 0.53
    return estimated_height

# Use front view for height estimation
front_kps = all_keypoints[0]
estimated_height_m = estimate_real_height_m(front_kps)
print(f"\n[INFO] Estimated height from keypoints: {estimated_height_m * 100:.1f} cm")

# Ask user for actual height
_env_height = os.environ.get('USER_HEIGHT_CM')
if _env_height:
    try:
        estimated_height_m = float(_env_height) / 100.0
        print(f"[INFO] Using height from env: {estimated_height_m * 100:.1f} cm")
    except ValueError:
        print(f"[WARNING] Invalid USER_HEIGHT_CM env. Using estimated: {estimated_height_m * 100:.1f} cm")
else:
    try:
        user_height_input = input("[INPUT] Enter your actual height in cm (or press Enter to use estimated): ").strip()
        if user_height_input:
            estimated_height_m = float(user_height_input) / 100.0
        print(f"[INFO] Using height: {estimated_height_m * 100:.1f} cm")
    except ValueError:
        print(f"[WARNING] Invalid input. Using estimated: {estimated_height_m * 100:.1f} cm")
# Rescale ALL views so that front view height estimate matches
for i in range(len(all_keypoints)):
    current_height = estimate_real_height_m(all_keypoints[i])
    if current_height > 0.1:
        all_keypoints[i] = all_keypoints[i] * (estimated_height_m / (current_height + 1e-8))

# ========== FIX 2: GENDER DETECTION ==========
def estimate_gender(kps):
    """
    Shoulder width / hip width ratio:
    > 1.10 → male, < 1.02 → female, else neutral
    """
    shoulder_w = np.linalg.norm(kps[11] - kps[12])
    hip_w      = np.linalg.norm(kps[23] - kps[24])
    ratio = shoulder_w / (hip_w + 1e-8)
    print(f"[INFO] Shoulder/hip ratio: {ratio:.3f}")
    if ratio > 1.10:
        return 'male'
    elif ratio < 1.02:
        return 'female'
    else:
        return 'neutral'

detected_gender = estimate_gender(all_keypoints[0])
print(f"[INFO] Detected gender: {detected_gender}")

# Ask user to confirm or override gender
_env_gender = os.environ.get('USER_GENDER', '').strip().lower()
if _env_gender in ['male', 'female', 'neutral']:
    detected_gender = _env_gender
    print(f"[INFO] Using gender from env: {detected_gender}")
else:
    try:
        user_gender_input = input("[INPUT] Enter gender (male/female/neutral) or press Enter to use detected: ").strip().lower()
        if user_gender_input in ['male', 'female', 'neutral']:
            detected_gender = user_gender_input
        print(f"[INFO] Using gender: {detected_gender}")
    except ValueError:
        print(f"[WARNING] Invalid input. Using detected gender: {detected_gender}")
# ========== SMPL-X JOINT MAPPING ==========
mp_to_smpl = {
    0:  15,   # nose  -> head
    11: 16,   # left  shoulder
    12: 17,   # right shoulder
    13: 18,   # left  elbow
    14: 19,   # right elbow
    15: 20,   # left  wrist
    16: 21,   # right wrist
    23: 1,    # left  hip
    24: 2,    # right hip
    25: 4,    # left  knee
    26: 5,    # right knee
    27: 7,    # left  ankle
    28: 8,    # right ankle
}
mp_indices   = np.array(list(mp_to_smpl.keys()),   dtype=int)
smpl_indices = np.array(list(mp_to_smpl.values()), dtype=int)

# ========== PROCRUSTES ALIGNMENT ==========
def procrustes_align(source, target, weights=None):
    """
    Align source (N,3) to target (N,3) via weighted Procrustes.
    Returns aligned source array of same shape.
    """
    if weights is None:
        weights = np.ones(len(source))

    weights = weights / (weights.sum() + 1e-8)
    source_center = (source.T @ weights)
    target_center = (target.T @ weights)

    source_centered = source - source_center
    target_centered = target - target_center

    W   = np.diag(weights)
    H   = source_centered.T @ W @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T

    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    # S is 1D (singular values), sum instead of trace
    source_var = np.sum(weights[:, None] * source_centered ** 2)
    scale = np.sum(S) / (source_var + 1e-8)

    aligned = scale * (source_centered @ R_mat.T) + target_center
    return aligned, R_mat, scale, target_center

# ========== FIX 5: AXIS-SPECIFIC WEIGHTS PER VIEW ==========
def get_axis_weights(view_name):
    """
    Front view: X and Y reliable, Z (depth) unreliable.
    Side views: Z (depth after rotation) reliable, X unreliable.
    """
    if 'front' in view_name.lower():
        return torch.tensor([1.0, 1.0, 0.2], dtype=torch.float32, device=DEVICE)
    elif 'left' in view_name.lower() or 'right' in view_name.lower():
        return torch.tensor([0.2, 1.0, 1.0], dtype=torch.float32, device=DEVICE)
    return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=DEVICE)

# ========== PREPARE TARGETS ==========
all_targets    = []
all_weights    = []
all_axis_weights = []

if len(all_keypoints) > 1:
    print("\n[INFO] Aligning multiple views using Procrustes analysis...")
    ref_kps  = all_keypoints[0].copy()
    ref_conf = all_confidences[0].copy()

    ref_root = (ref_kps[23] + ref_kps[24]) / 2.0
    ref_kps_centered  = ref_kps - ref_root
    ref_kps_selected  = ref_kps_centered[mp_indices]
    ref_conf_selected = ref_conf[mp_indices]

for view_idx, (kps, conf, view_name) in enumerate(zip(all_keypoints, all_confidences, all_view_names)):
    root = (kps[23] + kps[24]) / 2.0
    kps_centered = kps - root   # full 33-keypoint array

    # Apply view rotation BEFORE subsetting
    if 'left' in view_name.lower():
        rot = R.from_euler('y', -90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied -90° Y rotation for left view")
    elif 'right' in view_name.lower():
        rot = R.from_euler('y', 90, degrees=True).as_matrix()
        kps_centered = (rot @ kps_centered.T).T
        print(f"[INFO] Applied +90° Y rotation for right view")

    # Subset AFTER rotation
    kps_selected  = kps_centered[mp_indices]   # (13, 3)
    conf_selected = conf[mp_indices].copy()     # (13,)

    # Align non-reference views to front view
    if len(all_keypoints) > 1 and view_idx != 0:
        kps_selected, _, _, _ = procrustes_align(
            kps_selected,
            ref_kps_selected,
            weights=conf_selected * ref_conf_selected
        )
        print(f"[INFO] Aligned {view_name} view to reference using Procrustes")

    # Boost confidence for the more visible side in side views
    if 'left' in view_name.lower():
        left_mask = np.isin(mp_indices, [11, 13, 15, 23, 25, 27])
        conf_selected[left_mask] *= 1.5
    elif 'right' in view_name.lower():
        right_mask = np.isin(mp_indices, [12, 14, 16, 24, 26, 28])
        conf_selected[right_mask] *= 1.5

    if len(all_keypoints) == 1 and 'front' in view_name.lower():
        front_visible = np.isin(mp_indices, [11, 12, 13, 14, 25, 26])
        conf_selected[front_visible] *= 1.3
        depth_uncertain = np.isin(mp_indices, [15, 16, 27, 28])
        conf_selected[depth_uncertain] *= 0.7

    conf_selected = np.clip(conf_selected, 0, 1)

    all_targets.append(torch.tensor(kps_selected, dtype=torch.float32, device=DEVICE))
    all_weights.append(torch.tensor(conf_selected, dtype=torch.float32, device=DEVICE))
    all_axis_weights.append(get_axis_weights(view_name))

print(f"\n[INFO] Using {len(all_targets)} view(s) for optimization")

# ========== KEYPOINT ALIGNMENT VISUALIZATION ==========
print("[INFO] Saving keypoint alignment visualization...")
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b']
for i, (target, name) in enumerate(zip(all_targets, all_view_names)):
    pts = target.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
               s=40, c=colors[i % len(colors)], label=name, alpha=0.8)
ax.set_title("Aligned 3D Keypoints")
ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
ax.legend()
ax.view_init(elev=20, azim=70)
plt.tight_layout()
plt.savefig("keypoint_alignment.png", dpi=150, bbox_inches='tight')
plt.close()
print("[INFO] Saved keypoint_alignment.png")

# ========== SHAPE ESTIMATION FROM SEGMENTATION ==========
body_volume_estimate = 1.0
body_width_ratio     = 1.0

if len(all_segmentation_masks) > 0:
    print("\n[INFO] Estimating body shape from segmentation masks...")
    # Use front view mask preferentially
    front_masks = [(m, img, vn) for m, img, vn in all_segmentation_masks if 'front' in vn.lower()]
    masks_to_use = front_masks if front_masks else all_segmentation_masks

    volume_estimates = []
    width_estimates  = []
    for mask, img_rgb, vn in masks_to_use:
        body_area  = np.sum(mask > 0.5)
        total_area = mask.shape[0] * mask.shape[1]
        body_ratio = body_area / total_area

        mask_binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            aspect = w / (h + 1e-6)
            volume_estimates.append(np.clip(body_ratio / 0.15, 0.8, 1.3))
            width_estimates.append(np.clip(aspect / 0.4, 0.7, 1.5))
            print(f"[INFO] [{vn}] body area ratio: {body_ratio:.3f}, aspect: {aspect:.3f}")

    if volume_estimates:
        body_volume_estimate = float(np.mean(volume_estimates))
        body_width_ratio     = float(np.mean(width_estimates))
        print(f"[INFO] Body volume estimate: {body_volume_estimate:.3f}, width ratio: {body_width_ratio:.3f}")

# ========== LOAD SMPL-X MODEL ==========
smplx_model = smplx.create(
    model_path=MODEL_PATH,
    model_type='smplx',
    gender=detected_gender.upper(),
    num_betas=NUM_BETAS,
    use_face_contour=False,
    ext='npz'
).to(DEVICE)

# ========== FIX 6: COMPUTE FIXED GLOBAL SCALE ==========
with torch.no_grad():
    dummy_out = smplx_model(
        betas=torch.zeros(1, NUM_BETAS, device=DEVICE),
        body_pose=torch.zeros(1, 63, device=DEVICE),
        global_orient=torch.zeros(1, 3, device=DEVICE),
    )
    smpl_j = dummy_out.joints[0]
    smpl_rest_height = (smpl_j[:, 1].max() - smpl_j[:, 1].min()).item()

GLOBAL_SCALE = estimated_height_m / (smpl_rest_height + 1e-8)
print(f"[INFO] Fixed global scale: {GLOBAL_SCALE:.4f}  (target height: {estimated_height_m*100:.1f} cm)")

# ========== FIX 3: SEGMENT LENGTH LOSS ==========
def segment_length_loss(joints):
    """
    Symmetry and anthropometric ratio losses on key limb segments.
    joints: (J, 3) tensor
    """
    def seg_len(j1, j2):
        return torch.norm(joints[j1] - joints[j2])

    upper_arm_l = seg_len(16, 18)
    upper_arm_r = seg_len(17, 19)
    forearm_l   = seg_len(18, 20)
    forearm_r   = seg_len(19, 21)
    thigh_l     = seg_len(1,  4)
    thigh_r     = seg_len(2,  5)
    shin_l      = seg_len(4,  7)
    shin_r      = seg_len(5,  8)

    # Left-right symmetry
    sym_loss  = (upper_arm_l - upper_arm_r) ** 2
    sym_loss += (forearm_l   - forearm_r)   ** 2
    sym_loss += (thigh_l     - thigh_r)     ** 2
    sym_loss += (shin_l      - shin_r)      ** 2

    # Anthropometric ratios: thigh ≈ shin
    ratio_loss  = (thigh_l - shin_l) ** 2
    ratio_loss += (thigh_r - shin_r) ** 2

    # Upper arm ≈ forearm
    ratio_loss += (upper_arm_l - forearm_l) ** 2
    ratio_loss += (upper_arm_r - forearm_r) ** 2

    return sym_loss + 0.5 * ratio_loss

# ========== FIX 4: INITIALIZE BETAS ==========
initial_betas = torch.zeros([1, NUM_BETAS], dtype=torch.float32, device=DEVICE)
initial_betas[0, 0] = (body_volume_estimate - 1.0) * 2.0
initial_betas[0, 2] = (body_width_ratio - 1.0) * 1.5
betas = initial_betas.clone().requires_grad_(True)

body_pose     = torch.zeros([1, 21 * 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3],      dtype=torch.float32, device=DEVICE, requires_grad=True)
transl        = torch.zeros([1, 3],      dtype=torch.float32, device=DEVICE, requires_grad=True)

spine_joints = [0, 3, 6, 9, 12, 15]

# FIX 4: Weighted beta regularization (higher betas penalized more)
beta_weights = torch.linspace(1.0, 3.0, NUM_BETAS, device=DEVICE)

def clamp_betas():
    with torch.no_grad():
        betas[:, :5].clamp_(-2.5, 2.5)
        betas[:, 5:].clamp_(-1.5, 1.5)

def joint_loss(smpl_joints, target, weight, axis_w, view_name):
    """
    Compute keypoint loss using fixed global scale and per-axis weights.
    """
    # FIX 6: use fixed scale, no per-iteration re-estimation
    diff = (smpl_joints * GLOBAL_SCALE - target) ** 2
    # FIX 5: axis-specific weighting
    diff = diff * axis_w.unsqueeze(0)
    # Keypoint confidence weighting
    loss = (diff * weight.unsqueeze(1)).mean()
    return loss

# ========== STAGE 1: SHAPE + GLOBAL ORIENTATION ==========
print("\n[INFO] Stage 1: Optimizing shape and global orientation...")
opt1 = torch.optim.Adam([betas, global_orient], lr=LR)

for it in trange(150, desc="Stage 1"):
    opt1.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    smpl_root   = out.joints[0, 0, :]
    smpl_joints = out.joints[0, smpl_indices, :] - smpl_root

    total_loss = 0.0
    for target, weight, axis_w, vn in zip(all_targets, all_weights, all_axis_weights, all_view_names):
        total_loss += joint_loss(smpl_joints, target, weight, axis_w, vn)

    loss_beta = 5e-4 * torch.mean(beta_weights * betas[0] ** 2)
    loss_seg  = 1e-2 * segment_length_loss(out.joints[0])
    loss      = total_loss / len(all_targets) + loss_beta + loss_seg

    loss.backward()
    opt1.step()
    clamp_betas()

    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}, beta[0]={betas[0,0].item():.3f}")

# ========== STAGE 2: LIMBS (SPINE LOCKED) ==========
print("\n[INFO] Stage 2: Optimizing limb poses (spine locked)...")
opt2 = torch.optim.Adam([body_pose, betas, global_orient], lr=LR * 0.5)

for it in trange(250, desc="Stage 2"):
    opt2.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    smpl_root   = out.joints[0, 0, :]
    smpl_joints = out.joints[0, smpl_indices, :] - smpl_root

    total_loss = 0.0
    for target, weight, axis_w, vn in zip(all_targets, all_weights, all_axis_weights, all_view_names):
        total_loss += joint_loss(smpl_joints, target, weight, axis_w, vn)

    loss_spine = 5e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose  = 5e-4 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta  = 3e-4 * torch.mean(beta_weights * betas[0] ** 2)
    loss_seg   = 1e-2 * segment_length_loss(out.joints[0])
    loss       = total_loss / len(all_targets) + loss_spine + loss_pose + loss_beta + loss_seg

    loss.backward()
    opt2.step()

    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.2, 0.2)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
    clamp_betas()

    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}")

# ========== STAGE 3: FINE-TUNE ALL ==========
print("\n[INFO] Stage 3: Fine-tuning all parameters...")
opt3 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.2)

best_loss   = float('inf')   # avoid deprecated np.infty
best_params = None

for it in trange(200, desc="Stage 3"):
    opt3.zero_grad()
    out = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    smpl_root   = out.joints[0, 0, :]
    smpl_joints = out.joints[0, smpl_indices, :] - smpl_root

    total_loss = 0.0
    for target, weight, axis_w, vn in zip(all_targets, all_weights, all_axis_weights, all_view_names):
        total_loss += joint_loss(smpl_joints, target, weight, axis_w, vn)

    # Spine straightness
    spine_coords   = out.joints[0, spine_joints, :]
    spine_dirs     = spine_coords[1:] - spine_coords[:-1]
    spine_dirs_n   = spine_dirs / (torch.norm(spine_dirs, dim=1, keepdim=True) + 1e-8)
    loss_spine_align = 2e-2 * torch.mean(
        (1 - torch.sum(spine_dirs_n[:-1] * spine_dirs_n[1:], dim=1)) ** 2
    )
    loss_spine_pose = 3e-2 * torch.sum(body_pose[0, :12] ** 2)
    loss_pose       = 1e-3 * torch.sum(body_pose[0, 12:] ** 2)
    loss_beta       = 3e-4 * torch.mean(beta_weights * betas[0] ** 2)
    loss_seg        = 1e-2 * segment_length_loss(out.joints[0])

    loss = (total_loss / len(all_targets)
            + loss_spine_align + loss_spine_pose
            + loss_pose + loss_beta + loss_seg)

    loss.backward()
    opt3.step()

    with torch.no_grad():
        body_pose[0, :12].clamp_(-0.3, 0.3)
        body_pose[0, 12:].clamp_(-np.pi, np.pi)
        global_orient.clamp_(-np.pi, np.pi)
    clamp_betas()

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_params = {
            'betas':        betas.detach().clone(),
            'body_pose':    body_pose.detach().clone(),
            'global_orient': global_orient.detach().clone(),
            'transl':       transl.detach().clone(),
        }

    if (it + 1) % 50 == 0:
        print(f"  [ITER {it+1:03d}] loss={loss.item():.6f}")

print(f"\n[INFO] Optimization complete. Best loss: {best_loss:.6f}")
print(f"[INFO] Final betas[:5]: {best_params['betas'][0, :5].cpu().numpy()}")

# ========== GENERATE FINAL MESH ==========
final_out = smplx_model(
    betas=best_params['betas'],
    body_pose=best_params['body_pose'],
    global_orient=best_params['global_orient'],
    transl=best_params['transl'],
)

verts = final_out.vertices[0].cpu().detach().numpy()
faces = smplx_model.faces

# Center and fix orientation
verts_centered = verts - verts.mean(axis=0)
rotation_fix   = R.from_euler('x', 180, degrees=True).as_matrix()
verts_centered = (rotation_fix @ verts_centered.T).T

# Apply global scale so mesh is in real-world metres
verts_scaled = verts_centered * GLOBAL_SCALE

mesh_obj = trimesh.Trimesh(verts_scaled, faces)
mesh_obj.export("fitted_smplx_mesh.obj")
print("[INFO] Exported fitted_smplx_mesh.obj")

mesh_ply = trimesh.Trimesh(verts_scaled, faces)
mesh_ply.visual.vertex_colors = [200, 200, 230, 255]
mesh_ply.export("fitted_smplx_mesh_colored.ply")
print("[INFO] Exported fitted_smplx_mesh_colored.ply")
# Ask for age
_env_age = os.environ.get('USER_AGE')
if _env_age:
    try:
        age_years = float(_env_age)
        print(f"[INFO] Using age from env: {age_years} years")
    except ValueError:
        age_years = 25.0
        print(f"[WARNING] Invalid USER_AGE env. Using default: {age_years}")
else:
    try:
        user_age_input = input("[INPUT] Enter subject age in years: ").strip()
        age_years = float(user_age_input) if user_age_input else 25.0
        print(f"[INFO] Using age: {age_years} years")
    except ValueError:
        age_years = 25.0
        print(f"[WARNING] Invalid age input. Using default: {age_years}")

# Save gender and age for downstream pipeline
# Save gender, age and front image path for downstream pipeline
import json
import os

front_image_path = os.path.abspath(IMAGES['front'])

with open(USER_INPUTS_PATH, "w") as f:
    json.dump({
        "gender":           detected_gender,
        "age_years":        age_years,
        "front_image_path": front_image_path
    }, f, indent=2)
print("[INFO] Saved user_inputs.json")



# ========== SUMMARY ==========
print("\n" + "="*50)
print("[INFO] Pipeline complete!")
print(f"  Views used        : {len(all_targets)}")
print(f"  Gender   : {detected_gender}")
print(f"  Height       : {estimated_height_m*100:.1f} cm")
print(f"  Global scale      : {GLOBAL_SCALE:.4f}")
print(f"  Final betas[:5]   : {best_params['betas'][0, :5].cpu().numpy()}")
print(f"  Body volume factor: {body_volume_estimate:.3f}")
print(f"  Body width ratio  : {body_width_ratio:.3f}")
print("="*50)