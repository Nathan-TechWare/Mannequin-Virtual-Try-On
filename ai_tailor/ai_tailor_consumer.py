"""
ai_tailor_consumer.py

Long-running worker that listens on ai_tailor_queue for uploaded-photo
jobs, runs the MediaPipe + SMPL-X fitting, and publishes mesh.ready onto
the shared "pipeline" topic exchange for smpl_consumer.py downstream.

The fitting logic below is the original standalone pipeline script,
refactored into run_fitting(image_paths, gender, height_cm, job_dir) so
it can be driven by queue messages instead of input() prompts:
  - the two input() prompts are gone -- gender + height arrive in the job
  - it no longer mints its own job_id/folder -- server.py already created
    jobs/{job_id}/ and saved the photos there; we fit into that same dir
  - it no longer publishes mesh.ready itself -- the consumer does the
    single publish on the channel it already holds (avoids duplicates)
"""

import json
import os

import cv2
import mediapipe as mp
import numpy as np
import torch
import smplx
import trimesh
from tqdm import trange
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import pika

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ========== CONFIG ==========
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
# JOBS_DIR = "jobs"
JOBS_DIR = os.environ.get("JOBS_DIR", "jobs")
QUEUE_NAME = "ai_tailor_queue"
EXCHANGE_NAME = "pipeline"

# MODEL_PATH = os.environ.get("SMPLX_MODEL_PATH", "full_pipeline/models/")
MODEL_PATH = os.environ.get(
    "SMPLX_MODEL_PATH",
    str(Path(__file__).parent / "models"),
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BETAS = 30
LR = 3e-2

print(f"[INFO] Using device: {DEVICE}")


# ========== SMPL-X JOINT MAPPING (module-level: static) ==========
mp_to_smpl = {
    0: 15, 11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21,
    23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8,
}
mp_indices = np.array(list(mp_to_smpl.keys()), dtype=int)
smpl_indices = np.array(list(mp_to_smpl.values()), dtype=int)


def normalize_image_aspect(img):
    h, w = img.shape[:2]
    return img, (h, w)


def estimate_real_height_m(kps):
    """Estimate real height from hip-to-ankle (leg ~= 53% of height)."""
    left_hip, left_ankle = kps[23], kps[27]
    right_hip, right_ankle = kps[24], kps[28]
    leg_l = np.linalg.norm(left_ankle - left_hip)
    leg_r = np.linalg.norm(right_ankle - right_hip)
    avg_leg = (leg_l + leg_r) / 2.0
    return avg_leg / 0.53


def procrustes_align(source, target, weights=None):
    if weights is None:
        weights = np.ones(len(source))
    weights = weights / (weights.sum() + 1e-8)
    source_center = (source.T @ weights)
    target_center = (target.T @ weights)
    source_centered = source - source_center
    target_centered = target - target_center
    W = np.diag(weights)
    H = source_centered.T @ W @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    source_var = np.sum(weights[:, None] * source_centered ** 2)
    scale = np.sum(S) / (source_var + 1e-8)
    aligned = scale * (source_centered @ R_mat.T) + target_center
    return aligned, R_mat, scale, target_center


def get_axis_weights(view_name):
    if 'front' in view_name.lower():
        return torch.tensor([1.0, 1.0, 0.2], dtype=torch.float32, device=DEVICE)
    elif 'left' in view_name.lower() or 'right' in view_name.lower():
        return torch.tensor([0.2, 1.0, 1.0], dtype=torch.float32, device=DEVICE)
    return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=DEVICE)


def segment_length_loss(joints):
    def seg_len(j1, j2):
        return torch.norm(joints[j1] - joints[j2])
    upper_arm_l, upper_arm_r = seg_len(16, 18), seg_len(17, 19)
    forearm_l, forearm_r = seg_len(18, 20), seg_len(19, 21)
    thigh_l, thigh_r = seg_len(1, 4), seg_len(2, 5)
    shin_l, shin_r = seg_len(4, 7), seg_len(5, 8)
    sym_loss = (upper_arm_l - upper_arm_r) ** 2
    sym_loss += (forearm_l - forearm_r) ** 2
    sym_loss += (thigh_l - thigh_r) ** 2
    sym_loss += (shin_l - shin_r) ** 2
    ratio_loss = (thigh_l - shin_l) ** 2 + (thigh_r - shin_r) ** 2
    ratio_loss += (upper_arm_l - forearm_l) ** 2 + (upper_arm_r - forearm_r) ** 2
    return sym_loss + 0.5 * ratio_loss


def run_fitting(image_paths: dict, gender: str, height_cm: float, job_dir: str,
                 weight_kg: float, age: int) -> dict:
    """
    Run the full MediaPipe + SMPL-X fit for one job.

    image_paths: {"front": path, "left": path, "right": path}
    gender:      "male" | "female"
    height_cm:   real height in cm (may be None -> heuristic estimate)
    job_dir:     the already-created jobs/{job_id}/ folder to write into

    Returns the fields the consumer needs to build the mesh.ready payload:
      {"mesh_path": ..., "gender": ..., "height_cm": ...}
    """
    detected_gender = gender
    user_height_m = (height_cm / 100.0) if height_cm else None

    # ---- keypoint extraction ----
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

    all_keypoints, all_view_names, all_confidences = [], [], []
    all_images_rgb, all_segmentation_masks = [], []

    for view_name, img_path in image_paths.items():
        print(f"[INFO] Processing {view_name} view: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not load {img_path}, skipping...")
            continue

        img_rgb, (h, w) = normalize_image_aspect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        aspect_ratio = w / h
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
            vert_span = np.max(kps[:, 1]) - np.min(kps[:, 1])
            kps = kps * (1.7 / (vert_span + 1e-6))
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
        print(f"[INFO] Extracted {len(kps)} keypoints from {view_name}, "
              f"avg confidence: {confidences.mean():.3f}")

    if len(all_keypoints) == 0:
        raise ValueError("No valid poses detected in any image!")

    # ---- real-world scale ----
    front_kps = all_keypoints[0]
    estimated_height_m = estimate_real_height_m(front_kps)
    print(f"[INFO] Estimated height from keypoints: {estimated_height_m * 100:.1f} cm")

    if user_height_m is not None and (0.5 <= user_height_m <= 2.5):
        target_height_m = user_height_m
        print(f"[INFO] Using user-provided height: {user_height_m * 100:.1f} cm")
    else:
        if user_height_m is not None:
            print(f"[WARNING] user height {user_height_m} implausible; using heuristic.")
        target_height_m = estimated_height_m

    for i in range(len(all_keypoints)):
        current_height = estimate_real_height_m(all_keypoints[i])
        if current_height > 0.1:
            all_keypoints[i] = all_keypoints[i] * (target_height_m / (current_height + 1e-8))

    # ---- prepare targets (align views) ----
    all_targets, all_weights, all_axis_weights = [], [], []

    ref_kps_selected = None
    ref_conf_selected = None
    if len(all_keypoints) > 1:
        ref_kps = all_keypoints[0].copy()
        ref_conf = all_confidences[0].copy()
        ref_root = (ref_kps[23] + ref_kps[24]) / 2.0
        ref_kps_selected = (ref_kps - ref_root)[mp_indices]
        ref_conf_selected = ref_conf[mp_indices]

    for view_idx, (kps, conf, view_name) in enumerate(
            zip(all_keypoints, all_confidences, all_view_names)):
        root = (kps[23] + kps[24]) / 2.0
        kps_centered = kps - root

        if 'left' in view_name.lower():
            rot = R.from_euler('y', -90, degrees=True).as_matrix()
            kps_centered = (rot @ kps_centered.T).T
        elif 'right' in view_name.lower():
            rot = R.from_euler('y', 90, degrees=True).as_matrix()
            kps_centered = (rot @ kps_centered.T).T

        kps_selected = kps_centered[mp_indices]
        conf_selected = conf[mp_indices].copy()

        if len(all_keypoints) > 1 and view_idx != 0:
            kps_selected, _, _, _ = procrustes_align(
                kps_selected, ref_kps_selected,
                weights=conf_selected * ref_conf_selected)

        if 'left' in view_name.lower():
            left_mask = np.isin(mp_indices, [11, 13, 15, 23, 25, 27])
            conf_selected[left_mask] *= 1.5
        elif 'right' in view_name.lower():
            right_mask = np.isin(mp_indices, [12, 14, 16, 24, 26, 28])
            conf_selected[right_mask] *= 1.5

        if len(all_keypoints) == 1 and 'front' in view_name.lower():
            conf_selected[np.isin(mp_indices, [11, 12, 13, 14, 25, 26])] *= 1.3
            conf_selected[np.isin(mp_indices, [15, 16, 27, 28])] *= 0.7

        conf_selected = np.clip(conf_selected, 0, 1)
        all_targets.append(torch.tensor(kps_selected, dtype=torch.float32, device=DEVICE))
        all_weights.append(torch.tensor(conf_selected, dtype=torch.float32, device=DEVICE))
        all_axis_weights.append(get_axis_weights(view_name))

    print(f"[INFO] Using {len(all_targets)} view(s) for optimization")

    # ---- shape hints from segmentation ----
    body_volume_estimate, body_width_ratio = 1.0, 1.0
    if len(all_segmentation_masks) > 0:
        front_masks = [(m, img, vn) for m, img, vn in all_segmentation_masks
                       if 'front' in vn.lower()]
        masks_to_use = front_masks if front_masks else all_segmentation_masks
        volume_estimates, width_estimates = [], []
        for mask, img_rgb, vn in masks_to_use:
            body_ratio = np.sum(mask > 0.5) / (mask.shape[0] * mask.shape[1])
            mask_binary = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                aspect = w / (h + 1e-6)
                volume_estimates.append(np.clip(body_ratio / 0.15, 0.8, 1.3))
                width_estimates.append(np.clip(aspect / 0.4, 0.7, 1.5))
        if volume_estimates:
            body_volume_estimate = float(np.mean(volume_estimates))
            body_width_ratio = float(np.mean(width_estimates))

    # ---- load SMPL-X ----
    smplx_model = smplx.create(
        model_path=MODEL_PATH, model_type='smplx',
        gender=detected_gender.upper(), num_betas=NUM_BETAS,
        use_face_contour=False, ext='npz').to(DEVICE)

    with torch.no_grad():
        dummy_out = smplx_model(
            betas=torch.zeros(1, NUM_BETAS, device=DEVICE),
            body_pose=torch.zeros(1, 63, device=DEVICE),
            global_orient=torch.zeros(1, 3, device=DEVICE))
        smpl_j = dummy_out.joints[0]
        smpl_rest_height = (smpl_j[:, 1].max() - smpl_j[:, 1].min()).item()
    GLOBAL_SCALE = target_height_m / (smpl_rest_height + 1e-8)
    print(f"[INFO] Fixed global scale: {GLOBAL_SCALE:.4f}")

    # ---- init params ----
    initial_betas = torch.zeros([1, NUM_BETAS], dtype=torch.float32, device=DEVICE)
    initial_betas[0, 0] = (body_volume_estimate - 1.0) * 2.0
    initial_betas[0, 2] = (body_width_ratio - 1.0) * 1.5
    betas = initial_betas.clone().requires_grad_(True)
    body_pose = torch.zeros([1, 63], dtype=torch.float32, device=DEVICE, requires_grad=True)
    global_orient = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
    transl = torch.zeros([1, 3], dtype=torch.float32, device=DEVICE, requires_grad=True)
    spine_joints = [0, 3, 6, 9, 12, 15]
    beta_weights = torch.linspace(1.0, 3.0, NUM_BETAS, device=DEVICE)

    def clamp_betas():
        with torch.no_grad():
            betas[:, :5].clamp_(-2.5, 2.5)
            betas[:, 5:].clamp_(-1.5, 1.5)

    def joint_loss(smpl_joints, target, weight, axis_w):
        diff = (smpl_joints * GLOBAL_SCALE - target) ** 2
        diff = diff * axis_w.unsqueeze(0)
        return (diff * weight.unsqueeze(1)).mean()

    # ---- stage 1: shape + global orientation ----
    print("[INFO] Stage 1: shape + global orientation...")
    opt1 = torch.optim.Adam([betas, global_orient], lr=LR)
    for it in trange(150, desc="Stage 1"):
        opt1.zero_grad()
        out = smplx_model(betas=betas, body_pose=body_pose,
                          global_orient=global_orient, transl=transl)
        smpl_joints = out.joints[0, smpl_indices, :] - out.joints[0, 0, :]
        total = sum(joint_loss(smpl_joints, t, w, a)
                    for t, w, a in zip(all_targets, all_weights, all_axis_weights))
        loss = (total / len(all_targets)
                + 5e-4 * torch.mean(beta_weights * betas[0] ** 2)
                + 1e-2 * segment_length_loss(out.joints[0]))
        loss.backward()
        opt1.step()
        clamp_betas()

    # ---- stage 2: limbs (spine locked) ----
    print("[INFO] Stage 2: limbs (spine locked)...")
    opt2 = torch.optim.Adam([body_pose, betas, global_orient], lr=LR * 0.5)
    for it in trange(250, desc="Stage 2"):
        opt2.zero_grad()
        out = smplx_model(betas=betas, body_pose=body_pose,
                          global_orient=global_orient, transl=transl)
        smpl_joints = out.joints[0, smpl_indices, :] - out.joints[0, 0, :]
        total = sum(joint_loss(smpl_joints, t, w, a)
                    for t, w, a in zip(all_targets, all_weights, all_axis_weights))
        loss = (total / len(all_targets)
                + 5e-2 * torch.sum(body_pose[0, :12] ** 2)
                + 5e-4 * torch.sum(body_pose[0, 12:] ** 2)
                + 3e-4 * torch.mean(beta_weights * betas[0] ** 2)
                + 1e-2 * segment_length_loss(out.joints[0]))
        loss.backward()
        opt2.step()
        with torch.no_grad():
            body_pose[0, :12].clamp_(-0.2, 0.2)
            body_pose[0, 12:].clamp_(-np.pi, np.pi)
            global_orient.clamp_(-np.pi, np.pi)
        clamp_betas()

    # ---- stage 3: fine-tune all ----
    print("[INFO] Stage 3: fine-tune all...")
    opt3 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=LR * 0.2)
    best_loss, best_params = float('inf'), None
    for it in trange(200, desc="Stage 3"):
        opt3.zero_grad()
        out = smplx_model(betas=betas, body_pose=body_pose,
                          global_orient=global_orient, transl=transl)
        smpl_joints = out.joints[0, smpl_indices, :] - out.joints[0, 0, :]
        total = sum(joint_loss(smpl_joints, t, w, a)
                    for t, w, a in zip(all_targets, all_weights, all_axis_weights))
        spine_coords = out.joints[0, spine_joints, :]
        spine_dirs = spine_coords[1:] - spine_coords[:-1]
        spine_dirs_n = spine_dirs / (torch.norm(spine_dirs, dim=1, keepdim=True) + 1e-8)
        loss = (total / len(all_targets)
                + 2e-2 * torch.mean((1 - torch.sum(spine_dirs_n[:-1] * spine_dirs_n[1:], dim=1)) ** 2)
                + 3e-2 * torch.sum(body_pose[0, :12] ** 2)
                + 1e-3 * torch.sum(body_pose[0, 12:] ** 2)
                + 3e-4 * torch.mean(beta_weights * betas[0] ** 2)
                + 1e-2 * segment_length_loss(out.joints[0]))
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
                'betas': betas.detach().clone(),
                'body_pose': body_pose.detach().clone(),
                'global_orient': global_orient.detach().clone(),
                'transl': transl.detach().clone(),
            }
    print(f"[INFO] Optimization complete. Best loss: {best_loss:.6f}")

    # ---- generate + export mesh into the passed-in job_dir ----
    final_out = smplx_model(
        betas=best_params['betas'], body_pose=best_params['body_pose'],
        global_orient=best_params['global_orient'], transl=best_params['transl'])
    verts = final_out.vertices[0].cpu().detach().numpy()
    faces = smplx_model.faces
    verts_centered = verts - verts.mean(axis=0)
    verts_centered = (R.from_euler('x', 180, degrees=True).as_matrix() @ verts_centered.T).T
    verts_scaled = verts_centered * GLOBAL_SCALE

    mesh_path = os.path.join(job_dir, "mesh.obj")
    trimesh.Trimesh(verts_scaled, faces).export(mesh_path)
    print(f"[INFO] Exported {mesh_path}")

    mesh_coloured_path = os.path.join(job_dir, "mesh_colored.ply")
    mesh_ply = trimesh.Trimesh(verts_scaled, faces)
    mesh_ply.visual.vertex_colors = [200, 200, 230, 255]
    mesh_ply.export(mesh_coloured_path)

    return {
        "mesh_path": mesh_path,
        "gender": detected_gender,
        "height_cm": round(target_height_m * 100, 1),
        "weight_kg": weight_kg,
        "age": age,
    }


# ========== QUEUE PLUMBING ==========
# def process_job(job_id, image_paths, gender, height_cm):
#     job_dir = os.path.join(JOBS_DIR, job_id)
#     os.makedirs(job_dir, exist_ok=True)  # already exists if server.py made it
#     return run_fitting(image_paths, gender, height_cm, job_dir)

def process_job(job_id, image_paths, gender, height_cm, weight_kg, age):
    job_dir = os.path.abspath(os.path.join(JOBS_DIR, job_id))
    os.makedirs(job_dir, exist_ok=True)
    return run_fitting(image_paths, gender, height_cm, job_dir, weight_kg, age)

def on_message(channel, method, properties, body):
    payload = json.loads(body)
    job_id = payload["job_id"]
    image_paths = payload["image_paths"]
    gender = payload.get("gender")
    height_cm = payload.get("height_cm")
    weight_kg = payload.get("weight_kg")
    age = payload.get("age")

    print(f"[ai_tailor_consumer] processing job {job_id}")
    try:
        result = process_job(job_id, image_paths, gender, height_cm, weight_kg, age)
        out_payload = {
            "job_id": job_id,
            "mesh_path": result["mesh_path"],
            "gender": result["gender"],
            "height_cm": result["height_cm"],
            "weight_kg": result["weight_kg"],
            "age": result["age"],
            "status": "mesh_ready",
        }
        channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key="mesh.ready",
            body=json.dumps(out_payload),
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
        )
        print(f"[ai_tailor_consumer] published mesh.ready for job {job_id}")
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"[ai_tailor_consumer] failed job {job_id}: {e}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type="topic", durable=True)
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message)
    print("[ai_tailor_consumer] waiting for jobs on ai_tailor_queue...")
    channel.start_consuming()


if __name__ == "__main__":
    main()
