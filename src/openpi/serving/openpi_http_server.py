#!/usr/bin/env python3
"""
Flask HTTP wrapper for OpenPI Kuavo policy that matches your ROS-side GR00T-style request pattern.

POST /predict (multipart/form-data)
Files:
  ego_view: JPEG
  left_wrist_view: JPEG
  right_wrist_view: JPEG   (optional if you want)
Fields:
  task_description: string
  left_arm:  "a0,a1,...,a6"    (7)
  right_arm: "b0,b1,...,b6"    (7)
  left_hand: "h0,...,h5"       (6)
  right_hand:"k0,...,k5"       (6)

Response (GR00T-like):
{
  "success": true,
  "action.left_arm":  [[...7...], xT],
  "action.right_arm": [[...7...], xT],
  "action.left_hand": [[...6...], xT],
  "action.right_hand":[[...6...], xT],
  "meta": {...}
}
"""

import argparse
from typing import Dict, Any, Tuple, Optional

import numpy as np
import cv2
from flask import Flask, request, jsonify

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config


# ---- Kuavo joint layout (same as your other servers) ----
ARM_HAND_IDX: Tuple[int, ...] = tuple(list(range(0, 13)) + list(range(22, 35)))  # 26 dims


def parse_csv_floats(s: str, n: int, name: str) -> np.ndarray:
    parts = [p.strip() for p in (s or "").split(",") if p.strip() != ""]
    if len(parts) != n:
        raise ValueError(f"{name} must have {n} values, got {len(parts)}")
    return np.asarray([float(x) for x in parts], dtype=np.float32)


def build_state44(left_arm7: np.ndarray, left_hand6: np.ndarray,
                  right_arm7: np.ndarray, right_hand6: np.ndarray) -> np.ndarray:
    state = np.zeros((44,), dtype=np.float32)
    state[0:7] = left_arm7
    state[7:13] = left_hand6
    state[22:29] = right_arm7
    state[29:35] = right_hand6
    return state


def split_action44(action44: np.ndarray):
    left_arm = action44[0:7]
    left_hand = action44[7:13]
    right_arm = action44[22:29]
    right_hand = action44[29:35]
    return left_arm, right_arm, left_hand, right_hand


def repeat_traj(vec: np.ndarray, T: int) -> np.ndarray:
    return np.tile(vec.reshape(1, -1), (T, 1))


def decode_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_with_pad_rgb(img_rgb: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    h, w = img_rgb.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    top = (out_h - new_h) // 2
    left = (out_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas


def actions32_to_actions44(actions32: np.ndarray) -> np.ndarray:
    """
    actions32: (H, 32) from policy. We take first 26 dims as arm+hand,
    then scatter into 44D (others zero).
    """
    if actions32.ndim != 2 or actions32.shape[1] < 26:
        raise ValueError(f"Unexpected actions shape: {actions32.shape}")
    armhand26 = actions32[:, :26]
    out44 = np.zeros((actions32.shape[0], 44), dtype=np.float32)
    out44[:, ARM_HAND_IDX] = armhand26
    return out44


def create_app(
    *,
    policy_config_name: str,
    checkpoint_dir: str,
    default_prompt: Optional[str],
    required_http_camera_keys: Tuple[str, ...],
    model_image_size: Tuple[int, int],
    return_horizon_steps: int,
    action_index: int,
) -> Flask:
    train_cfg = _config.get_config(policy_config_name)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        checkpoint_dir,
        default_prompt=default_prompt,
    )

    # Use config horizon for dummy-actions and sanity
    action_horizon = int(train_cfg.model.action_horizon)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "ok": True,
            "policy_config": policy_config_name,
            "checkpoint_dir": checkpoint_dir,
            "action_horizon": action_horizon,
            "action_dim": int(train_cfg.model.action_dim),
            "required_http_camera_keys": list(required_http_camera_keys),
            "model_image_size": list(model_image_size),
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            # 1) validate cams
            missing = [k for k in required_http_camera_keys if k not in request.files]
            if missing:
                return jsonify({
                    "success": False,
                    "error": "missing_camera",
                    "missing": missing,
                    "required": list(required_http_camera_keys),
                }), 400

            # 2) parse joints (same as diffusion wrapper)
            left_arm7 = parse_csv_floats(request.form.get("left_arm", ""), 7, "left_arm")
            right_arm7 = parse_csv_floats(request.form.get("right_arm", ""), 7, "right_arm")
            left_hand6 = parse_csv_floats(request.form.get("left_hand", ""), 6, "left_hand")
            right_hand6 = parse_csv_floats(request.form.get("right_hand", ""), 6, "right_hand")
            state44 = build_state44(left_arm7, left_hand6, right_arm7, right_hand6)

            # 3) build OpenPI inputs in YOUR Kuavo format from config.py
            # Kuavo expects: {"state": (44,), "actions": (H,44), "image": {...}, "prompt": str}
            images: Dict[str, np.ndarray] = {}

            # Map HTTP camera field names -> Kuavo IMAGE_KEYS
            cam_map = {
                "ego_view": "base_0_rgb",
                "left_wrist_view": "left_wrist_0_rgb",
                "right_wrist_view": "right_wrist_0_rgb",
            }

            for http_key, dest_key in cam_map.items():
                if http_key not in request.files:
                    # if optional camera missing, black image (mask transform can handle)
                    images[dest_key] = np.zeros((model_image_size[0], model_image_size[1], 3), dtype=np.uint8)
                    continue
                img_bytes = request.files[http_key].read()
                rgb = decode_jpeg_bytes(img_bytes)
                rgb = resize_with_pad_rgb(rgb, model_image_size)
                images[dest_key] = rgb

            prompt = request.form.get("task_description", None) or default_prompt or ""
            if not prompt.strip():
                return jsonify({"success": False, "error": "bad_request", "message": "task_description empty and no default_prompt"}), 400

            dummy_actions44 = np.zeros((action_horizon, 44), dtype=np.float32)

            openpi_in = {
                "state": state44,
                "actions": dummy_actions44,  # needed by your SliceStateAndActions transform
                "image": images,
                "prompt": prompt,
            }

            # 4) infer
            out = policy.infer(openpi_in)
            actions32 = np.asarray(out["actions"], dtype=np.float32)   # (H,32)
            actions44 = actions32_to_actions44(actions32)              # (H,44)

            idx = action_index
            if idx < 0 or idx >= actions44.shape[0]:
                raise IndexError(f"action_index={idx} out of range for horizon={actions44.shape[0]}")

            # 5) split into left/right arm/hand + return trajectories (GR00T style)
            action44 = actions44[idx]
            left_arm, right_arm, left_hand, right_hand = split_action44(action44)

            T = int(return_horizon_steps)
            resp = {
                "success": True,
                "action.left_arm": repeat_traj(left_arm, T).tolist(),
                "action.right_arm": repeat_traj(right_arm, T).tolist(),
                "action.left_hand": repeat_traj(left_hand, T).tolist(),
                "action.right_hand": repeat_traj(right_hand, T).tolist(),
                "meta": {
                    "task_description": prompt,
                    "used_cameras": list(required_http_camera_keys),
                    "policy_config": policy_config_name,
                    "checkpoint_dir": checkpoint_dir,
                    "openpi_action_horizon": int(actions32.shape[0]),
                    "openpi_action_dim": int(actions32.shape[1]),
                    "returned_step": idx,
                },
            }
            return jsonify(resp)

        except ValueError as e:
            return jsonify({"success": False, "error": "bad_request", "message": str(e)}), 400
        except Exception as e:
            return jsonify({"success": False, "error": "server_error", "message": str(e)}), 500

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6633)

    ap.add_argument("--policy.config", dest="policy_config", required=True)
    ap.add_argument("--policy.dir", dest="checkpoint_dir", required=True)
    ap.add_argument("--default_prompt", default=None)

    ap.add_argument("--model_image_h", type=int, default=224)
    ap.add_argument("--model_image_w", type=int, default=224)

    ap.add_argument("--return_horizon_steps", type=int, default=16)
    ap.add_argument("--action_index", type=int, default=0)

    # Cameras required by ROS controller
    ap.add_argument("--required_cams", nargs="+", default=["ego_view", "left_wrist_view", "right_wrist_view"])

    args = ap.parse_args()

    app = create_app(
        policy_config_name=args.policy_config,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        required_http_camera_keys=tuple(args.required_cams),
        model_image_size=(args.model_image_h, args.model_image_w),
        return_horizon_steps=args.return_horizon_steps,
        action_index=args.action_index,
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()