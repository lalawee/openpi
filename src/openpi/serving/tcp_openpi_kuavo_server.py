#!/usr/bin/env python3
import dataclasses
import logging
import socket
import struct
from typing import Any, Dict, Tuple

import cv2
import msgpack
import numpy as np
import tyro

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config


# -----------------------------
# Kuavo arm+hand indices (same as config.py)
# -----------------------------
ARM_HAND_IDX: Tuple[int, ...] = tuple(list(range(0, 13)) + list(range(22, 35)))  # 26 dims


# -----------------------------
# CLI
# -----------------------------
@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 5555

    policy_config: str = "pi05_kuavo_armhand_26d_lora"
    checkpoint_dir: str = ""

    # Client is unchanged, so prompt must be injected server-side.
    default_prompt: str = "do the task"

    # Return which step from the horizon
    action_index: int = 0

    # Images are resized in transforms too, but doing it here avoids unexpected shapes.
    image_hw: Tuple[int, int] = (224, 224)


# -----------------------------
# TCP framing (4-byte len + msgpack)
# -----------------------------
def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf += chunk
    return buf


def recv_msg(conn: socket.socket) -> Dict[str, Any]:
    header = recv_exact(conn, 4)
    (size,) = struct.unpack("!I", header)
    payload = recv_exact(conn, size)
    return msgpack.unpackb(payload, raw=False)


def send_msg(conn: socket.socket, obj: Dict[str, Any]) -> None:
    payload = msgpack.packb(obj, use_bin_type=True)
    conn.sendall(struct.pack("!I", len(payload)) + payload)


# -----------------------------
# JPEG -> uint8 RGB HWC
# -----------------------------
def decode_jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


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
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


# -----------------------------
# Build obs dict matching your unchanged client schema
# Client sends: {"state": [...], "images": {"head_cam_h": jpeg, "wrist_cam_l": jpeg, "wrist_cam_r": jpeg}, "meta": {...}}
# We build what your Kuavo transforms expect: {"state": (44,), "image": {...}, "actions": (H,44), "prompt": ...}
# -----------------------------
def build_openpi_inputs(req: Dict[str, Any], *, action_horizon: int, default_prompt: str, image_hw: Tuple[int, int]) -> Dict[str, Any]:
    state = req.get("state", None)
    images = req.get("images", None)
    if state is None or images is None:
        raise KeyError("Request must include 'state' and 'images' (client unchanged expectation).")

    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_np.shape[0] != 44:
        raise ValueError(f"Expected state dim 44 (client), got {state_np.shape[0]}")

    # Map client camera names -> Kuavo config image keys
    # (these keys match IMAGE_KEYS in your config.py)
    cam_map = {
        "head_cam_h": "base_0_rgb",
        "wrist_cam_l": "left_wrist_0_rgb",
        "wrist_cam_r": "right_wrist_0_rgb",
    }

    image_dict: Dict[str, np.ndarray] = {}
    for client_cam, dest_key in cam_map.items():
        blob = images.get(client_cam, None)
        if blob is None:
            # If a camera is missing, provide black image; your AddImageMask will mark it False anyway.
            image_dict[dest_key] = np.zeros((image_hw[0], image_hw[1], 3), dtype=np.uint8)
            continue
        rgb = decode_jpeg_to_rgb(blob)
        rgb = resize_with_pad_rgb(rgb, image_hw)
        image_dict[dest_key] = rgb

    # IMPORTANT: your SliceStateAndActions transform expects 'actions' key too.
    # Provide dummy actions (zeros) so transforms don't crash during inference.
    dummy_actions = np.zeros((action_horizon, 44), dtype=np.float32)

    return {
        "state": state_np,
        "actions": dummy_actions,
        "image": image_dict,
        "prompt": default_prompt,  # client unchanged; we inject
    }


# -----------------------------
# Convert policy output 32D -> client 44D
# -----------------------------
def actions32_to_actions44(actions32: np.ndarray) -> np.ndarray:
    # actions32: (H, 32)
    if actions32.ndim != 2 or actions32.shape[1] < 26:
        raise ValueError(f"Expected actions shape (H,>=26), got {actions32.shape}")
    armhand26 = actions32[:, :26]
    out44 = np.zeros((actions32.shape[0], 44), dtype=np.float32)
    out44[:, ARM_HAND_IDX] = armhand26
    return out44


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    if not args.checkpoint_dir:
        raise ValueError("--checkpoint_dir is required")

    train_cfg = _config.get_config(args.policy_config)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,  # also handled here, but we inject explicitly too
    )

    # We rely on the config model horizon for dummy actions shape.
    action_horizon = int(train_cfg.model.action_horizon)

    logging.info("Loaded policy config=%s checkpoint=%s", args.policy_config, args.checkpoint_dir)
    logging.info("Model action_horizon=%d action_dim=%d", action_horizon, int(train_cfg.model.action_dim))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    logging.info("Listening on %s:%d", args.host, args.port)

    while True:
        conn, addr = s.accept()
        logging.info("Client connected: %s", addr)
        try:
            while True:
                req = recv_msg(conn)

                obs = build_openpi_inputs(
                    req,
                    action_horizon=action_horizon,
                    default_prompt=args.default_prompt,
                    image_hw=args.image_hw,
                )

                out = policy.infer(obs)
                actions32 = np.asarray(out["actions"], dtype=np.float32)  # (H,32)
                actions44 = actions32_to_actions44(actions32)             # (H,44)

                idx = args.action_index
                if idx < 0 or idx >= actions44.shape[0]:
                    raise IndexError(f"action_index={idx} out of range (horizon={actions44.shape[0]})")

                # Match replay_dataset_tcp expectation: resp["action"] exists.
                resp = {
                    "action": actions44[idx].tolist(),
                    "mode": "absolute",
                    "horizon": int(actions44.shape[0]),
                    "action_dim": int(actions44.shape[1]),
                }
                send_msg(conn, resp)

        except Exception as e:
            logging.exception("Client loop ended: %s", e)
        finally:
            conn.close()
            logging.info("Client disconnected: %s", addr)


if __name__ == "__main__":
    main(tyro.cli(Args))