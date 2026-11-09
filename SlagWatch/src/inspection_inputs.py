"""Deterministic synthetic inspection conditions at the original image resolution."""
import hashlib

import numpy as np

BINARY_CONDITIONS = (("clean",0),("noise_weak",101),("noise_weak",102),
                     ("noise_strong",101),("noise_strong",102),("dark",0),("bright",0))
NOISE = {"noise_weak":0.02,"noise_strong":0.08}
GAIN = {"dark":0.8,"bright":1.2}


def change_image(raw, sample_id, condition, seed):
    if raw.ndim != 3 or raw.shape[-1] != 3 or raw.dtype != np.float32:
        raise ValueError("Expected native RGB float32 image")
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError("Native image must be finite in [0,1]")
    if condition == "clean":
        return raw.copy()
    if condition in GAIN:
        return np.clip(raw * np.float32(GAIN[condition]), 0, 1)
    if condition not in NOISE:
        raise ValueError(f"Unknown binary inspection condition: {condition}")
    token = f"inspection-rgb-v1|{sample_id}|{condition}|{int(seed)}".encode()
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8],"big")
    offset = np.random.default_rng(derived).normal(0, NOISE[condition], raw.shape).astype(np.float32)
    return np.clip(raw + offset, 0, 1)
