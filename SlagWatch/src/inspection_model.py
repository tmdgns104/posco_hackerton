"""Registered KSDD2 models: one whole RGB image in, one binary decision out.

Masks, names, source IDs and labels are intentionally absent from the inference API.
Heads use defect=1; a larger decision score means stronger evidence for a defect.
"""
from functools import lru_cache
from pathlib import Path
from threading import local
import hashlib
import json

import cv2
import joblib
import numpy as np
import torch
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/inspection_binary"
WEIGHTS = ROOT / "experiments/surface_v1/models/resnet18-f37072fd.pth"
ENCODER_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PREP = "ksdd2-whole-rgb-fit192x512-reflect-meanmax-stage3_4-v1"
BATCH = 8
H, W = 512, 192
MEAN = torch.tensor([.485, .456, .406], dtype=torch.float32)[None, :, None, None]
STD = torch.tensor([.229, .224, .225], dtype=torch.float32)[None, :, None, None]
_PER_THREAD = local()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _encoder():
    if sha(WEIGHTS) != ENCODER_SHA:
        raise ValueError("Frozen ImageNet ResNet18 file differs from registered weights")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    net = resnet18(weights=None)
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return net.eval().requires_grad_(False).to(memory_format=torch.channels_last)


def prepare(raw):
    if (not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.shape[2] != 3
            or raw.dtype != np.float32 or min(raw.shape[:2]) < 1
            or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1):
        raise ValueError("Expected finite RGB HWC float32 pixels in [0,1]")
    # Preserve orientation for the native portrait data; landscape uploads are
    # rotated clockwise for a full, aspect-preserving view on this model's canvas.
    img = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE) if raw.shape[1] > raw.shape[0] else raw
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    rh, rw = max(1, round(h * s)), max(1, round(w * s))
    rs = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    top, left = (H-rh)//2, (W-rw)//2
    return cv2.copyMakeBorder(rs, top, H-rh-top, left, W-rw-left, cv2.BORDER_REFLECT_101)


@torch.inference_mode()
def extract(raws):
    if not len(raws):
        return np.empty((0, 1536), dtype=np.float64)
    net = _encoder()
    parts = []
    for i in range(0, len(raws), BATCH):
        ps = [prepare(x) for x in raws[i:i+BATCH]]
        # Fixed tensor shape, including a 1-image case replay, avoids different
        # convolution kernels just because a sample was run alone in the UI.
        t = np.stack(ps + [ps[-1]] * (BATCH-len(ps)))
        rgb = torch.from_numpy(t).permute(0, 3, 1, 2)
        a = ((rgb - MEAN)/STD).contiguous(memory_format=torch.channels_last)
        x = net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(a))))))
        l3 = net.layer3(x)
        l4 = net.layer4(l3)
        # Whole map max lets a small localized response survive spatial pooling.
        parts.append(torch.cat([l3.mean((2,3)), l3.amax((2,3)),
                                l4.mean((2,3)), l4.amax((2,3))], dim=1)[:len(ps)].numpy())
    return np.concatenate(parts).astype(np.float64)


def _shared_features(raws):
    # Only share the most recent exact arrays within one thread. No stored
    # prediction, manifest, example or label is consulted, even on a hit.
    key = (ENCODER_SHA, PREP, tuple((id(a), a.shape, a.dtype.str,
           hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).digest()) for a in raws))
    last = getattr(_PER_THREAD, "last", None)
    if last is not None and last[0] == key:
        return last[1]
    features = extract(raws)
    _PER_THREAD.last = (key, features)
    return features


def decision(head, features):
    # Versioned score precision is applied identically for fitting the cutoff,
    # initial prediction and replay. Labels are not based on score argmax.
    return np.asarray(head.decision_function(features)).round(6)


class BinaryAdapter:
    def __init__(self, model_id, head, identity):
        self.model_id, self.head, self.identity = model_id, head, identity

    def predict(self, raws):
        if not len(raws):
            return []
        values = decision(self.head, _shared_features(raws))
        return [dict(predicted_label="defect" if z >= self.identity["threshold"] else "normal",
                     scores={"normal": float(-z), "defect": float(z)},
                     score_type="decision_function", status="ok") for z in values]


def load_binary(model_id):
    if model_id not in {"K0", "K1"}:
        raise ValueError("Unknown registered KSDD2 model")
    selection = json.loads((RUN/"selection.json").read_text(encoding="utf-8"))
    rel = f"models/{model_id}.joblib"
    path = RUN/rel
    digest = sha(path)
    if (digest != selection["frozen_files"][rel]
            or sha(RUN/"protocol.json") != selection["frozen_files"]["protocol.json"]
            or sha(Path(__file__)) != selection["model_source_sha256"]
            or sha(WEIGHTS) != ENCODER_SHA):
        raise ValueError("Registered head, code, protocol or encoder checksum differs")
    # Only allow the two trusted, local, checksum-matched heads. No uploaded pickle.
    b = joblib.load(path)
    if (b["model_id"] != model_id or b["preprocessing"] != PREP or b["encoder_sha256"] != ENCODER_SHA
            or not np.array_equal(b["head"].classes_, [0, 1]) or not np.isfinite(b["threshold"])):
        raise ValueError("Binary model registry/label mapping mismatch")
    identity = dict(model_id=model_id, artifact_sha256=digest, head_sha256=digest,
                    preprocessing=PREP, encoder="ImageNet1K V1 / frozen ResNet18",
                    encoder_sha256=ENCODER_SHA, threshold=float(b["threshold"]),
                    class_mapping={"0":"normal","1":"defect"}, protocol_sha256=sha(RUN/"protocol.json"),
                    score_rule="round(decision_function,6) >= threshold for defect",
                    model_source_sha256=selection["model_source_sha256"])
    return BinaryAdapter(model_id, b["head"], identity)
