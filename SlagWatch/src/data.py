"""SSFD v3 recordings, labels and 0.5-second windows."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

HZ = 6400
SIZE = HZ // 2
EXPECTED = {f"{phase}-{n}.csv" for phase in "EBS" for n in range(1, 17)}
FILENAME = re.compile(r"([EBS])-(1[0-6]|[1-9])\.csv")


def recording_paths(folder):
    paths = {}
    for path in sorted(Path(folder).rglob("*.csv")):
        if not FILENAME.fullmatch(path.name):
            continue
        if path.name in paths:
            raise ValueError(f"중복 파일: {path.name}")
        paths[path.name] = path
    if missing := EXPECTED - paths.keys():
        raise ValueError("누락 파일: " + ", ".join(sorted(missing)))
    return paths


def read_recording(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["time", "x", "y", "z"] or len(table) != SIZE * 10:
        raise ValueError(f"{path.name}: 32,000행과 time,x,y,z 열이 필요합니다.")
    values = np.ascontiguousarray(table[["x", "y", "z"]], dtype=float)
    if table.time.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{path.name}: 원본에 결측값 또는 비유한 값이 있습니다.")
    return values.reshape(10, SIZE, 3)


def load_dataset(folder):
    folder = Path(folder)
    source = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    hashes = {Path(row["path"]).name: row["sha256"] for row in source["files"]}
    metadata, signals, manifest = [], [], []
    for name, path in recording_paths(folder).items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != hashes[name]:
            raise ValueError(f"{name}: SSFD 원본 SHA256 불일치")
        phase, number = FILENAME.fullmatch(name).groups()
        meta = dict(file_id=name, phase=phase, group_id=int(number), y_true=int(phase == "S"))
        windows = read_recording(path)
        manifest.append(dict(**meta, rows=SIZE * 10, sha256=digest))
        for i, window in enumerate(windows):
            metadata.append(dict(**meta, window_idx=i, start_s=i/2, end_s=(i+1)/2))
            signals.append(window)
    return pd.DataFrame(metadata), signals, pd.DataFrame(manifest)
