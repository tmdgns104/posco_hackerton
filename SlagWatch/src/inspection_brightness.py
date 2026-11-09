"""Original image pixel brightness, not measured physical illumination."""
import numpy as np
import pandas as pd

from .inspection_reliability import TRANSITIONS, classify_pairs


def original_brightness(project, split, pairs):
    data = classify_pairs(pairs)
    if not data["split"].eq(split).all():
        raise ValueError("Pairs do not belong to the selected split.")
    rows = project.rows(split).set_index("sample_id", drop=False, verify_integrity=True)
    ids = list(dict.fromkeys(data.sample_id.astype(str)))
    if set(ids) - set(rows.index.astype(str)):
        raise ValueError("Run contains originals missing from the registered project.")
    measurements = []
    for sample_id in ids:
        row = rows.loc[sample_id]
        matched = data[data.sample_id.astype(str).eq(sample_id)]
        if (not matched.file_sha256.eq(row.file_sha256).all()
                or not matched.true_label.eq(row.true_label).all()):
            raise ValueError(f"Original lineage differs from the selected run: {sample_id}")
        raw = project.load(row)
        if (not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.shape[2] != 3
                or raw.dtype != np.float32 or not np.isfinite(raw).all()
                or raw.min() < 0 or raw.max() > 1):
            raise ValueError(f"Invalid original RGB pixels: {sample_id}")
        value = float(np.mean(raw, axis=(0, 1), dtype=np.float64) @ [0.2126, 0.7152, 0.0722])
        measurements.append({"sample_id": sample_id, "brightness": value})
    return pd.DataFrame(measurements)


def brightness_summary(pairs, measurements, *, low_below, high_from):
    if not 0 < low_below < high_from < 1:
        raise ValueError("Brightness boundaries must satisfy 0 < low < high < 1.")
    if (measurements.sample_id.duplicated().any()
            or not np.isfinite(measurements.brightness.to_numpy(dtype=float)).all()
            or not measurements.brightness.between(0, 1).all()):
        raise ValueError("Original brightness measurements are invalid.")
    data = classify_pairs(pairs).merge(measurements, on="sample_id", how="left", validate="many_to_one")
    if data.brightness.isna().any():
        raise ValueError("Missing brightness measurements for selected pairs.")
    data["brightness_group"] = np.select(
        [data.brightness < low_below, data.brightness < high_from], ["low", "middle"], default="high")
    result = []
    for name in ("low", "middle", "high"):
        group = data[data.brightness_group.eq(name)]
        counts = group.transition.value_counts()
        result.append({"brightness_group": name, "originals": group.sample_id.nunique(), "decisions": len(group),
                       "defect_originals": group.loc[group.true_label.eq("defect"), "sample_id"].nunique(),
                       "normal_originals": group.loc[group.true_label.eq("normal"), "sample_id"].nunique(),
                       **{t: int(counts.get(t, 0)) for t in TRANSITIONS}})
    return pd.DataFrame(result)
