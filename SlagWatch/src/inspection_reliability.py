"""Analysis of verified paired binary inspection records without running models."""
from __future__ import annotations

import numpy as np
import pandas as pd


TRANSITIONS = (
    "new_fn", "fixed_fn", "persistent_fn", "stable_tp",
    "new_fp", "fixed_fp", "persistent_fp", "stable_tn", "failed",
)


def classify_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "condition", "seed", "true_label", "before_pred", "after_pred",
                "before_status", "after_status"}
    if not required.issubset(pairs.columns) or pairs.empty:
        raise ValueError("Nonempty paired binary predictions with sample, condition and status are required.")
    if pairs[list(required)].isna().any().any():
        raise ValueError("Paired prediction fields cannot be missing.")
    if pairs.duplicated(["sample_id", "condition", "seed"]).any():
        raise ValueError("Duplicate original, condition and seed.")
    if (pairs.sample_id.astype(str).str.strip().eq("").any()
            or not pairs.true_label.isin(["normal", "defect"]).all()
            or pairs.groupby("sample_id").true_label.nunique().gt(1).any()):
        raise ValueError("Invalid original ID or inconsistent binary ground truth.")
    out = pairs.copy()
    complete = out.before_status.eq("ok") & out.after_status.eq("ok")
    if (not out.loc[complete, "before_pred"].isin(["normal", "defect"]).all()
            or not out.loc[complete, "after_pred"].isin(["normal", "defect"]).all()):
        raise ValueError("Completed predictions must have binary labels.")
    defect = out.true_label.eq("defect")
    old_positive = out.before_pred.eq("defect")
    new_positive = out.after_pred.eq("defect")
    conditions = [
        complete & defect & old_positive & ~new_positive,
        complete & defect & ~old_positive & new_positive,
        complete & defect & ~old_positive & ~new_positive,
        complete & defect & old_positive & new_positive,
        complete & ~defect & ~old_positive & new_positive,
        complete & ~defect & old_positive & ~new_positive,
        complete & ~defect & old_positive & new_positive,
        complete & ~defect & ~old_positive & ~new_positive,
    ]
    out["transition"] = np.select(conditions, TRANSITIONS[:-1], default="failed")
    return out


def transition_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    data = classify_pairs(pairs)
    rows = []
    groups = [("ALL", -1, data)]
    groups += [(condition, seed, group) for (condition, seed), group
               in data.groupby(["condition", "seed"], sort=False, dropna=False)]
    for condition, seed, group in groups:
        counts = group.transition.value_counts()
        rows.append({"condition": condition, "seed": seed,
                     "originals": group.sample_id.nunique(), "decisions": len(group),
                     "defect_originals": group.loc[group.true_label.eq("defect"), "sample_id"].nunique(),
                     "normal_originals": group.loc[group.true_label.eq("normal"), "sample_id"].nunique(),
                     **{name: int(counts.get(name, 0)) for name in TRANSITIONS}})
    return pd.DataFrame(rows)


def defect_size_summary(pairs: pd.DataFrame, index: pd.DataFrame, *, small_below: float,
                        medium_below: float) -> pd.DataFrame:
    """Group by annotated defect-pixel fraction, not inferred defect shape or severity."""
    if not 0 < small_below < medium_below <= 1:
        raise ValueError("Defect area boundaries must satisfy 0 < small < medium <= 1.")
    required = {"sample_id", "defect_pixels", "width", "height"}
    if not required.issubset(index.columns) or index.sample_id.duplicated().any():
        raise ValueError("A unique image index with annotated dimensions and defect pixels is required.")
    defects = classify_pairs(pairs)
    defects = defects[defects.true_label.eq("defect")]
    defects = defects.merge(index[list(required)], on="sample_id", how="left", validate="many_to_one")
    for column in ("defect_pixels", "width", "height"):
        defects[column] = pd.to_numeric(defects[column], errors="coerce")
    pixels, width, height = (defects[c] for c in ("defect_pixels", "width", "height"))
    if (defects[list(required)].isna().any().any()
            or not np.isfinite(defects[["defect_pixels", "width", "height"]].to_numpy()).all()
            or (width <= 0).any() or (height <= 0).any()
            or (pixels <= 0).any() or (pixels > width * height).any()):
        raise ValueError("Missing or invalid defect annotations or image dimensions.")
    fraction = pixels / (width * height)
    defects["size"] = np.select([fraction < small_below, fraction < medium_below],
                                ["small", "medium"], default="large")
    rows = []
    for size in ("small", "medium", "large"):
        group = defects[defects["size"].eq(size)]
        counts = group.transition.value_counts()
        rows.append({"size": size, "originals": group.sample_id.nunique(), "decisions": len(group),
                     **{name: int(counts.get(name, 0)) for name in
                        ("new_fn", "fixed_fn", "persistent_fn", "stable_tp", "failed")}})
    return pd.DataFrame(rows)


def group_bootstrap(pairs: pd.DataFrame, *, repetitions: int = 2000, seed: int = 42,
                    weighting: str = "decisions") -> pd.DataFrame:
    """Paired percentile intervals, resampling original IDs with their repeated decisions."""
    data = classify_pairs(pairs)
    if not isinstance(repetitions, int) or not 1 <= repetitions <= 10000:
        raise ValueError("Bootstrap repetitions must be between 1 and 10000.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer.")
    if weighting not in ("decisions", "originals"):
        raise ValueError("Unknown weighting: use decisions or originals.")
    data = data.assign(
        defect=data.true_label.eq("defect").astype(int),
        normal=data.true_label.eq("normal").astype(int),
        before_fn=(data.true_label.eq("defect") & data.before_pred.eq("normal")).astype(int),
        after_fn=(data.true_label.eq("defect") & data.after_pred.eq("normal")).astype(int),
        before_fp=(data.true_label.eq("normal") & data.before_pred.eq("defect")).astype(int),
        after_fp=(data.true_label.eq("normal") & data.after_pred.eq("defect")).astype(int),
        before_failed=data.before_status.ne("ok").astype(int),
        after_failed=data.after_status.ne("ok").astype(int),
    )
    grouped = data.groupby("sample_id", sort=True)
    columns = ["defect", "normal", "before_fn", "after_fn", "before_fp", "after_fp"]
    values = grouped[columns].sum().astype(float)
    truth = grouped.true_label.first()
    if weighting == "originals":
        values = values.div(grouped.size(), axis=0)
    values = values.to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=repetitions)
    draws = weights @ values
    point = values.sum(axis=0)
    failed = {
        (side, label): bool(data.loc[data.true_label.eq(label), side + "_status"].ne("ok").any())
        for side in ("before", "after") for label in ("defect", "normal")
    }

    def rate(numerator, denominator, unavailable):
        if unavailable:
            return None, np.full(repetitions, np.nan)
        p = point[numerator] / point[denominator] if point[denominator] else None
        out = np.full(repetitions, np.nan)
        np.divide(draws[:, numerator], draws[:, denominator], out=out, where=draws[:, denominator] != 0)
        return p, out

    results = []
    for metric, truth_label, denominator, numerator in (
        ("recall", "defect", 0, (2, 3)),
        ("fn_rate", "defect", 0, (2, 3)),
        ("fp_rate", "normal", 1, (4, 5)),
    ):
        computed = []
        for side, index in zip(("before", "after"), numerator, strict=True):
            estimate, distribution = rate(index, denominator, failed[side, truth_label])
            if metric == "recall":
                estimate = None if estimate is None else 1 - estimate
                distribution = 1 - distribution
            computed.append((estimate, distribution))
        a, b = computed
        delta = (None if a[0] is None or b[0] is None else b[0] - a[0], b[1] - a[1])
        for model, (estimate, distribution) in zip(("before", "after", "difference"), (a, b, delta), strict=True):
            valid = distribution[np.isfinite(distribution)]
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
            results.append({"metric": metric, "model": model, "estimate": estimate, "lower": low,
                            "upper": high, "valid_repetitions": len(valid), "repetitions": repetitions,
                            "originals": n, "class_originals": int(truth.eq(truth_label).sum()),
                            "decisions": len(data), "seed": seed, "weighting": weighting})
    return pd.DataFrame(results)
