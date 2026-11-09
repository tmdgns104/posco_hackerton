"""Paired percentile bootstrap of saved tabular predictions by sample ID."""
import numpy as np
import pandas as pd

from . import p1_tabular as tabular


def paired_intervals(pairs, *, positive=None, repetitions=2000, seed=42):
    required = {"sample_id", "condition", "true_label", "predicted_class_before",
                "predicted_class_after", "status_before", "status_after"}
    if not required.issubset(pairs.columns) or pairs.empty:
        raise ValueError("비어 있지 않은 짝 비교 결과가 필요합니다.")
    if not isinstance(repetitions, int) or not 1 <= repetitions <= 10000:
        raise ValueError("반복 횟수는 1~10,000이어야 합니다.")
    if not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
        raise ValueError("seed가 유효하지 않습니다.")
    if pairs[list(required)].isna().any().any() or pairs.duplicated(["sample_id", "condition"]).any():
        raise ValueError("누락되거나 중복된 표형 판정이 있습니다.")
    if pairs.groupby("sample_id").true_label.nunique().gt(1).any():
        raise ValueError("같은 sample_id의 정답이 일치하지 않습니다.")
    if pairs.status_before.ne("ok").any() or pairs.status_after.ne("ok").any():
        raise ValueError("미완료 추론이 있어 신뢰구간을 계산할 수 없습니다.")
    classes = set(pairs.true_label)
    if positive is not None and (not isinstance(positive, str) or not positive or len(classes | {positive}) > 2):
        raise ValueError("이진 FN/FP 신뢰구간에는 유효한 양성 클래스가 필요합니다.")
    rng = np.random.default_rng(seed)
    output = []
    groups = [("ALL", pairs), *list(pairs.groupby("condition", sort=False))]
    for condition, data in groups:
        data = data.copy()
        truth = data.true_label
        data["n"] = 1
        data["before_correct"] = data.predicted_class_before.eq(truth).astype(int)
        data["after_correct"] = data.predicted_class_after.eq(truth).astype(int)
        if positive is not None:
            is_positive = truth.eq(positive)
            data["n_positive"] = is_positive.astype(int)
            data["n_negative"] = (~is_positive).astype(int)
            for side in ("before", "after"):
                pred_pos = data[f"predicted_class_{side}"].eq(positive)
                data[f"{side}_fn"] = (is_positive & ~pred_pos).astype(int)
                data[f"{side}_fp"] = (~is_positive & pred_pos).astype(int)
        columns = ["n", "before_correct", "after_correct"]
        metrics = [("accuracy", "n", "before_correct", "after_correct")]
        if positive is not None:
            columns += ["n_positive", "n_negative", "before_fn", "after_fn", "before_fp", "after_fp"]
            metrics += [("fn_rate", "n_positive", "before_fn", "after_fn"),
                        ("fp_rate", "n_negative", "before_fp", "after_fp")]
        values = data.groupby("sample_id", sort=True)[columns].sum().to_numpy(dtype=float)
        n_groups = len(values)
        point = values.sum(axis=0)
        draws = np.empty((repetitions, len(columns)))
        for i in range(repetitions):
            indices = rng.integers(0, n_groups, n_groups)
            draws[i] = values[indices].sum(axis=0)
        for metric, denominator, before, after in metrics:
            d, b, a = (columns.index(x) for x in (denominator, before, after))
            for side, numerator in (("before", b), ("after", a), ("difference", None)):
                point_num = point[a] - point[b] if numerator is None else point[numerator]
                sample_num = draws[:, a] - draws[:, b] if numerator is None else draws[:, numerator]
                estimates = np.divide(sample_num, draws[:, d], out=np.full(repetitions, np.nan),
                                      where=draws[:, d] != 0)
                valid = estimates[np.isfinite(estimates)]
                output.append({"condition": condition, "metric": metric, "model": side,
                               "original_groups": n_groups, "decisions": int(point[0]),
                               "denominator": int(point[d]),
                               "estimate": float(point_num / point[d]) if point[d] else None,
                               "lower_95": float(np.percentile(valid, 2.5)) if len(valid) else None,
                               "upper_95": float(np.percentile(valid, 97.5)) if len(valid) else None,
                               "valid_resamples": len(valid), "repetitions": repetitions, "seed": seed})
    return pd.DataFrame(output)


def analyze_run(root, run_id, *, repetitions=2000, seed=42):
    report = tabular.load_run(root, run_id)
    pairs = pd.read_csv(root / "runs" / run_id / "pairs.csv",
                        dtype={"sample_id": str, "true_label": str, "predicted_class_before": str,
                               "predicted_class_after": str}, keep_default_na=False)
    return paired_intervals(pairs, positive=report["config"]["positive_class"],
                            repetitions=repetitions, seed=seed)
