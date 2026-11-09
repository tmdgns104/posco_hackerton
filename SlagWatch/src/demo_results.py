"""Read-only presentation of the frozen G1 comparisons."""
import numpy as np
import pandas as pd

from .surface_study import assess


def counts(run, split):
    summary = run.table(split, "summary")
    changes = run.table(split, "changes").set_index(["sigma", "seed"], verify_integrity=True)
    conditions = set(map(tuple, run.protocol["conditions"][split]))
    before, after = [
        summary[summary.model_id == name].set_index(["sigma", "seed"], verify_integrity=True)
        for name in ("V0", "G1")
    ]
    if any(set(frame.index) != conditions for frame in (before, after, changes)):
        raise ValueError("저장된 평가 조건이 프로토콜과 다릅니다.")
    n = run.man["n_originals"][split]
    if not all(frame.n_originals.eq(n).all() for frame in (before, after, changes)):
        raise ValueError("조건별 원본 수가 다릅니다.")
    rows = changes[["fixed", "new_error", "both_wrong", "both_correct"]].copy()
    for name, frame in (("V0", before), ("G1", after)):
        rows[name + "_correct"] = frame.correct
        rows[name + "_wrong"] = frame.wrong
        rows[name + "_f1"] = frame.macro_f1
    if not ((rows.fixed + rows.both_wrong == rows.V0_wrong)
            & (rows.new_error + rows.both_wrong == rows.G1_wrong)
            & (rows.fixed + rows.new_error + rows.both_wrong + rows.both_correct == n)
            & (rows.V0_wrong + rows.V0_correct == n) & (rows.G1_wrong + rows.G1_correct == n)).all():
        raise ValueError("같은 조건의 오류 변화와 전체 분류 결과가 일치하지 않습니다.")
    return rows.sort_index().reset_index()


def aggregate(rows):
    """Give each condition equal weight for F1; sum matched decisions for counts."""
    totals = rows[["V0_correct","G1_correct","V0_wrong","G1_wrong","fixed","new_error","both_wrong"]].sum().to_dict()
    totals.update(rows[["V0_f1","G1_f1"]].mean().to_dict())
    return totals


def criteria(run, split):
    saved = (run.seal if split == "validation" else run.status)["assessment"]
    measured = assess(run.table(split, "summary"), run.table(split, "classes"), "G1")
    goals = run.protocol["goals"]
    specs = [
        ("clean_f1","원본 G1 macro F1",saved["clean_macro_f1"],goals["min_clean"]),
        ("clean_drop","원본 macro F1 변화",saved["clean_delta"],-goals["max_clean_loss"]),
        ("avg_gain","잡음 9조건 평균 macro F1 변화",saved["noise_gain"],goals["min_noise_gain"]),
        ("avg_noise_f1","잡음 9조건 평균 G1 macro F1",saved["noise_macro_f1"],goals["min_noise_mean"]),
        ("class_recall","원본 클래스별 재현율 변화 중 최솟값",min(saved["clean_recall_delta"].values()),-goals["max_class_clean_recall_loss"]),
        ("per_sigma","강도별 평균 macro F1 변화 중 최솟값",min(saved["noise_delta_by_sigma"].values()),-goals["max_per_sigma_loss"]),
    ]
    for key in ("clean_macro_f1","baseline_clean","clean_delta","noise_gain","noise_macro_f1"):
        if not np.isclose(saved[key], measured[key], rtol=0, atol=1e-12):
            raise ValueError("저장된 측정값과 평가 CSV의 불일치 · "+key)
    for key in ("clean_recall_delta","noise_delta_by_sigma"):
        if any(not np.isclose(v, measured[key][k], rtol=0, atol=1e-12) for k,v in saved[key].items()):
            raise ValueError("저장된 세부 측정값 불일치 · "+key)
    output = []
    for key, label, value, bound in specs:
        passed = value >= bound - 1e-12
        if passed != saved["checks"][key] or passed != measured["checks"][key]:
            raise ValueError("동결된 채택 기준과 표시값 불일치 · "+key)
        output.append({"기준":label,"기준값":f"≥ {bound:.2f}","측정값":f"{value:.4f}",
                       "판정":"통과" if passed else "미달"})
    return pd.DataFrame(output)
