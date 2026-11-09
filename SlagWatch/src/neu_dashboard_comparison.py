"""Verified historical V0/V1/V2 paired-validation comparisons."""
import numpy as np
import pandas as pd

from .surface_data import CFG
from .surface_replay import SurfaceRun
from .surface_study import conditions

CANDIDATES = ("V1", "V2")
SCOPES = ("전체 조건", "원본", "잡음 전체", "잡음 σ 0.02", "잡음 σ 0.05", "잡음 σ 0.10")


def validated_pairs(run):
    pairs = run.table("validation", "pairs").copy()
    required = {"sample_id", "parent_image_id", "label", "sigma", "seed", "before", "after",
                "prediction_before", "prediction_after", "raw_input_hash_before",
                "raw_input_hash_after", "model_run_id_before", "model_run_id_after", "change"}
    if not required.issubset(pairs.columns) or pairs.empty or pairs[list(required)].isna().any().any():
        raise ValueError("V0·V1·V2 비교에 필요한 검증 기록이 누락되었습니다.")
    if set(pairs["before"]) != {"V0"} or set(pairs["after"]) != set(CANDIDATES):
        raise ValueError("검증 비교의 기준 또는 후보 모델이 일치하지 않습니다.")
    keys = ["sample_id", "sigma", "seed"]
    expected_conditions = set(conditions("validation"))
    sample_rows = run.meta[run.meta["split"].eq("validation")]
    expected_samples = set(sample_rows["sample_id"])
    expected_keys = {(sample, sigma, seed) for sample in expected_samples
                     for sigma, seed in expected_conditions}
    for candidate in CANDIDATES:
        rows = pairs[pairs["after"].eq(candidate)]
        if rows.duplicated(keys).any() or set(map(tuple, rows[keys].itertuples(index=False, name=None))) != expected_keys:
            raise ValueError(f"{candidate}의 원본·조건·seed 비교 범위가 완전하지 않습니다.")
        if not rows["raw_input_hash_before"].eq(rows["raw_input_hash_after"]).all():
            raise ValueError(f"{candidate}가 V0와 같은 시험 입력을 사용하지 않았습니다.")
        before, after = run.detector("V0"), run.detector(candidate)
        if (not rows["model_run_id_before"].eq(before.run_id).all()
                or not rows["model_run_id_after"].eq(after.run_id).all()):
            raise ValueError("저장된 예측과 동봉 모델의 identity가 다릅니다.")
        metadata = sample_rows.set_index("sample_id")
        for field in ("label", "parent_image_id"):
            if not rows[field].eq(rows["sample_id"].map(metadata[field])).all():
                raise ValueError("저장된 예측과 검증 원본 목록이 다릅니다.")
        old_correct = rows["prediction_before"].eq(rows["label"])
        new_correct = rows["prediction_after"].eq(rows["label"])
        expected = np.select([~old_correct & new_correct, old_correct & ~new_correct,
                              ~old_correct & ~new_correct],
                             ["fixed", "new_error", "both_wrong"], default="both_correct")
        if not rows["change"].eq(expected).all():
            raise ValueError("저장된 오류 변화와 실제 예측이 일치하지 않습니다.")
    reference = pairs[pairs["after"].eq("V1")].set_index(keys).sort_index()
    other = pairs[pairs["after"].eq("V2")].set_index(keys).sort_index()
    for field in ("label", "parent_image_id", "raw_input_hash_before", "prediction_before", "model_run_id_before"):
        if not reference[field].equals(other[field]):
            raise ValueError("V1과 V2 비교의 V0 기준 입력 또는 예측이 일치하지 않습니다.")
    return pairs


def select_scope(pairs, scope):
    if scope == "전체 조건":
        return pairs.copy()
    if scope == "원본":
        return pairs[pairs["sigma"].eq(0)].copy()
    if scope == "잡음 전체":
        return pairs[pairs["sigma"].gt(0)].copy()
    scopes = {f"잡음 σ {sigma:.2f}": sigma for sigma in CFG["sigmas"]}
    if scope not in scopes:
        raise ValueError("지원하지 않는 비교 조건입니다.")
    return pairs[np.isclose(pairs["sigma"], scopes[scope])].copy()


def summarize(pairs):
    result = []
    for candidate in CANDIDATES:
        rows = pairs[pairs["after"].eq(candidate)]
        if rows.empty:
            raise ValueError(f"{candidate}의 선택 조건 결과가 없습니다.")
        before_errors = int(rows["prediction_before"].ne(rows["label"]).sum())
        after_errors = int(rows["prediction_after"].ne(rows["label"]).sum())
        fixed = int(rows["change"].eq("fixed").sum())
        new_errors = int(rows["change"].eq("new_error").sum())
        both_wrong = int(rows["change"].eq("both_wrong").sum())
        if before_errors != fixed + both_wrong or after_errors != new_errors + both_wrong:
            raise ValueError("오류 전이 합계가 모델별 오류 수와 다릅니다.")
        result.append({"후보": candidate, "원본 이미지": rows["sample_id"].nunique(),
                       "짝지은 판정": len(rows), "V0 오류": before_errors, "후보 오류": after_errors,
                       "고친 오류": fixed, "새 오류": new_errors, "둘 다 오류": both_wrong,
                       "순 변화": after_errors - before_errors})
    return pd.DataFrame(result)


def load_comparison():
    return validated_pairs(SurfaceRun())
