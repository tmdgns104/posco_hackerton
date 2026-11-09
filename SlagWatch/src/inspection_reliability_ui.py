"""Presentation for paired reliability analysis and local evaluation design records."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .inspection_governance import freeze_design, split_conflicts, verify_design
from .inspection_reliability import defect_size_summary, group_bootstrap, transition_summary


@st.cache_data(show_spinner=False)
def _bootstrap(pairs, repetitions, seed, weighting):
    return group_bootstrap(pairs, repetitions=repetitions, seed=seed, weighting=weighting)


def reliability(pairs, manifest, project):
    from .inspection_brightness import brightness_summary, original_brightness
    st.subheader("오류 전이 및 통계 신뢰성")
    st.caption("저장된 동일 입력의 이진 판정을 분석합니다. 원본 ID를 복원추출하며, 별도 제품·설비 단위의 독립성을 보장하지 않습니다.")
    summary = transition_summary(pairs)
    all_row = summary.iloc[0]
    st.write(f"원본 **{all_row.originals:,}장** · 짝지은 판정 **{all_row.decisions:,}회** · "
             f"원본 ID 기준 재표본 그룹 **{all_row.originals:,}개**")
    names = {
        "condition": "조건", "seed": "seed", "originals": "원본", "decisions": "판정",
        "defect_originals": "불량 원본", "normal_originals": "정상 원본",
        "new_fn": "새 FN", "fixed_fn": "해결 FN", "persistent_fn": "공통 FN",
        "stable_tp": "공통 TP", "new_fp": "새 FP", "fixed_fp": "해결 FP",
        "persistent_fp": "공통 FP", "stable_tn": "공통 TN", "failed": "실패",
    }
    st.dataframe(summary.rename(columns=names), hide_index=True, width="stretch")
    st.caption("FN/FP 전이는 판정 횟수입니다. 개별 원본은 기존 ‘실패 사례’에서 확인할 수 있습니다. "
               "결함 형태와 실제 촬영 조건은 이 분석에서 추정하지 않습니다.")
    st.download_button("오류 전이 CSV", summary.to_csv(index=False).encode("utf-8-sig"),
                       f"{manifest['run_id']}_transitions.csv", "text/csv", on_click="ignore")

    choices = [("ALL", -1)] + [(str(c), int(s)) for c, s in manifest["conditions"]]
    condition = st.selectbox("통계 분석 조건", choices,
                             format_func=lambda x: "전체 조건" if x[0] == "ALL" else f"{x[0]} · seed {x[1]}")
    selected = pairs if condition[0] == "ALL" else pairs[
        pairs.condition.astype(str).eq(condition[0]) & pairs.seed.astype(int).eq(condition[1])]
    with st.expander("원본 이미지 밝기별 오류 전이"):
        st.caption("해시를 확인한 원본 RGB에서 평균(0.2126R + 0.7152G + 0.0722B)을 계산합니다. "
                   "0~1 픽셀 밝기이며 실제 조도나 교란 후 밝기가 아닙니다. 경계는 탐색용입니다.")
        bounds = st.columns(2)
        low = float(bounds[0].number_input("어두움 상한", min_value=0.01, max_value=0.98, value=0.17, step=0.01))
        high = float(bounds[1].number_input("밝음 하한", min_value=0.02, max_value=0.99, value=0.18, step=0.01))
        cache_key = "brightness:" + manifest["run_id"]
        if st.button("원본 밝기 측정", key=cache_key + ":measure"):
            st.session_state.pop(cache_key, None)
            try:
                with st.spinner("원본 무결성을 확인하고 픽셀 밝기를 측정합니다…"):
                    measurements = original_brightness(project, manifest["split"], pairs)
                st.session_state[cache_key] = measurements
            except (ValueError, OSError, KeyError) as exc:
                st.error(f"원본 밝기를 측정하지 못했습니다: {exc}")
        if cache_key in st.session_state:
            if low >= high:
                st.error("어두움 상한은 밝음 하한보다 작아야 합니다.")
            else:
                try:
                    brightness = brightness_summary(selected, st.session_state[cache_key], low_below=low, high_from=high)
                    st.dataframe(brightness.rename(columns={**names, "brightness_group": "밝기 구간"}),
                                 hide_index=True, width="stretch")
                    st.caption(f"low < {low:g} · middle {low:g} 이상 {high:g} 미만 · high {high:g} 이상. "
                               "원본 장수와 조건별 판정 횟수는 다릅니다. 재측정하려면 버튼을 다시 누르세요.")
                    st.download_button("밝기별 전이 CSV", brightness.to_csv(index=False).encode("utf-8-sig"),
                                       f"{manifest['run_id']}_brightness.csv", "text/csv", on_click="ignore")
                except ValueError as exc:
                    st.error(f"밝기별 전이를 계산하지 못했습니다: {exc}")
    if project.id == "ksdd2" and {"defect_pixels", "width", "height"}.issubset(project.index.columns):
        with st.expander("주석 면적별 불량 전이"):
            st.caption("원본 마스크의 결함 픽셀 / 전체 픽셀을 사용합니다. 경계는 탐색용 사용자 설정이며 현장 위험 기준이 아닙니다.")
            bounds = st.columns(2)
            small = float(bounds[0].number_input("Small 상한 (%)", min_value=0.0001, max_value=99.0, value=0.1, format="%.4f"))
            medium = float(bounds[1].number_input("Medium 상한 (%)", min_value=0.0002, max_value=100.0, value=1.0, format="%.4f"))
            if small >= medium:
                st.error("Small 상한은 Medium 상한보다 작아야 합니다.")
            else:
                try:
                    sizes = defect_size_summary(selected, project.index, small_below=small / 100, medium_below=medium / 100)
                except ValueError as exc:
                    st.error(f"주석 면적을 분석할 수 없습니다: {exc}")
                else:
                    st.dataframe(sizes.rename(columns={"size": "면적 구간", "originals": "불량 원본",
                        "decisions": "판정", "new_fn": "새 FN", "fixed_fn": "해결 FN",
                        "persistent_fn": "공통 FN", "stable_tp": "공통 TP", "failed": "실패"}),
                        hide_index=True, width="stretch")
                    st.caption(f"Small < {small:g}% · Medium {small:g}% 이상, {medium:g}% 미만 · Large {medium:g}% 이상. 정상 이미지의 FP에는 적용하지 않습니다.")
    weighting = st.radio("집계 가중치", ["decisions", "originals"], horizontal=True,
                         format_func=lambda x: "모든 판정 동일 가중" if x == "decisions" else "원본별 동일 가중")
    cols = st.columns(2)
    repetitions = int(cols[0].number_input("Bootstrap 반복", min_value=100, max_value=10000, value=2000, step=100))
    seed = int(cols[1].number_input("Bootstrap seed", min_value=0, max_value=2147483647, value=42))
    with st.spinner("원본 그룹을 재표본 추출하고 있습니다."):
        result = _bootstrap(selected, repetitions, seed, weighting)
    labels = {"recall": "불량 재현율", "fn_rate": "미검출률", "fp_rate": "정상 오경보율"}
    models = {"before": manifest["before_model_id"], "after": manifest["after_model_id"],
              "difference": "후보 − 기존"}
    display = result.copy()
    display["metric"] = display.metric.map(labels)
    display["model"] = display.model.map(models)
    for column in ("estimate", "lower", "upper"):
        display[column] = display[column].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
    st.dataframe(display.rename(columns={
        "metric": "지표", "model": "모델", "estimate": "점추정", "lower": "95% 하한",
        "upper": "95% 상한", "valid_repetitions": "유효 재표본", "repetitions": "전체 재표본",
        "originals": "원본 그룹", "class_originals": "해당 정답 원본", "decisions": "판정",
        "seed": "seed", "weighting": "가중치",
    }), hide_index=True, width="stretch")
    st.caption("95% 백분위 구간 · 원본 ID 복원추출 · 기존/후보 쌍 유지. 분모 0인 재표본은 제외합니다. "
               "해당 정답의 추론 실패가 있으면 관련 비율은 N/A입니다.")
    if (result.class_originals.min() < 30 or
            result.valid_repetitions.min() < repetitions * .95 or
            ((result.upper - result.lower).dropna() > .2).any()):
        st.warning("일부 조건의 표본이 적거나 신뢰구간이 넓어 성능 변화의 확정에 주의가 필요합니다. "
                   "통계 결과는 현장 안전성 보증이 아닙니다.")
    chart = go.Figure()
    for side in ("before", "after"):
        rows = result[result.model.eq(side)].dropna(subset=["estimate", "lower", "upper"])
        if not rows.empty:
            chart.add_trace(go.Scatter(
                name=models[side], x=[labels[v] for v in rows.metric], y=rows.estimate,
                mode="markers", marker={"size": 10, "color": "#60bfff" if side == "before" else "#f5ae50"},
                error_y={"type": "data", "symmetric": False,
                         "array": (rows.upper - rows.estimate).clip(lower=0),
                         "arrayminus": (rows.estimate - rows.lower).clip(lower=0), "visible": True}))
    if chart.data:
        chart.update_layout(height=300, yaxis={"tickformat": ".0%", "range": [0, 1]},
                            margin={"l": 8, "r": 8, "t": 20, "b": 10}, legend={"orientation": "h"})
        st.plotly_chart(chart, width="stretch")
    st.download_button("통계 결과 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                       f"{manifest['run_id']}_bootstrap.csv", "text/csv", on_click="ignore")


def governance(project, split, root):
    from .inspection_event_log import record_event
    from .surface_data import ROOT
    st.subheader("평가 설계 기록과 분할 점검")
    st.info("로컬 설계 기록은 공식 시험 접근 통제나 승인 시스템이 아닙니다. "
            "기록 이후 평가와 자동으로 연결되지 않으므로 최종 교체 판정은 미평가입니다.")
    conflicts = split_conflicts(project.index)
    if conflicts.empty:
        st.success("제공된 ID·파일 해시·그룹 필드에서 분할 간 중복이 발견되지 않았습니다.")
    else:
        st.error(f"분할 간 중복 {len(conflicts)}건을 발견했습니다. 해결 전 설계를 동결할 수 없습니다.")
        st.dataframe(conflicts, hide_index=True, width="stretch")
    st.caption("없는 설비·촬영 시점 필드나 유사 이미지는 검사하지 않습니다. 이 검사만으로 외부 독립성을 보장하지 않습니다.")
    st.write(f"데이터 목록 SHA-256: `{project.index_hash}`")
    with st.form("inspection-design-freeze"):
        actor = st.text_input("작성자 (본인 입력; 인증되지 않음)")
        hypothesis = st.text_area("변경 가설과 개발 데이터 근거")
        criteria = st.text_area("평가 전에 정한 성공 기준과 허용 부작용")
        submit = st.form_submit_button("검증 설계 스냅샷 저장")
    if submit:
        try:
            record_event(ROOT/"inspection_events", "design_snapshot", project_id=project.id, outcome="started",
                         details={"split": split})
            path = freeze_design(project, split, root, actor=actor, hypothesis=hypothesis, criteria=criteria)
            record_event(ROOT/"inspection_events", "design_snapshot", project_id=project.id, outcome="completed",
                         details={"split": split, "snapshot_id": path.name})
        except Exception as exc:
            try:
                record_event(ROOT/"inspection_events", "design_snapshot", project_id=project.id, outcome="failed",
                             details={"split": split, "error": str(exc)[:500]})
            except Exception as log_exc:
                st.error(f"이벤트 기록도 실패했습니다: {log_exc}")
            st.error(f"설계 기록 또는 이벤트 기록에 실패했습니다. 저장되었을 수 있으니 목록을 확인하세요: {exc}")
        else:
            st.success(f"설계 스냅샷을 저장했습니다: {path.name}")
    root = Path(root)
    files = sorted(root.glob("*.json"), reverse=True) if root.is_dir() else []
    if files:
        chosen = st.selectbox("저장된 설계 기록", files, format_func=lambda p: p.name)
        if st.button("현재 설정과 기록 비교"):
            try:
                record = verify_design(chosen, project, split)
            except (ValueError, OSError) as exc:
                st.error(f"기록을 검증하지 못했습니다: {exc}")
            else:
                st.success("현재 모델·데이터·조건·코드가 기록과 일치합니다.")
                st.json(record, expanded=False)
