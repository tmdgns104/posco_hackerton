"""Historical and uploaded V0 comparisons inside the model comparison workspace."""
import pandas as pd
import streamlit as st

from . import neu_dashboard_comparison as comparison
from . import neu_uploaded_models
from .surface_replay import CHANGES, LABELS


def render():
    st.subheader("기본 모델 V0와 추가 모델 비교")
    st.markdown("#### 동봉 모델의 기존 검증 기록")
    st.caption("동봉된 V0·V1·V2의 기존 전체 검증 기록입니다. V1·V2 등록 시험의 12장 결과와는 별개이며 새 추론을 실행하지 않습니다.")
    st.caption("동일한 검증 이미지와 교란 입력을 사용하며, V1과 V2는 각자의 가우시안 필터와 학습된 분류기를 적용합니다.")
    try:
        pairs = comparison.load_comparison()
        scope = st.selectbox("V0·V1·V2 비교 조건", comparison.SCOPES, key="neu-overview-comparison-scope")
        selected = comparison.select_scope(pairs, scope)
        summary = comparison.summarize(selected)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"V0·V1·V2 비교 기록을 확인할 수 없습니다: {exc}")
        return
    columns = st.columns(2)
    for column, row in zip(columns, summary.to_dict("records"), strict=True):
        change = int(row["순 변화"])
        verdict = "변화 없음" if change == 0 else f"{abs(change):,}회 {'증가' if change > 0 else '감소'}"
        with column:
            st.markdown(f"#### V0 → {row['후보']}")
            st.metric("오분류", f"{row['V0 오류']:,} → {row['후보 오류']:,}회", verdict,
                      delta_color="inverse")
            st.write(f"고친 오류 **{row['고친 오류']:,}회** · 새 오류 **{row['새 오류']:,}회**")
            st.caption(f"둘 다 오분류 {row['둘 다 오류']:,}회 · 짝지은 판정 {row['짝지은 판정']:,}회")
    st.caption("순 변화 = 새 오류 − 고친 오류입니다. 새 오류는 V0 정답 → 후보 오답이며, 고친 오류는 V0 오답 → 후보 정답입니다.")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.bar_chart(summary.set_index("후보")[["V0 오류", "후보 오류", "고친 오류", "새 오류"]],
                 color=["#455a74", "#7856f7", "#229970", "#d46a4c"])
    st.caption("원본 360장입니다. 전체는 원본 360회와 잡음 9조건 3,240회를 합한 3,600회이며 반복 판정은 독립 이미지가 아닙니다.")
    st.caption("NEU는 결함 6종 분류 자료이므로 정상/불량 FN·FP가 아닌 결함 종류 간 오분류입니다. 공개 데이터의 개발 검증이며 공식 승인·현장 성능이 아닙니다.")
    with st.expander("조건별 오류 변화와 사례", expanded=False):
        conditions = []
        for item in comparison.SCOPES[1:]:
            frame = comparison.summarize(comparison.select_scope(pairs, item)).copy()
            frame.insert(0, "조건", item)
            conditions.append(frame)
        st.dataframe(pd.concat(conditions, ignore_index=True), hide_index=True, width="stretch")
        model = st.selectbox("사례 비교 후보", comparison.CANDIDATES, key="neu-overview-case-model")
        change = st.selectbox("사례 변화 유형", ["fixed", "new_error", "both_wrong"],
                              format_func=CHANGES.get, key="neu-overview-case-change")
        cases = selected[selected["after"].eq(model) & selected["change"].eq(change)].copy()
        cases = cases[["sample_id", "label", "sigma", "seed", "prediction_before", "prediction_after"]]
        cases = cases.rename(columns={"sample_id": "원본 ID", "label": "정답", "sigma": "잡음 σ",
                                      "seed": "seed", "prediction_before": "V0 예측",
                                      "prediction_after": f"{model} 예측"})
        for name in ("정답", "V0 예측", f"{model} 예측"):
            cases[name] = cases[name].map(LABELS)
        st.caption(f"선택 조건에서 {CHANGES[change]} {len(cases):,}건")
        st.dataframe(cases, hide_index=True, width="stretch", height=250)
        st.download_button("선택 조건 사례 CSV", cases.to_csv(index=False).encode("utf-8-sig"),
                           f"neu_v0_{model}_{change}.csv", "text/csv", key="neu-overview-case-download")


def _uploaded_results(runs=None, *, key_prefix="neu-upload"):
    if runs is None:
        runs = neu_uploaded_models.list_runs(neu_uploaded_models.REGISTRY)
    if not runs:
        st.info("업로드 모델의 V0 비교 기록이 없습니다. 모델 등록·교체 화면에서 모델을 등록하고 실제 비교를 실행하세요.")
        return
    run_id = st.selectbox("업로드 모델 비교 기록", runs, key=f"{key_prefix}-run")
    try:
        cfg, pairs = neu_uploaded_models.load_run(neu_uploaded_models.REGISTRY, run_id)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        st.error(f"업로드 모델 평가 기록을 확인할 수 없습니다: {exc}")
        return
    candidate = cfg["after_model_id"]
    conditions = ["전체 조건"] + list(dict.fromkeys(pairs["condition"]))
    scope = st.selectbox("업로드 모델 비교 조건", conditions, key=f"{key_prefix}-scope")
    rows = pairs if scope == "전체 조건" else pairs[pairs["condition"].eq(scope)]
    completed = rows[rows["status"].eq("ok")]
    failed = len(rows) - len(completed)
    before_errors = int(completed["before_pred"].ne(completed["true_label"]).sum())
    after_errors = int(completed["after_pred"].ne(completed["true_label"]).sum())
    fixed = int(completed["change"].eq("fixed").sum())
    new = int(completed["change"].eq("new_error").sum())
    st.markdown(f"#### V0 → {candidate}")
    st.metric("업로드 모델 오분류", f"{before_errors:,} → {after_errors:,}회",
              f"{after_errors - before_errors:+,}회", delta_color="inverse")
    st.write(f"고친 오류 **{fixed:,}회** · 새 오류 **{new:,}회** · 둘 다 오류 **{int(completed['change'].eq('both_wrong').sum()):,}회**")
    st.bar_chart(pd.DataFrame({"V0 오류": [before_errors], "후보 오류": [after_errors],
                               "고친 오류": [fixed], "새 오류": [new]}, index=[candidate]),
                 color=["#455a74", "#7856f7", "#229970", "#d46a4c"])
    st.caption(f"검증 {'전체' if cfg['scope'] == 'full' else '부분집합'} · 원본 {rows['sample_id'].nunique():,}장 · "
               f"완료 {len(completed):,}/{len(rows):,}회 · 실패 {failed:,}회 · 실행 {run_id}")
    st.caption("같은 원본·조건·seed의 반복 판정이며 독립 이미지가 아닙니다. 두 모델이 모두 성공한 판정만 오류 수에 포함합니다.")
    if failed:
        st.warning("추론 실패가 있어 비교가 불완전합니다. 실패는 정답이나 오분류로 계산하지 않습니다.")
    with st.expander("업로드 모델 비교 사례 및 실패"):
        st.dataframe(rows[["sample_id", "true_label", "condition", "seed", "before_pred", "after_pred",
                           "before_status", "after_status", "change"]], hide_index=True, width="stretch", height=250)
        if failed:
            failures = pd.read_csv(neu_uploaded_models.REGISTRY / "runs" / run_id / "failures.csv",
                                   keep_default_na=False)
            if scope != "전체 조건":
                failures = failures[failures["condition"].eq(scope)]
            st.dataframe(failures[["sample_id", "condition", "seed", "model_id", "error"]],
                         hide_index=True, width="stretch")
        st.download_button("업로드 모델 비교 CSV", rows.to_csv(index=False).encode("utf-8-sig"),
                           f"neu_{run_id}.csv", "text/csv", key=f"{key_prefix}-download")


def render_recent_comparisons():
    from . import inspection_runner
    from .inspection_projects import load_project

    st.subheader("최근 등록 모델과 V0 비교")
    st.caption("아래 영역에서 동봉 모델의 기존 결과를 확인하고 스크롤해 등록 모델과 평가 기록을 선택하세요. 이전 비교 화면은 이 영역 아래에 유지됩니다.")
    with st.container(height=650, border=True):
        render()
        st.divider()
        st.markdown("#### 업로드 모델 비교")
        try:
            models = neu_uploaded_models.list_models(neu_uploaded_models.REGISTRY)
            runs = neu_uploaded_models.list_runs(neu_uploaded_models.REGISTRY)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            st.error(f"등록 모델 또는 비교 기록을 읽을 수 없습니다: {exc}")
            return
        if not models:
            st.info("등록된 업로드 이미지 모델이 없습니다. 사이드바의 모델 등록·교체에서 NEU 이미지 번들을 등록하세요.")
            return
        by_model = {item["model_id"]: [] for item in models}
        for run_id in runs:
            try:
                cfg = inspection_runner.load_run(neu_uploaded_models.REGISTRY / "runs" / run_id)
                if (cfg["run_id"] != run_id or cfg["project_id"] != "neu_cls6"
                        or cfg["split"] != "validation" or cfg["before_model_id"] != "V0"):
                    raise ValueError("V0 검증 기록이 아닙니다.")
                if cfg["after_model_id"] not in by_model:
                    raise ValueError("등록된 후보 모델을 찾을 수 없습니다.")
                by_model[cfg["after_model_id"]].append(run_id)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                st.warning(f"비교 기록 {run_id}을 목록에 표시할 수 없습니다: {exc}")
        model_ids = sorted(by_model, key=lambda model: (by_model[model][0] if by_model[model] else "", model),
                           reverse=True)
        model_id = st.selectbox("비교할 등록 모델", model_ids, key="neu-compare-model")
        st.caption("V0와 같은 검증 이미지·조건·seed로 실제 추론한 기록만 표시합니다.")
        with st.expander("선택 모델을 V0와 새로 평가", expanded=not by_model[model_id]):
            st.warning("신뢰한 joblib 모델을 다시 로드합니다. 평가에는 로그인과 감사 기록이 필요하며 전체 검증은 시간이 걸릴 수 있습니다.")
            scope = st.radio("새 비교 표본", ["전체 검증 이미지", "클래스별 시연 표본"],
                             horizontal=True, key="neu-compare-new-scope")
            if st.button("선택 모델과 V0 비교 실행", key="neu-compare-evaluate"):
                try:
                    ids = None if scope == "전체 검증 이미지" else load_project("neu_cls6").demo_ids("validation")
                    with st.spinner("선택 모델과 V0로 동일 입력을 추론합니다."):
                        path = neu_uploaded_models.evaluate(neu_uploaded_models.REGISTRY, model_id,
                            auth_token=st.session_state.get("security-token"), sample_ids=ids)
                    st.session_state["neu-compare-run"] = path.name
                    st.rerun()
                except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
                    st.error(f"V0 비교를 실행하지 못했습니다: {exc}")
        if not by_model[model_id]:
            st.info("이 모델의 V0 비교 기록이 없습니다. 위에서 새 평가를 실행하세요.")
            return
        st.caption("비교 기록은 최신순입니다. 다른 조건이나 표본의 실행은 합산하지 않습니다.")
        _uploaded_results(by_model[model_id], key_prefix="neu-compare")
