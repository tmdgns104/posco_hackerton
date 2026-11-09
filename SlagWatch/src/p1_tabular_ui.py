"""Separate local tabular lab; existing image workspaces are unchanged."""
import hashlib
import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from . import p1_tabular as lab
from . import p1_abstention as policy_lab
from . import p1_confidence
from . import neu_image_demo
from . import neu_uploaded_models
from . import security
from .inspection_event_log import read_events
from .surface_data import ROOT as APP_ROOT

ROOT = APP_ROOT / "p1_tabular_local"
NEU_DEMO_ROOT = ROOT / "neu_image_demo"


def _show_error(title, exc):
    st.error(f"{title}: {exc}")


def _neu_upload():
    st.subheader("강판 이미지 모델 업로드 및 V0 비교")
    st.warning("joblib 로드 시 임의 코드가 실행될 수 있습니다. 출처를 신뢰하는 파일만 등록하세요. 해시는 안전성 검사가 아닙니다.")
    st.caption("기존 NEU 번들(.joblib): 동일 ResNet18 인코더, 지원 전처리, 512차원 분류기 및 6개 클래스 순서가 필요합니다. 다른 형식이나 인코더는 지원하지 않습니다.")
    with st.form("neu-upload-form"):
        model_id = st.text_input("이미지 모델 ID", help="영문으로 시작하는 영문·숫자·_·- 2~40자")
        upload = st.file_uploader("신뢰한 NEU 이미지 번들", type=["joblib"], max_upload_size=128)
        trusted = st.checkbox("파일을 신뢰하며 로드할 때 코드가 실행될 수 있음을 이해합니다.")
        submitted = st.form_submit_button("이미지 모델 등록")
    if submitted:
        try:
            if upload is None:
                raise ValueError("모델 파일을 선택하세요.")
            neu_uploaded_models.register(neu_uploaded_models.REGISTRY, model_id, upload.getvalue(), upload.name,
                                         trusted=trusted, auth_token=st.session_state.get("security-token"))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            _show_error("이미지 모델 등록 실패", exc)
        else:
            st.success("이미지 모델이 등록되었습니다. 아래에서 V0와 실제 비교를 실행하세요.")
    models = neu_uploaded_models.list_models(neu_uploaded_models.REGISTRY)
    if not models:
        st.info("등록된 업로드 이미지 모델이 없습니다.")
        return
    names = [item["model_id"] for item in models]
    chosen = st.selectbox("V0와 비교할 업로드 모델", names, key="neu-upload-candidate")
    scope = st.radio("이미지 비교 표본", ["전체 검증 이미지", "클래스별 시연 표본"], horizontal=True)
    st.caption("모든 선택 표본을 원본과 9개 잡음·seed 조건에서 V0와 동일한 입력으로 평가합니다. 전체 실행은 시간이 걸릴 수 있습니다.")
    if st.button("업로드 모델과 V0 실제 비교 실행"):
        try:
            from .inspection_projects import load_project
            ids = None if scope == "전체 검증 이미지" else load_project("neu_cls6").demo_ids("validation")
            with st.spinner("V0와 업로드 모델로 동일 입력을 추론합니다."):
                path = neu_uploaded_models.evaluate(neu_uploaded_models.REGISTRY, chosen,
                    auth_token=st.session_state.get("security-token"), sample_ids=ids)
            cfg, _ = neu_uploaded_models.load_run(neu_uploaded_models.REGISTRY, path.name)
        except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
            _show_error("이미지 모델 비교 실패", exc)
        else:
            st.success(f"비교 기록 {path.name} 저장 · 완료 {cfg['completed_pairs']}/{cfg['planned_pairs']} · 실패 {cfg['failed_pairs']}회. 모델 비교에서 확인하세요.")


def _neu_image_demo():
    st.subheader("강판 이미지 모델 V1·V2 교체 시험")
    st.caption("프로젝트에 동봉된 V1·V2를 연결해 검증 이미지 12장으로 시험합니다. 기존 개요·NEU 작업공간·생산 설비 모델은 변경하지 않습니다.")
    st.warning("이는 작은 표본의 로컬 교체 시연이며 공식 승인이나 성능 보증이 아닙니다.")
    models = neu_image_demo.registered_models(NEU_DEMO_ROOT)
    registered = {item["model_id"] for item in models}
    for model_id in neu_image_demo.MODEL_IDS:
        col = st.columns([1, 3])
        if model_id in registered:
            col[0].success(f"{model_id} 등록됨")
        elif col[0].button(f"{model_id} 연결 등록", key=f"neu-demo-register-{model_id}"):
            try:
                neu_image_demo.register_builtin(NEU_DEMO_ROOT, model_id,
                                                auth_token=st.session_state.get("security-token"))
            except (ValueError, OSError, KeyError) as exc:
                _show_error("이미지 모델 등록 실패", exc)
            else:
                st.rerun()
        col[1].caption(f"동봉된 {model_id}.joblib 및 해당 이미지 전처리를 사용합니다.")
    if len(registered) < 2:
        return
    active = neu_image_demo.active_model(NEU_DEMO_ROOT)
    candidate = next(mid for mid in neu_image_demo.MODEL_IDS if mid != active)
    st.write(f"현재 **이미지 시험용 활성 모델: {active}** · 비교 후보: {candidate}")
    if st.button("V1·V2 동일 이미지 비교 실행", key="neu-demo-compare"):
        try:
            with st.spinner("두 모델로 같은 검증 이미지를 추론합니다."):
                result = neu_image_demo.compare(NEU_DEMO_ROOT, active, candidate)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            _show_error("이미지 모델 비교 실패", exc)
        else:
            st.session_state["neu-demo-selected-run"] = result["run_id"]
            st.session_state["neu-demo-history"] = result["run_id"]
            st.rerun()
    runs = neu_image_demo.list_runs(NEU_DEMO_ROOT)
    if not runs:
        st.info("동일 이미지 비교를 먼저 실행하세요.")
        return
    selected = st.session_state.get("neu-demo-selected-run")
    run_id = st.selectbox("이미지 비교 기록", runs, index=runs.index(selected) if selected in runs else 0,
                          key="neu-demo-history")
    report = neu_image_demo.load_run(NEU_DEMO_ROOT, run_id)
    st.write(f"**{report['before_model_id']} {report['correct_before']}/{report['n_samples']}** → "
             f"**{report['after_model_id']} {report['correct_after']}/{report['n_samples']}** · "
             f"시연 조건: {'통과' if report['gate_passed'] else '미통과'}")
    with st.expander("이미지별 예측 결과"):
        st.dataframe(pd.DataFrame(report["samples"]), hide_index=True, width="stretch")
    confirmed = st.checkbox("공식 승인이나 설비 배포가 아닌 이미지 시험용 교체임을 확인합니다.",
                            key="neu-demo-confirm")
    if st.button("이미지 시험 활성 모델 교체", key="neu-demo-activate",
                 disabled=not report["gate_passed"] or report["before_model_id"] != active):
        try:
            with st.spinner("저장된 비교를 다시 확인하고 교체합니다."):
                state = neu_image_demo.activate(NEU_DEMO_ROOT, run_id, confirmed=confirmed,
                                                auth_token=st.session_state.get("security-token"))
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            _show_error("이미지 시험 모델 교체 실패", exc)
        else:
            st.session_state["neu-demo-message"] = f"이미지 시험 활성 모델: {state['previous_model_id']} → {state['model_id']}"
            st.rerun()
    message = st.session_state.pop("neu-demo-message", None)
    if message:
        st.success(message)
    if st.button("현재 이미지 시험 모델로 다시 추론", key="neu-demo-retest"):
        try:
            with st.spinner("현재 활성 이미지 모델을 다시 실행합니다."):
                result = neu_image_demo.evaluate_active(NEU_DEMO_ROOT)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            _show_error("이미지 모델 재시험 실패", exc)
        else:
            st.success(f"{result['model_id']} 재추론: {result['correct']}/{result['samples']}장 정분류")


def _registered():
    return lab.list_models(ROOT)


def _register():
    st.subheader("표형 모델 등록")
    st.warning("joblib은 로딩할 때 임의 코드를 실행할 수 있습니다. 출처를 신뢰하는 파일만 등록하세요. 해시는 안전성 검사가 아닙니다.")
    st.caption("학습된 scikit-learn LogisticRegression 또는 RandomForestClassifier만 지원합니다. 전처리는 모델 외부에서 완료해야 합니다. 다른 프레임워크와 Pipeline은 미지원입니다.")
    with st.form("p1-register"):
        model_id = st.text_input("모델 ID", help="영문으로 시작하는 영문·숫자·_·- 2~40자")
        name = st.text_input("모델 이름")
        version = st.text_input("모델 버전", value="1")
        features = st.text_input("숫자형 피처 이름 (쉼표로 학습 순서대로)")
        source = st.radio("파일 제공 방식", ["업로드", "서버의 절대 경로"], horizontal=True)
        uploaded = st.file_uploader("신뢰한 .joblib 모델 파일", type=["joblib"], max_upload_size=128)
        local_path = st.text_input("서버 절대 경로 (경로 방식일 때만)")
        trusted = st.checkbox("이 파일을 신뢰하며 로드 시 코드가 실행될 수 있음을 이해합니다.")
        submitted = st.form_submit_button("모델 등록")
    if submitted:
        try:
            if not trusted:
                raise lab.ContractError("TRUST_REQUIRED", "신뢰 확인이 필요합니다.")
            if source == "업로드":
                if uploaded is None:
                    raise lab.ContractError("MODEL_FILE_INVALID", "모델 파일을 선택하세요.")
                payload, filename = uploaded.getvalue(), uploaded.name
            else:
                path = Path(local_path)
                if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > lab.MAX_MODEL_BYTES:
                    raise lab.ContractError("MODEL_FILE_INVALID", "128 MiB 이하의 일반 로컬 파일 절대 경로가 필요합니다.")
                payload, filename = path.read_bytes(), path.name
            lab.register_model(ROOT, payload, filename, model_id, name, version,
                               [x.strip() for x in features.split(",")], trusted=trusted,
                               auth_token=st.session_state.get("security-token"))
        except (ValueError, OSError, TypeError) as exc:
            _show_error("모델을 등록하지 못했습니다", exc)
        else:
            st.success(f"{model_id} 등록 완료")
            st.rerun()


def _registry(models):
    st.subheader("등록 모델")
    if not models:
        st.info("등록된 표형 모델이 없습니다. 아래에서 신뢰한 모델을 등록하세요.")
    else:
        st.dataframe(pd.DataFrame([{
            "이름": m["model_name"], "ID": m["model_id"], "버전": m["model_version"],
            "종류": m["model_type"], "프레임워크": m["framework"], "어댑터": m["adapter_type"],
            "상태": m["status"], "SHA-256": m["model_hash"],
        } for m in models]), hide_index=True, width="stretch")
        selected = st.selectbox("상세 모델", [m["model_id"] for m in models], key="p1-details")
        metadata = next(m for m in models if m["model_id"] == selected)
        with st.expander("모델 계약 상세"):
            st.json(metadata)
        if metadata["status"] == "ready":
            cols = st.columns(2)
            if cols[0].button("모델 로드 확인", key="p1-validate-model"):
                try:
                    lab.TabularAdapter(ROOT, selected)
                except (ValueError, OSError, TypeError) as exc:
                    _show_error("모델 검증 실패", exc)
                else:
                    st.success("등록 계약과 모델 해시를 확인했습니다. 데이터와의 호환성은 별도 확인이 필요합니다.")
            if cols[1].button("모델 등록 해제", key="p1-disable-model"):
                try:
                    lab.deactivate_model(ROOT, selected, auth_token=st.session_state.get("security-token"))
                except (ValueError, OSError, TypeError) as exc:
                    _show_error("등록 해제 실패", exc)
                else:
                    st.rerun()
    with st.expander("+ 모델 추가", expanded=not models):
        _register()


def _comparison(models):
    st.subheader("표형 모델 비교 및 로컬 교체")
    st.caption("정답 CSV 전체를 개발 검증에 사용합니다. 동일 데이터·seed·잡음 조건을 두 모델에 적용하며, 학습 데이터와의 독립성은 사용자가 확인해야 합니다.")
    available = [m["model_id"] for m in models if m["status"] == "ready"]
    if len(available) < 2:
        st.info("비교하려면 사용 가능한 서로 다른 모델 두 개를 등록하세요.")
        return
    active = lab.active_model(ROOT)
    next_active = st.session_state.pop("p1-next-active", None)
    if next_active and next_active[0] in available and next_active[1] in available:
        st.session_state["p1-before"], st.session_state["p1-after"] = next_active
    if st.session_state.get("p1-before") not in available:
        st.session_state["p1-before"] = active if active in available else available[0]
    if st.session_state.get("p1-after") not in available:
        st.session_state["p1-after"] = next(x for x in available if x != st.session_state["p1-before"])
    def swap():
        st.session_state["p1-before"], st.session_state["p1-after"] = st.session_state["p1-after"], st.session_state["p1-before"]
    left, right = st.columns(2)
    before = left.selectbox("기준 모델", available, key="p1-before")
    after = right.selectbox("후보 모델", available, key="p1-after")
    st.button("기준·후보 바꾸기", on_click=swap, key="p1-swap")
    st.caption(f"현재 표형 로컬 활성 모델: {active or '미설정'} · 선택 위치 바꾸기는 활성 모델을 변경하지 않습니다.")
    message = st.session_state.pop("p1-activation-message", None)
    if message:
        st.success(message)
    data = st.file_uploader("정답 표 데이터 CSV", type=["csv"], max_upload_size=25, key="p1-dataset")
    st.caption("CSV 열 순서: sample_id, true_label, 모델에 등록한 숫자 피처들. sample_id는 고유해야 하며 빈 값·NaN·Inf는 허용하지 않습니다.")
    seed = int(st.number_input("공통 seed", min_value=0, max_value=2**32-1, value=42, step=1))
    noise = float(st.number_input("가우시안 잡음 표준편차 (각 피처의 원래 단위)", min_value=0.0, value=0.0, step=0.01, format="%.4f"))
    classes = next(m["classes"] for m in models if m["model_id"] == before)
    positive = st.selectbox("FN/FP 양성 클래스 (이진 분류만)", classes, key="p1-positive") if len(classes) == 2 else None
    payload = data.getvalue() if data else None
    token = hashlib.sha256(json.dumps([before, after, hashlib.sha256(payload).hexdigest() if payload else None],
                                      ensure_ascii=False).encode()).hexdigest()
    if st.button("모델·데이터 호환성 검사", key="p1-compat"):
        st.session_state.pop("p1-compatible", None)
        try:
            if payload is None:
                raise lab.ContractError("DATASET_INVALID", "CSV를 업로드하세요.")
            adapters = [lab.TabularAdapter(ROOT, before), lab.TabularAdapter(ROOT, after)]
            lab.validate_dataset(payload, adapters)
        except (ValueError, OSError, TypeError) as exc:
            _show_error("호환성 검사 실패", exc)
            try:
                lab._event(ROOT, "compatibility", "failed", {"before": before, "after": after, "error": str(exc)[:500]})
            except OSError as log_exc:
                _show_error("호환성 오류 기록 실패", log_exc)
        else:
            st.session_state["p1-compatible"] = token
            st.success("피처, 순서, 값과 클래스 계약을 확인했습니다.")
    if st.session_state.get("p1-compatible") == token:
        st.caption("호환성 확인됨 · 실행 시 모델과 데이터 계약을 다시 확인합니다.")
    st.info("로컬 시연 기준: 각 조건에서 후보 정확도가 기준 모델보다 낮지 않고 모든 추론이 완료되어야 교체할 수 있습니다. 공식 승인 기준이 아닙니다.")
    if st.button("동일 조건 비교 실행", disabled=payload is None or st.session_state.get("p1-compatible") != token,
                 type="primary", key="p1-run"):
        progress = st.progress(0, text="표형 모델 비교 실행 중")
        try:
            report = lab.run_comparison(ROOT, before, after, payload, seed=seed, noise_std=noise,
                                        positive=positive, progress=lambda done, total:
                                        progress.progress(done / total, text=f"추론 {done:,} / {total:,}"))
        except (ValueError, OSError, TypeError) as exc:
            _show_error("비교 실행 실패", exc)
        else:
            st.session_state["p1-run-id"] = report["run_id"]
            st.session_state["p1-history"] = report["run_id"]
            st.rerun()


def _history():
    st.subheader("실행 기록 및 보고서")
    runs = ROOT / "runs"
    ids = sorted((p.name for p in runs.iterdir() if p.is_dir() and not p.is_symlink()
                  and not p.name.startswith(".")), reverse=True) if runs.exists() else []
    if not ids:
        st.info("표형 비교 기록이 없습니다.")
        return
    wanted = st.session_state.get("p1-run-id")
    run_id = st.selectbox("저장된 표형 실행", ids, index=ids.index(wanted) if wanted in ids else 0, key="p1-history")
    try:
        report = lab.load_run(ROOT, run_id)
    except (ValueError, OSError, KeyError) as exc:
        _show_error("실행 기록 무결성 확인 실패", exc)
        return
    st.write(f"**실행:** {report['status']}　·　**로컬 기준:** {report['decision']}　·　"
             f"**모델:** {report['before_model_id']} → {report['after_model_id']}　·　"
             f"**원본:** {report['n_originals']:,}행　·　**추론 실패:** {report['failed_predictions']:,}회")
    st.caption("COMPLETED는 평가 절차 종료이며 PASS나 공식 승인과 다릅니다. 지표는 성공한 추론만 기준이며 실패가 있으면 교체를 차단합니다.")
    metrics = pd.DataFrame(report["metrics"])
    st.dataframe(metrics, hide_index=True, width="stretch")
    import plotly.express as px
    chart = px.bar(metrics, x="condition", y="accuracy", color="model_id", barmode="group",
                   labels={"condition": "조건", "accuracy": "정확도", "model_id": "모델"})
    chart.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(chart, width="stretch", key="p1-comparison-chart")
    st.markdown("#### 동일 입력의 오류 변화")
    st.dataframe(pd.DataFrame(report["transitions"]), hide_index=True, width="stretch")
    with st.expander("표형 통계적 불확실성"):
        st.caption("sample_id 단위의 짝 Bootstrap 95% 백분위 구간입니다. 같은 ID의 반복 조건은 함께 추출합니다. 서로 다른 ID의 독립성은 보장하지 않습니다.")
        repetitions = int(st.number_input("Bootstrap 반복 횟수", min_value=1, max_value=10000, value=2000,
                                          step=100, key="p1-bootstrap-count"))
        boot_seed = int(st.number_input("Bootstrap seed", min_value=0, max_value=2**32-1, value=42,
                                        step=1, key="p1-bootstrap-seed"))
        cache_key = f"p1-bootstrap-{run_id}"
        if st.button("신뢰구간 계산", key="p1-bootstrap-run"):
            st.session_state.pop(cache_key, None)
            try:
                st.session_state[cache_key] = p1_confidence.analyze_run(ROOT, run_id, repetitions=repetitions, seed=boot_seed)
            except (ValueError, OSError, KeyError) as exc:
                _show_error("신뢰구간 계산 실패", exc)
        if cache_key in st.session_state:
            confidence = st.session_state[cache_key]
            st.dataframe(confidence, hide_index=True, width="stretch")
            st.download_button("신뢰구간 CSV", confidence.to_csv(index=False).encode("utf-8-sig"),
                               f"{run_id}_confidence.csv", "text/csv", on_click="ignore")
    folder = runs / run_id
    with st.expander("실패와 모델별 예측"):
        predictions = pd.read_csv(folder / "predictions.csv", keep_default_na=False)
        st.dataframe(predictions, hide_index=True, width="stretch")
    with st.expander("실행 설정과 제한"):
        st.json(report, expanded=False)
    st.caption("JSON과 CSV는 같은 저장 실행의 결과입니다. HTML은 이 저장된 수치를 그대로 표시하며 재추론하지 않습니다.")
    report_bytes = (folder / "report.json").read_bytes()
    html = ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>SteelTrust report</title>"
            "<h1>SteelTrust 표형 비교 보고서</h1><p>로컬 개발 검증 · 공식 승인 아님</p><pre>"
            + escape(report_bytes.decode("utf-8")) + "</pre></html>").encode("utf-8")
    cols = st.columns(4)
    cols[0].download_button("보고서 JSON", report_bytes, f"{run_id}_report.json", "application/json", on_click="ignore")
    cols[1].download_button("지표 CSV", (folder / "summary.csv").read_bytes(), f"{run_id}_summary.csv", "text/csv", on_click="ignore")
    cols[2].download_button("사례 CSV", (folder / "pairs.csv").read_bytes(), f"{run_id}_pairs.csv", "text/csv", on_click="ignore")
    cols[3].download_button("보고서 HTML", html, f"{run_id}_report.html", "text/html", on_click="ignore")
    with st.expander("로컬 활성 모델 변경"):
        st.warning("로그인과 감사 기록이 필요합니다. 이 변경은 생산 설비, NEU 모델과 공식 승인에 영향을 주지 않습니다.")
        with st.form("p1-replacement-form"):
            confirm = st.checkbox("위 제한을 이해하고 표형 로컬 활성 모델을 변경합니다.", key="p1-confirm")
            submitted = st.form_submit_button("로컬 모델 교체 확인",
                                              disabled=report["status"] != "COMPLETED" or report["decision"] != "PASS")
        if submitted:
            try:
                auth_token = st.session_state.get("security-token")
                actor = security.authorize(auth_token)["username"]
                state = lab.replace_model(ROOT, run_id, actor=actor, confirmed=confirm, auth_token=auth_token)
            except (ValueError, OSError, TypeError, KeyError) as exc:
                _show_error("로컬 모델 변경 실패", exc)
                st.caption("이벤트 기록 오류가 발생한 경우 변경이 저장되었을 수 있습니다. 현재 모델을 다시 확인하세요.")
            else:
                st.session_state["p1-activation-message"] = f"표형 로컬 활성 모델 변경: {state['previous_model_id']} → {state['model_id']}"
                st.session_state["p1-next-active"] = (state["model_id"], state["previous_model_id"])
                st.rerun()


def _abstention():
    st.subheader("보류 정책 실험")
    st.caption("모델 교체 판정과 별개인 개발 실험입니다. 예측 클래스의 확률이 동결 임계값 이상이면 자동 전달, 그 외에는 보류합니다. 보류가 정확히 재검토되었다고 가정하지 않습니다.")
    runs = ROOT / "runs"
    ids = sorted((p.name for p in runs.iterdir() if p.is_dir() and not p.is_symlink()
                  and not p.name.startswith(".")), reverse=True) if runs.exists() else []
    if not ids:
        st.info("정책을 동결하려면 개발 데이터의 표형 비교 실행이 필요합니다.")
        return
    with st.expander("개발 데이터에서 정책 생성 및 동결"):
        dev_id = st.selectbox("개발 실행", ids, key="p1-policy-dev")
        try:
            dev = lab.load_run(ROOT, dev_id)
        except (ValueError, OSError, KeyError) as exc:
            _show_error("개발 실행을 읽지 못했습니다", exc)
            return
        model = st.selectbox("정책 대상 모델", [dev["before_model_id"], dev["after_model_id"]], key="p1-policy-model")
        condition = st.selectbox("동결할 입력 조건", dev["config"]["conditions"], key="p1-policy-condition")
        name = st.text_input("정책 이름", value="신뢰도 정책", key="p1-policy-name")
        version = st.text_input("정책 버전", value="1", key="p1-policy-version")
        target = st.selectbox("개발 데이터 목표 coverage", policy_lab.TARGETS,
                              format_func=lambda value: f"{value:.0%}", key="p1-policy-target")
        st.caption("개발 예측의 점수만으로 임계값을 정합니다. 동점이면 실제 coverage가 목표보다 높을 수 있습니다.")
        if st.button("정책 생성·동결", key="p1-freeze"):
            try:
                frozen = policy_lab.freeze_policy(ROOT, dev_id, model, condition, target, name=name, version=version)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                _show_error("정책 동결 실패", exc)
            else:
                st.success(f"동결 완료: {frozen['policy_id']} · 임계값 {frozen['threshold']:.4f} · 개발 coverage {frozen['development_coverage']:.1%}")
    folder = ROOT / "policies"
    policy_ids = sorted((p.stem for p in folder.glob("policy-*.json") if not p.is_symlink()), reverse=True) if folder.exists() else []
    if not policy_ids:
        st.info("동결된 정책이 없습니다.")
        return
    policies = [policy_lab.load_policy(ROOT, policy_id) for policy_id in policy_ids]
    labels = {p["policy_id"]: f"{p['name']} v{p['version']} · {p['model_id']} · {p['condition']} · 목표 {p['target_coverage']:.0%} · {p['policy_id'][-8:]}"
              for p in policies}
    chosen = st.multiselect("비교할 동결 정책", policy_ids, format_func=labels.get, key="p1-policy-selection")
    evaluation = st.selectbox("별도 평가 실행", ids, key="p1-policy-evaluation")
    st.caption("개발·평가 파일 해시와 sample_id가 달라야 합니다. 이 검사만으로 학습/평가 독립성이 증명되지는 않습니다.")
    if st.button("보류 정책 실험 실행", disabled=not chosen, key="p1-policy-run"):
        try:
            result = policy_lab.evaluate_policies(ROOT, chosen, evaluation)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            _show_error("보류 정책 실험 실패", exc)
        else:
            st.session_state["p1-policy-result"] = result["experiment_id"]
            st.session_state["p1-policy-history"] = result["experiment_id"]
            st.rerun()
    experiments = ROOT / "policy_experiments"
    exp_ids = sorted((p.name for p in experiments.iterdir() if p.is_dir() and not p.is_symlink()
                      and p.name.startswith("abstention-")), reverse=True) if experiments.exists() else []
    if not exp_ids:
        return
    current = st.session_state.get("p1-policy-result")
    exp_id = st.selectbox("저장된 보류 실험", exp_ids, index=exp_ids.index(current) if current in exp_ids else 0,
                          key="p1-policy-history")
    result = policy_lab.load_experiment(ROOT, exp_id)
    metrics = pd.DataFrame(result["metrics"])
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.caption("비율의 분자·분모를 함께 표시합니다. eligible은 성공한 추론만 포함하며 실패 수를 별도 표시합니다. 불필요한 보류는 정답을 이용한 사후 분석입니다.")
    import plotly.express as px
    for column, title in (("automatic_error_rate", "Coverage–자동 전달 오류"),
                          ("fn_rate", "Coverage–FN"), ("fp_rate", "Coverage–FP"),
                          ("abstention_rate", "Coverage–보류율")):
        if metrics[column].notna().any():
            chart = px.scatter(metrics, x="coverage", y=column, color="model_id",
                               hover_data=["target_coverage", "coverage_gap", "automatic", "held"], title=title)
            chart.update_xaxes(tickformat=".0%", range=[0, 1])
            chart.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(chart, width="stretch", key=f"p1-policy-chart-{column}")
    case_path = experiments / exp_id / "cases.csv"
    with st.expander("자동 전달 및 보류 사례"):
        cases = pd.read_csv(case_path, dtype={"sample_id": str}, keep_default_na=False)
        st.dataframe(cases[["policy_id", "sample_id", "true_label", "predicted_class", "confidence", "automatic"]],
                     hide_index=True, width="stretch")
    cols = st.columns(2)
    cols[0].download_button("보류 실험 JSON", (experiments / exp_id / "report.json").read_bytes(),
                            f"{exp_id}.json", "application/json", on_click="ignore")
    cols[1].download_button("보류 사례 CSV", case_path.read_bytes(), f"{exp_id}_cases.csv",
                            "text/csv", on_click="ignore")


def page():
    st.title("표형 모델 검증 실험실")
    st.warning("별도 개발 시연 공간입니다. NEU 이미지 검증과 기존 시험 잠금은 변경하지 않습니다. 임시 로컬 감사 저장소는 외부 독립 저장소가 아닙니다.")
    try:
        _neu_upload()
        _neu_image_demo()
        models = _registered()
        _registry(models)
        _comparison(models)
        _history()
        _abstention()
        with st.expander("로컬 이벤트 기록"):
            events = read_events(ROOT / "events", project_id="p1_tabular")
            if events:
                st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
            else:
                st.caption("기록된 이벤트가 없습니다.")
        with st.expander("인증된 작업 감사 기록 (임시 로컬)"):
            token = st.session_state.get("security-token")
            if not token:
                st.caption("로그인 후 확인할 수 있습니다. 외부 서버 사용 시 외부 서버에서 기록을 확인하세요.")
            else:
                events = security.read_local_audit(token)
                if events:
                    st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
                else:
                    st.caption("로컬 감사 기록이 없습니다.")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _show_error("표형 실험실을 표시하지 못했습니다", exc)
