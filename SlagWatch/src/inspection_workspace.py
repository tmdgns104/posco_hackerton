"""Operational views backed by verified local inspection records."""
from html import escape
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from .inspection_activation import activate_model, activation_evidence, active_model, check_compatibility
from .inspection_event_log import read_events
from .inspection_model_registry import list_models, load_registered
from .surface_data import ROOT

ACTIVE = ROOT / "inspection_active_models"
EVENTS = ROOT / "inspection_events"
MODELS = ROOT / "inspection_models"


def annotation_overlay(project, manifest, sample_id, original):
    if project.id != "ksdd2" or not project.official or project.index_hash != manifest.get("source_manifest_sha256"):
        raise ValueError("Verified official KSDD2 annotations are unavailable for this run.")
    rows = project.index.loc[project.index.sample_id.eq(sample_id)]
    if len(rows) != 1:
        raise ValueError("Annotation is not uniquely identified.")
    row = rows.iloc[0]
    project.check_scope(str(row["split"]))
    allowed = (ROOT / "data" / "ksdd2" / "raw" / ("train" if row["split"] == "validation" else "test")).resolve()
    path = (ROOT / str(row["mask_path"])).resolve()
    if not path.is_relative_to(allowed):
        raise ValueError("Annotation path is outside the registered data.")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["mask_sha256"]:
        raise ValueError("Original annotation hash differs.")
    with Image.open(io.BytesIO(payload)) as image:
        mask = np.asarray(image.convert("L")) != 0
    if mask.shape != original.shape[:2] or int(mask.sum()) != int(row["defect_pixels"]):
        raise ValueError("Annotation dimensions or defect area differ.")
    overlay = np.asarray(original, dtype=np.float32).copy()
    overlay[mask] = overlay[mask] * .55 + np.array([1., 0., 0.], dtype=np.float32) * .45
    return overlay


def _jump(project, split, section):
    from .inspection_ui import section_jump
    section_jump(project.id, split, section)


def _adapters(project, ids):
    adapters = dict(project.models)
    for model in ids:
        if model not in adapters:
            adapters[model] = load_registered(MODELS, model)
    return adapters


def introduction(project, split, path, manifest):
    from .demo_ui import current, specimen_strip
    from .inspection_ui import SPLITS, cards, k, table_file

    demo = current()
    st.html(f'''<header class="steel-hero"><div>
<span class="steel-scope">강판 표면 결함 6종 분류 · 공개 NEU-CLS 데이터 · 합성 가우시안 잡음</span>
<h1>철강 표면 결함 AI의 오류를 찾고, 개선 효과를 확인합니다</h1>
<p>SteelTrust는 강판 표면의 결함 6종을 구분하는 AI를 시험합니다. 이미지에 잡음을 더해 오분류가 발생하는 사례를 찾고,
같은 입력에서 후보 모델이 기존 오류를 얼마나 고쳤는지 확인합니다. 새로 생긴 오류도 함께 살펴봅니다.</p></div>
<div class="steel-specimens">{specimen_strip(demo.id, demo)}</div></header>''')
    st.caption("이미지는 공개 NEU-CLS 학습 원본의 결함 종류별 예시입니다. 선택한 프로젝트의 평가 이미지가 아닐 수 있습니다.")
    st.html('''<section class="project-story" aria-labelledby="project-story-title">
<div class="project-story-header">
<span class="project-story-eyebrow">WHY STEELTRUST</span>
<h2 id="project-story-title">좋아진 모델, 정말 더 믿을 수 있을까요?</h2>
<p>입력 변화에서 발견한 오류부터 비교 가능한 검증 근거까지, SteelTrust가 문제를 푸는 과정입니다.</p>
</div>
<ol class="project-story-steps">
<li class="project-story-step"><span class="project-story-number">01</span>
<h3>문제발견</h3><strong>같은 강판도 입력이 달라지면</strong>
<p>원본에서 맞힌 결함을 잡음이 더해진 이미지에서는 잘못 분류할 수 있습니다.</p>
<span class="project-story-tag">입력 변화 · 오분류</span></li>
<li class="project-story-step"><span class="project-story-number">02</span>
<h3>문제정의</h3><strong>평균 성능만으로는 부족합니다</strong>
<p>후보 모델이 기존 오류를 고쳐도, 이전에 맞힌 사례에서 새로운 오류가 생길 수 있습니다.</p>
<span class="project-story-tag">개선과 악화 함께 확인</span></li>
<li class="project-story-step"><span class="project-story-number">03</span>
<h3>해결방법 모색</h3><strong>동일한 입력으로 나란히 시험</strong>
<p>원본과 조건별 잡음 입력을 두 모델에 적용하고, 같은 원본과 seed의 결과를 비교합니다.</p>
<span class="project-story-tag">동일 입력 · 조건별 비교</span></li>
<li class="project-story-step"><span class="project-story-number">04</span>
<h3>해결 방법</h3><strong>오류를 근거로 남기고 다시 검증</strong>
<p>오분류 변화와 실제 사례를 살펴보고 저장해, 모델 비교와 사례 재시험에 활용합니다.</p>
<span class="project-story-tag">사례 탐색 · 저장 · 재시험</span></li>
</ol></section>''')
    st.subheader("선택한 검증 기록 한눈에")
    if not manifest:
        st.info("이 분할에 저장된 평가가 없습니다. 새 검증 실행에서 첫 결과를 만들 수 있습니다.")
        st.button("새 검증 실행", on_click=_jump, args=(project, split, "새 검증 실행"),
                  key=k("intro-run", project.id))
        return
    if project.kind != "binary":
        summary = table_file(path, "summary", manifest)
        changes = table_file(path, "changes", manifest)
        models = summary[summary.scope.eq("all")].set_index("model_id")
        counts = changes[changes.scope.eq("all")].iloc[0]
        before, after = manifest["before_model_id"], manifest["after_model_id"]
        def count(value):
            return f"{int(value):,}" if value != "" and pd.notna(value) else "계산 불가"
        cards([
            ("전체 조건의 오분류", f"{count(models.loc[before, 'completed'] - models.loc[before, 'correct'])} → "
             f"{count(models.loc[after, 'completed'] - models.loc[after, 'correct'])}회", f"{before} → {after}", ""),
            ("바로잡은 오류", f"{count(counts['fixed'])}회", "기존 오답 → 후보 정답", "good"),
            ("새로 발생한 오류", f"{count(counts['new_error'])}회", "기존 정답 → 후보 오답", "caution"),
            ("짝지은 판정", f"{manifest['completed_pairs']:,} / {manifest['planned_pairs']:,}회",
             f"실패 {manifest['failed_pairs']:,}회", ""),
        ])
    scope = "전체 평가" if manifest["scope"] == "full" else "시연 부분집합"
    st.caption(f"{SPLITS.get(split, split)} · {scope} · 고유 원본 {manifest['n_originals']:,}장. "
               "원본과 조건·seed별 반복 판정을 합산한 결과이며 서로 독립된 이미지 수가 아닙니다.")
    if manifest["failed_pairs"]:
        st.warning("완료하지 못한 판정이 있습니다. 실패를 정답이나 오분류로 계산하지 않습니다.")
    st.caption("결과는 선택한 데이터와 시험 조건에 한정되며, 공식 승인이나 실제 생산 설비의 안전성을 보장하지 않습니다.")
    with st.container(horizontal=True):
        st.button("검증 개요 보기", on_click=_jump, args=(project, split, "개요"),
                  key=k("intro-overview", project.id), type="primary")
        st.button("오분류 사례 찾아보기", on_click=_jump, args=(project, split, "실패 사례"),
                  key=k("intro-cases", project.id))
        st.button("개선 전후 비교", on_click=_jump, args=(project, split, "모델 비교"),
                  key=k("intro-compare", project.id))


def overview(project, split, path, manifest):
    from .inspection_ui import _result, k
    active = active_model(ACTIVE, project)
    st.html(f'<div class="workspace-panel"><div class="workspace-eyebrow">Active validation</div>'
            f'<div class="workspace-title">{escape(project.title)}</div>'
            f'<div class="workspace-muted">로컬 활성 모델: {escape(active)} · '
            f'데이터 분할: {escape(split)} · 실제 설비에는 연결되지 않음</div></div>')
    if not manifest:
        st.info("저장된 평가가 없습니다. 모델과 데이터를 확인하고 새 검증을 실행하세요.")
        st.button("새 검증 구성", on_click=_jump, args=(project, split, "새 검증 실행"))
        return
    _result(project, split, path, manifest, overview=True)
    st.markdown("#### 검증 흐름")
    st.caption("● 기록된 평가 완료  →  ◌ 실패 분석 검토 필요  →  ◌ 공식 최종 판단 미평가")
    cols = st.columns(3)
    cols[0].button("모델 비교 열기", on_click=_jump, args=(project, split, "모델 비교"), key=k("overview-comparison", project.id))
    cols[1].button("실패 사례 조사", on_click=_jump, args=(project, split, "실패 사례"), key=k("overview-failures", project.id))
    cols[2].button("판단과 교체 조건", on_click=_jump, args=(project, split, "교체 판단"), key=k("overview-decision", project.id))
    with st.expander("최근 로컬 이벤트"):
        try:
            events = read_events(EVENTS, project_id=project.id, limit=8)
        except Exception as exc:
            st.error(f"이벤트를 읽지 못했습니다: {exc}")
        else:
            if events:
                st.dataframe(pd.DataFrame(events)[["timestamp_utc", "event", "outcome", "details"]],
                             hide_index=True, width="stretch")
            else:
                st.caption("기록된 UI 이벤트가 없습니다.")


def registry(project, split, runs):
    from .inspection_ui import k
    st.subheader("모델 레지스트리")
    registered = list_models(MODELS) if project.kind == "binary" else []
    ids = list(project.models) + registered
    active = active_model(ACTIVE, project)
    model = st.selectbox("모델 객체", ids, index=ids.index(active) if active in ids else 0,
                         key=k("registry-model", project.id))
    st.caption("로컬 활성 모델: " + active + " · 운영 설비 모델이 아닙니다.")
    try:
        adapter = _adapters(project, [model])[model]
        identity = adapter.identity
    except Exception as exc:
        st.error(f"모델을 확인할 수 없습니다: {exc}")
        return
    linked = [(rid, m) for rid, m in runs.items() if model in m.get("models", {})]
    rows = [
        ("역할", "앱 기본 모델" if model in project.models else "신뢰한 로컬 등록 모델"),
        ("입력 계약", "RGB 이미지 / 고정 ResNet18" if project.kind == "binary" else "NEU 이미지 어댑터"),
        ("지원 정답", ", ".join(project.classes)),
        ("관련 실행", str(len(linked))),
        ("파일 SHA-256", identity.get("artifact_sha256", "등록 정보 참조")),
    ]
    st.table(pd.DataFrame(rows, columns=["항목", "내용"]), hide_index=True)
    if linked:
        rid, m = linked[0]
        st.write(f"최근 관련 기록: `{rid}` · {m['before_model_id']} → {m['after_model_id']}")
        if st.button("연결된 평가 열기", key=k("registry-run", project.id, model, rid)):
            from .inspection_ui import remember
            remember(project.id, split, rid)
            st.session_state[k("history", project.id, split)] = rid
            _jump(project, split, "모델 비교")
            st.rerun()
    with st.expander("모델 identity / 계약 상세"):
        st.json(identity, expanded=False)
    if project.kind == "binary":
        st.button("모델 등록으로 이동", on_click=_jump, args=(project, split, "프로젝트 설정"),
                  key=k("registry-add", project.id))
    else:
        st.info("NEU 모델 파일 추가 등록은 아직 지원하지 않습니다.")
        st.button("프로젝트 설정", on_click=_jump, args=(project, split, "프로젝트 설정"),
                  key=k("registry-settings", project.id))


def decision_workspace(project, split, path, manifest):
    from .inspection_ui import _event, _result, jread, k
    st.subheader("교체 판단")
    st.warning("여기서의 변경은 이 앱의 로컬 작업 기준만 바꿉니다. 공식 승인·현장 안전 보증·생산 설비 배포가 아닙니다.")
    if not manifest:
        st.info("평가 기록이 없어 판단할 수 없습니다.")
        return
    decision = jread(path / "decision.json")
    before, after = manifest["before_model_id"], manifest["after_model_id"]
    active = active_model(ACTIVE, project)
    st.write(f"**판단 대상:** {before} → {after} · {project.title} · {manifest['split']} · {manifest['scope']}")
    st.write(f"**개발 검증 결과:** {decision['label']} · **현재 로컬 활성 모델:** {active}")
    for reason in decision.get("reasons", [])[:4]:
        st.write("•", reason)
    st.caption("독립 시험 승인, 사전 설계와 실행의 강제 연결 및 인증된 공식 승인 절차는 제공하지 않습니다.")
    _result(project, split, path, manifest, overview=True)
    if project.kind != "binary" or project.template_id != "ksdd2" or split != "validation":
        st.info("이 프로젝트에서는 로컬 모델 변경이 아직 지원되지 않습니다.")
        return
    if before != active:
        st.warning("선택한 실행의 기준 모델이 현재 로컬 활성 모델과 다릅니다. 현재 모델을 기준으로 다시 평가하세요.")
        return
    token = k("compatibility", project.id, manifest["run_id"], before, after)
    if st.button("호환성 및 전체 검증 확인", key=k("compat-check", token)):
        st.session_state.pop(token, None)
        try:
            adapters = _adapters(project, [before, after])
            check_compatibility(project, before, after, adapters)
            evidence = activation_evidence(project, path, before, after, adapters)
        except Exception as exc:
            st.error(f"교체 조건을 충족하지 못했습니다: {exc}")
        else:
            st.session_state[token] = evidence
            st.success("입력 계약과 전체 개발 검증 기준을 확인했습니다. 아래 사용자 확인 후 로컬 변경이 가능합니다.")
    if token not in st.session_state:
        return
    with st.form(k("activate-form", token)):
        confirmed = st.checkbox("독립 시험·공식 승인·설비 배포가 아님을 이해하고 로컬 모델을 변경합니다.")
        submit = st.form_submit_button("로컬 활성 모델 변경")
    if submit:
        details = {"run_id": manifest["run_id"], "before_model_id": before, "after_model_id": after}
        try:
            _event(project.id, "local_model_activation", "started", details)
            adapters = _adapters(project, [before, after])
            from . import security
            auth_token = st.session_state.get("security-token")
            actor = security.authorize(auth_token)["username"]
            state = activate_model(ACTIVE, project, path, before, after, adapters, actor=actor, confirmed=confirmed,
                                   auth_token=auth_token)
            _event(project.id, "local_model_activation", "completed", details)
        except Exception as exc:
            try:
                _event(project.id, "local_model_activation", "failed", {**details, "error": str(exc)[:500]})
            except Exception as log_exc:
                st.error(f"이벤트 기록도 실패했습니다: {log_exc}")
            st.error(f"변경 또는 이벤트 기록 실패: {exc}. 변경이 저장되었을 수 있으므로 현재 모델을 확인하세요.")
        else:
            st.session_state.pop(token, None)
            st.session_state.pop(k("baseline", project.id, split), None)
            st.session_state.pop(k("candidate", project.id, split), None)
            st.session_state[k("activation-message", project.id)] = state["model_id"]
            st.rerun()


def reports(project, split, path, manifest):
    from .inspection_ui import jread
    st.subheader("보고서 및 근거")
    if not manifest:
        st.info("저장된 평가 기록이 없습니다.")
        return
    st.caption("선택한 실행의 원본 파일을 제공합니다. 공식 승인 보고서가 아닙니다.")
    decision = jread(path / "decision.json")
    st.json({"run_id": manifest["run_id"], "models": [manifest["before_model_id"], manifest["after_model_id"]], "split": split,
             "scope": manifest["scope"], "decision": decision}, expanded=False)
    st.download_button("판정 JSON", (path / "decision.json").read_bytes(),
                       f"{manifest['run_id']}_decision.json", "application/json", on_click="ignore")
    st.download_button("조건별 결과 CSV", (path / "summary.csv").read_bytes(),
                       f"{manifest['run_id']}_summary.csv", "text/csv", on_click="ignore")
