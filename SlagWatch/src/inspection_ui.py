"""Interactive review of immutable inference runs. The same UI handles both tasks."""
from __future__ import annotations

import hashlib
import io
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .demo_design import heading

NAVIGATION = {
    "Workspace": ["프로젝트 소개", "개요", "모델 비교", "실패 사례"],
    "Models & Evaluation": ["모델 레지스트리", "새 검증 실행", "전이·통계 분석", "사례 모음·재시험"],
    "Governance": ["교체 판단", "평가 설계 기록", "보고서", "이벤트 기록", "프로젝트 설정"],
}
SECTIONS = [section for group in NAVIGATION.values() for section in group]
SPLITS = {"validation": "검증 · validation", "test": "동결 후 독립 시험 · test"}
LABELS = {"normal": "정상", "defect": "불량", "crazing": "망상 균열", "inclusion": "개재물",
          "patches": "반점", "pitted_surface": "표면 패임", "rolled_in_scale": "압연 스케일", "scratches": "긁힘"}
KINDS = {"new_fn":"새 미검출", "new_fp":"새 오경보", "fixed_fn":"고친 미검출", "fixed_fp":"해소한 오경보",
         "persistent_fn":"공통 미검출", "stable_tp":"공통 불량 검출", "persistent_fp":"공통 오경보", "stable_tn":"공통 정상 판정",
         "fixed":"고친 오분류", "new_error":"새 오분류", "both_wrong":"둘 다 틀림", "both_right":"둘 다 맞힘",
         "failed":"추론 실패"}
CONDS = {"ALL":"전체 조건", "clean":"원본", "original":"원본", "dark":"밝기 감소", "bright":"밝기 증가",
         "noise_weak":"약한 잡음", "noise_strong":"강한 잡음"}
NAMES = {"model_id":"모델", "condition":"입력 조건", "seed":"seed", "n_originals":"원본 장수", "planned":"예정",
         "completed":"완료", "failed":"실패", "correct":"정답", "accuracy":"정확도", "recall":"재현율",
         "miss_rate":"미검출률", "fpr":"정상 오경보율", "n_defect":"실제 불량", "n_normal":"실제 정상", "class_label":"실제 클래스", "tp":"TP", "fn":"FN",
         "fp":"FP", "tn":"TN", "macro_f1":"macro F1", **KINDS}
NAMES["failed"] = "실패"


def label(x):
    return LABELS.get(str(x), str(x))


def cname(x):
    if str(x).startswith("noise_0."):
        return f"가우시안 잡음 σ {str(x)[6:]}"
    return CONDS.get(str(x), str(x))


def cell_label(cond, seed):
    return f"{cname(cond)} · seed {seed}"


def k(*parts):
    return "inspection_" + hashlib.sha256(json.dumps(parts, ensure_ascii=False, default=str).encode()).hexdigest()[:22]


def show_error(title, exc):
    st.error(title)
    with st.expander("오류 상세"):
        st.text(str(exc))


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


@st.cache_data(show_spinner=False, max_entries=36)
def _csv(path, digest):
    # Caller verified this exact immutable run; the content digest is part of the cache key.
    return pd.read_csv(path, dtype={"sample_id": str, "group_id":str, "source_id":str, "true_label":str,
                                   "before_pred":str, "after_pred":str}, keep_default_na=False)


def table_file(path, name, manifest):
    return _csv(str(path / (name + ".csv")), manifest["run_id"] + str(manifest.get("files", {}).get(name + ".csv", "")))


def runs_for(root, project_id, split):
    # Untrusted upload metadata is never used to make a path. Select only enumerated local runs.
    found = {}
    for entry in sorted(Path(root).glob("*/manifest.json"), reverse=True):
        try:
            m = jread(entry)
            if m["project_id"] == project_id and m["split"] == split and m["run_id"] == entry.parent.name:
                found[m["run_id"]] = m
        except (ValueError, OSError, KeyError):
            continue
    return found


def remember(pid, split, rid):
    st.session_state[k("active",pid,split)] = rid


def section_jump(pid, split, section):
    st.session_state[k("section",pid,split)] = section


def pick_run(project, split, runs, root, *, interactive):
    if not runs:
        return None, None
    ids = list(runs)
    active = st.session_state.get(k("active",project.id,split))
    if active not in runs:
        active = ids[0]
    if interactive:
        def fmt(rid):
            m = runs[rid]
            return f"{'전체' if m['scope']=='full' else '시연 부분집합'} · {m['n_originals']}장 · {rid}"
        widget = k("history", project.id, split)
        rid = st.selectbox("저장된 실행 기록", ids, index=0 if widget in st.session_state else ids.index(active), format_func=fmt, key=widget)
        remember(project.id, split, rid)
    else:
        rid = active
    from .inspection_runner import load_run
    path = Path(root) / rid
    return path, load_run(path)


def field_table(rows):
    st.table(pd.DataFrame(rows, columns=["항목", "내용"]), hide_index=True, border="horizontal")


def download(label_, path, name, key, mime="text/csv"):
    st.download_button(label_, path.read_bytes(), name, mime, key=key, on_click="ignore")


def cards(items):
    def card(n, v, note, tone):
        return (f'<article class="steel-card {escape(tone)}"><span>{escape(str(n))}</span>'
                f'<strong>{escape(str(v))}</strong><small>{escape(str(note))}</small></article>')
    st.html('<div class="steel-cards">' + ''.join(card(*x) for x in items) + '</div>')


def format_frame(df, columns):
    data = df[[c for c in columns if c in df.columns]].copy()
    if "condition" in data:
        data["condition"] = data.condition.map(cname)
    if "class_label" in data:
        data["class_label"] = data.class_label.map(label)
    for col in ("recall", "miss_rate", "fpr", "accuracy", "macro_f1"):
        if col in data:
            numeric = pd.to_numeric(data[col], errors="coerce")
            data[col] = numeric.map(lambda val: f"{val:.1%}" if pd.notna(val) else "계산 불가")
    return data.rename(columns=NAMES)


def _ids(project):
    if isinstance(project.models, dict):
        return list(project.models)
    return [m if isinstance(m, str) else m.model_id for m in project.models]


def _event(project_id, event, outcome, details=None):
    from .inspection_event_log import record_event
    from .surface_data import ROOT
    return record_event(ROOT / "inspection_events", event, project_id=project_id, outcome=outcome, details=details)


def _model_root():
    from .surface_data import ROOT
    return ROOT / "inspection_models"


def _steel_projects():
    from .inspection_projects import BUILTIN, DATASETS, list_projects
    from .surface_g1 import read_json
    projects = list_projects()
    return {pid: title for pid, title in projects.items()
            if pid == "neu_cls6" or pid not in BUILTIN and
            read_json(DATASETS / pid / "project.json")["template_id"] == "neu_cls6"}


def _case_model_view(project, manifest, extra=None):
    from .inspection_model_registry import ModelView, load_registered
    models = dict(project.models)
    needed = [extra] if extra else [manifest["before_model_id"], manifest["after_model_id"]]
    for model in needed:
        if model not in models:
            models[model] = load_registered(_model_root(), model)
    return ModelView(project, models)


def _short(v):
    if v is None or v == "":
        return "계산 불가"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _criteria_view(project, split, d, m):
    criteria = d.get("criteria", [])
    if not criteria:
        for reason in d.get("reasons", []):
            st.write(reason)
        return
    def row(c):
        return {"시험 항목":c["reason"], "조건":cname(c["condition"]), "seed":"—" if c["seed"] in (None,-1) else str(c["seed"]),
                "관측값":_short(c["value"]), "기준":f"{c['operator']} {_short(c['limit'])}" if c["limit"] is not None else "—",
                "판정":{True:"통과",False:"미달",None:"확인 불가"}.get(c["passed"], "확인 불가")}
    failed = [c for c in criteria if c["passed"] is not True]
    if failed:
        chosen = st.selectbox("미달·확인 불가 항목", range(len(failed)), key=k("criterion",m["run_id"]),
                              format_func=lambda i:f"{cname(failed[i]['condition'])} · {failed[i]['reason']}")
        c = failed[chosen]
        def go():
            metric = str(c["metric"])
            target = ("normal" if "fpr" in metric else "defect" if "recall" in metric else "ALL") if project.kind=="binary" else next(
                (name for name in project.classes if c["id"]=="class_recall" and name in c["reason"]), "ALL")
            st.session_state[k("focus",m["run_id"])] = (c["condition"], c["seed"], metric, target)
            section_jump(project.id, split, "실패 사례")
        st.button("이 기준의 실제 사례 보기", on_click=go, key=k("go-failure",m["run_id"]))
    with st.expander(f"사전 기준별 근거 · {len(criteria)}개"):
        st.dataframe(pd.DataFrame(row(c) for c in criteria), hide_index=True, width="stretch")
        if d.get("representative"):
            st.caption("규약에 정한 대표 평가 범위의 집계")
            st.json(d["representative"], expanded=False)


def _plot(summary, metrics, rid):
    import plotly.graph_objects as go
    cells = summary[summary.scope == "cell"]
    metric = st.radio("그래프 지표", metrics, format_func=lambda a:NAMES[a], horizontal=True, key=k("plot",rid))
    fig = go.Figure()
    xs = []
    for (mid, group), color in zip(cells.groupby("model_id", sort=False), ["#7856f7","#14b8a6"]):
        x = [cell_label(c,s) for c,s in zip(group.condition,group.seed)]
        if not xs:
            xs = x
        fig.add_bar(name=str(mid), x=x, y=pd.to_numeric(group[metric], errors="coerce"), marker_color=color,
                    hovertemplate="%{x}<br>%{y:.2%}<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="group", height=295, margin=dict(l=8,r=8,t=10,b=10), yaxis=dict(range=[0,1],tickformat=".0%"),
                      xaxis=dict(categoryorder="array",categoryarray=xs), legend=dict(orientation="h", y=1.13))
    st.plotly_chart(fig, width="stretch", key=k("chart",rid,metric))


def _result(project, split, path, m, *, overview=False):
    rid = m["run_id"]
    d = jread(path / "decision.json")
    b, a = m["before_model_id"], m["after_model_id"]
    summary = table_file(path,"summary",m)
    changes = table_file(path,"changes",m)
    all_models = summary[summary.scope == "all"].set_index("model_id")
    counts = changes[changes.scope == "all"].iloc[0]
    def n(v):
        return f"{int(v):,}" if v != "" and pd.notna(v) else "계산 불가"
    def trans(col):
        return f"{n(all_models.loc[b,col])} → {n(all_models.loc[a,col])}"
    scope = "전체 평가" if m["scope"] == "full" else "시연 부분집합 · 전체 채택 판정 불가"
    st.subheader(str(d["label"]))
    st.caption(f"{scope}　·　{SPLITS.get(split,split)}　·　기존 {b} → 후보 {a}　·　고유 원본 {m['n_originals']:,}장")
    if project.kind == "binary":
        cards([("불량 미검출 (FN)", trans("fn"), "기존 → 후보 · 불량을 정상으로 판정", ""),
               ("정상 오경보 (FP)", trans("fp"), "기존 → 후보 · 정상을 불량으로 판정", ""),
               ("새 미검출", n(counts["new_fn"]), "기존 검출 → 후보 미검출", "caution"),
               ("새 오경보", n(counts["new_fp"]), "기존 정상 → 후보 오경보", "caution")])
        st.write(f"고친 미검출 **{n(counts['fixed_fn'])}회**　·　새 미검출 **{n(counts['new_fn'])}회**　·　"
                 f"해소한 오경보 **{n(counts['fixed_fp'])}회**　·　새 오경보 **{n(counts['new_fp'])}회**")
        if not overview:
            matrix = pd.DataFrame([
                {"비교 항목": "불량 미검출", "기존 → 후보": trans("fn"), "상태": "불완전" if m["failed_pairs"] else "감소" if all_models.loc[a,"fn"] < all_models.loc[b,"fn"] else "증가" if all_models.loc[a,"fn"] > all_models.loc[b,"fn"] else "변화 없음"},
                {"비교 항목": "정상 오경보", "기존 → 후보": trans("fp"), "상태": "불완전" if m["failed_pairs"] else "감소" if all_models.loc[a,"fp"] < all_models.loc[b,"fp"] else "증가" if all_models.loc[a,"fp"] > all_models.loc[b,"fp"] else "변화 없음"},
                {"비교 항목": "노이즈 강건성", "기존 → 후보": "기록된 시험 조건에서 확인", "상태": "분석 가능"},
                {"비교 항목": "저대비 결함", "기존 → 후보": "저대비 정답 라벨 없음", "상태": "미평가"},
                {"비교 항목": "통계적 불확실성", "기존 → 후보": "전이·통계 분석에서 확인", "상태": "별도 분석"},
                {"비교 항목": "최종 승인", "기존 → 후보": "독립 시험 및 승인 절차 필요", "상태": "미평가"},
            ])
            with st.expander("비교 항목 요약"):
                st.dataframe(matrix, hide_index=True, width="stretch")
    else:
        cards([("기존 모델의 오분류", n(all_models.loc[b,"completed"]-all_models.loc[b,"correct"]), b, ""),
               ("후보 모델의 오분류", n(all_models.loc[a,"completed"]-all_models.loc[a,"correct"]), a, ""),
               ("고친 오분류", n(counts["fixed"]), "기존 오답 → 후보 정답", "good"),
               ("새 오분류", n(counts["new_error"]), "기존 정답 → 후보 오답", "caution")])
        st.write(f"고친 오분류 **{n(counts['fixed'])}회** · 새 오분류 **{n(counts['new_error'])}회**")
        st.caption("NEU에는 정상 클래스가 없습니다. 여섯 결함 종류 사이의 오분류를 비교합니다.")
    st.caption(f"짝지은 판정: 예정 {m['planned_pairs']:,}회 · 완료 {m['completed_pairs']:,}회 · 실패 {m['failed_pairs']:,}회. "
               "같은 원본에 조건과 seed를 반복한 횟수로, 서로 독립적인 제품 수가 아닙니다.")
    if m["failed_pairs"]:
        st.warning("실패는 정상이나 정답으로 계산하지 않습니다. 해당 클래스의 예정 건수가 채워지지 않으면 그 비율은 계산 불가로 표시합니다.")
    if not m.get("official", False):
        st.info("이 실행은 개발용 비교입니다. 공식 승인이나 공식 시험 접근에 사용할 수 없습니다.")
    if m.get("official", False) and project.official and project.kind=="binary" and m["scope"]=="full" and split=="validation":
        st.caption("이 화면은 개발 검증 결과이며 독립 최종 시험의 교체 승인 판정이 아닙니다.")
        if d["status"]=="fail":
            st.warning("교체 판단: 후보 미채택. 전체 검증에서 사전 기준을 충족하지 못해 이 결과를 근거로 공식 시험을 열지 않습니다.")
        elif d["status"]=="pass":
            st.info("검증 분할에서 시제품 기준을 충족했습니다. 최종 판단에는 모델 확정 후 독립 시험이 필요합니다.")
    detailed = {c["reason"]: c for c in d.get("criteria", [])}
    lines = []
    for reason in d.get("reasons", [])[:3]:
        c = detailed.get(reason)
        if c and c["id"] in {"absolute_recall", "absolute_fpr", "relative_recall", "relative_fpr"} and c["value"] is not None:
            title = {"absolute_recall":"불량 재현율","absolute_fpr":"정상 오경보율",
                     "relative_recall":"불량 재현율 하락","relative_fpr":"정상 오경보율 상승"}[c["id"]]
            unit = "%p" if c["id"].startswith("relative") else "%"
            v, bound = float(c["value"])*100, float(c["limit"])*100
            direction = "이상" if c["operator"]==">=" else "이하"
            lines.append(f"{cell_label(c['condition'],c['seed'])} · {title} {v:.2f}{unit} (기준 {bound:g}{unit} {direction})")
        else:
            lines.append(str(reason))
    for reason in lines:
        st.write("•",reason)
    _criteria_view(project, split, d, m)
    metrics = ["recall","miss_rate","fpr"] if project.kind == "binary" else ["accuracy"] + (["macro_f1"] if "macro_f1" in summary else [])
    cols = ["model_id","n_originals","planned","completed","failed"] + (["tp","fn","fp","tn"] if project.kind=="binary" else ["correct"]) + metrics
    st.subheader("전체 및 원본 성능")
    st.caption("전체는 원본을 포함한 모든 조건·seed의 합계입니다. 원본은 추가 교란이 없는 입력만 셉니다.")
    orig = str(m["conditions"][0][0])
    subset = summary[(summary.scope=="all") | (summary.condition.astype(str)==orig)]
    st.table(format_frame(subset, ["condition"]+cols), hide_index=True, border="horizontal")
    with st.expander("조건·seed별 전후 비교", expanded=True):
        _plot(summary,metrics,rid)
        st.dataframe(format_frame(summary[summary.scope=="cell"],["condition","seed","n_defect","n_normal"]+cols),hide_index=True,width="stretch")
        st.caption("오류 변화 유형별 횟수")
        st.dataframe(format_frame(changes[changes.scope=="cell"],["condition","seed","n_originals","planned","completed","failed"]+
                                  [q for q in KINDS if q != "failed"]),hide_index=True,width="stretch")
    with st.expander("클래스별 분모와 성능"):
        if project.kind == "binary":
            st.caption("실제 불량이 재현율·미검출률의 분모이고 실제 정상이 오경보율의 분모입니다. 분모가 0이거나 해당 클래스의 추론이 누락되면 비율을 계산하지 않습니다.")
        klass = table_file(path,"classes",m)
        st.dataframe(format_frame(klass,["condition","seed","model_id","class_label","n_originals","planned","completed","failed","correct","recall"]),
                     hide_index=True,width="stretch")
        if project.kind=="binary":
            st.caption("항상 정상 또는 항상 불량으로 판정하는 단순 기준")
            st.dataframe(format_frame(table_file(path,"baselines",m),["condition","seed","n_defect","n_normal"]+cols),hide_index=True,width="stretch")
        by_class = table_file(path,"changes_by_class",m)
        st.caption("클래스별 판정 변화 횟수")
        st.dataframe(format_frame(by_class,["condition","seed","class_label","n_originals","planned","completed","failed"]+
                                  [q for q in KINDS if q!="failed"]),hide_index=True,width="stretch")
    with st.expander("실행 기록 및 CSV 다운로드",expanded=True):
        st.code(rid,language=None)
        csvs = [("조건별 결과 CSV","summary"),("원본별 비교 CSV","pairs"),("변화 유형 CSV","changes"),("클래스별 결과 CSV","classes"),("클래스별 변화 CSV","changes_by_class")]
        if (path/"failures.csv").is_file():
            csvs.append(("실패 기록 CSV","failures"))
        with st.container(horizontal=True):
            for name,base in csvs:
                download(name,path/(base+".csv"),f"{rid}_{base}.csv",k("csv",rid,base))
        st.caption(f"규약 SHA-256 · {m['protocol_sha256']}")


def _multiclass_analysis(project, split, path, m):
    st.subheader("조건별 오분류와 모델 성능")
    st.caption("선택한 실행 기록의 동일 원본·조건·seed를 비교합니다. 판정 횟수와 독립 원본 수는 다릅니다.")
    summary = table_file(path, "summary", m)
    cells = summary[summary.scope.eq("cell")]
    metrics = ["accuracy"] + (["macro_f1"] if "macro_f1" in cells else [])
    _plot(summary, metrics, m["run_id"])
    st.dataframe(format_frame(cells, ["condition", "seed", "model_id", "n_originals", "planned",
                                      "completed", "failed", "correct"] + metrics),
                 hide_index=True, width="stretch")
    changes = table_file(path, "changes", m)
    st.markdown("#### 조건별 오류 변화")
    st.dataframe(format_frame(changes[changes.scope.eq("cell")],
                             ["condition", "seed", "n_originals", "planned", "completed",
                              "failed", "fixed", "new_error", "both_wrong", "both_right"]),
                 hide_index=True, width="stretch")
    with st.expander("결함 종류별 재현율과 오분류"):
        classes = table_file(path, "classes", m)
        st.dataframe(format_frame(classes[classes.scope.eq("cell")],
                                 ["condition", "seed", "model_id", "class_label", "n_originals",
                                  "planned", "completed", "failed", "correct", "recall"]),
                     hide_index=True, width="stretch")
        st.caption("macro F1은 각 결함 종류의 F1을 동일 가중한 값입니다. 추론 실패가 있으면 해당 수치 해석에 주의하세요.")
    conditions = [(str(c), int(seed)) for c, seed in m["conditions"]]
    chosen = st.selectbox("사례를 확인할 조건", conditions, format_func=lambda value: cell_label(*value),
                          key=k("analysis-condition", m["run_id"]))
    def open_cases():
        st.session_state[k("focus", m["run_id"])] = (chosen[0], chosen[1], "macro_f1", "ALL")
        section_jump(project.id, split, "실패 사례")
    st.button("이 조건의 실패 사례 보기", on_click=open_cases, key=k("analysis-cases", m["run_id"]))
    st.caption("이진 FN/FP 전이와 원본 단위 Bootstrap은 다중 클래스 결과에 적용하지 않습니다.")


def _run(project, split, root):
    from .inspection_runner import run_project, load_run
    from .inspection_model_registry import list_models, load_registered, ModelView
    import time
    st.subheader("새 검증 실행")
    st.write("등록된 원본에서 시험 입력을 만들고, 같은 입력을 기존 모델과 후보 모델에 넣습니다. 새 실행에는 별도 기록을 남깁니다.")
    builtins = _ids(project)
    before, after = builtins
    if project.kind == "binary" and split == "validation":
        from .inspection_activation import active_model
        from .inspection_workspace import ACTIVE
        registered = list_models(_model_root())
        choices = builtins + registered
        bk, ak = k("baseline",project.id,split), k("candidate",project.id,split)
        active = active_model(ACTIVE, project)
        if active not in choices:
            st.error(f"로컬 활성 모델 {active}을 레지스트리에서 찾을 수 없습니다.")
            return
        if st.session_state.get(bk) not in choices:
            st.session_state[bk] = active
        if st.session_state.get(ak) not in choices:
            st.session_state[ak] = next(v for v in choices if v != active)
        left, right = st.columns(2)
        before = left.selectbox("기준 모델", choices, key=bk)
        after = right.selectbox("후보 모델", choices, key=ak)
        def swap():
            st.session_state[bk], st.session_state[ak] = st.session_state[ak], st.session_state[bk]
        st.button("기준·후보 바꾸기", on_click=swap, key=k("swap",project.id,split))
        st.button("모델 추가", on_click=section_jump, args=(project.id,split,"프로젝트 설정"), key=k("add-model",project.id,split))
        if before == after:
            st.warning("기준과 후보는 서로 다른 모델이어야 합니다.")
            return
        if before in registered or after in registered or [before, after] != builtins:
            st.warning("등록 후보는 개발용 검증만 가능합니다. 공식 시험·승인과 연결되지 않습니다.")
    rows = project.rows(split)
    n = len(rows)
    demo = project.demo_ids(split)
    full = project.is_full_split(split)
    # No duplicate options when every available image is already in a short registration.
    opts = (["full"] if full else []) + (["demo"] if demo and (not full or len(demo)<n) else [])
    if not opts:
        st.warning("실행 가능한 이미지가 없습니다. 데이터 복원 또는 등록을 먼저 진행하세요.")
        return
    scope = st.radio("실행할 범위", opts, index=opts.index("demo") if "demo" in opts else 0, horizontal=True,
                     format_func=lambda x:f"전체 평가 · 원본 {n:,}장" if x=="full" else f"시연 부분집합 · 원본 {len(demo):,}장",
                     key=k("run-scope",project.id,split))
    if not full:
        st.info("이 분할의 전체 원본은 현재 폴더에 없습니다. 동봉한 범위만 실행할 수 있습니다.")
    count = n if scope=="full" else len(demo)
    cells = project.conditions(split)
    st.caption(f"{SPLITS.get(split,split)} · {count:,}장 × {len(cells)}개 조건·seed = 예정 {count*len(cells):,}회 짝 비교 · "
               + f"{before} → {after}" + (" · 부분집합으로 채택을 결정하지 않습니다." if scope=="demo" else ""))
    result_key = k("live-result",project.id,split,scope,before,after)
    if st.button("새 추론 실행", type="primary", key=k("start",project.id,split,scope)):
        st.session_state.pop(result_key,None)
        bar = st.progress(0, text="입력과 모델을 확인하는 중…")
        last = [0.0]
        def progress(done,total):
            now = time.monotonic()
            if done==total or now-last[0] >= 0.22:
                bar.progress(min(1.0,done/max(1,total)),text=f"두 모델 추론 중 · 처리 {done:,} / {total:,}회")
                last[0] = now
        details = {"split": split, "before_model_id": before, "after_model_id": after, "scope": scope}
        path = None
        try:
            _event(project.id, "evaluation", "started", details)
            compare = project
            if [before, after] != builtins:
                adapters = dict(project.models)
                for model in (before, after):
                    if model not in adapters:
                        adapters[model] = load_registered(_model_root(), model)
                compare = ModelView(project, {before: adapters[before], after: adapters[after]}, development_only=True)
            path = Path(run_project(compare,split,root,sample_ids=(demo if scope=="demo" else None),progress=progress))
            m = load_run(path)
            _event(project.id, "evaluation", "completed",
                   {**details, "run_id": m["run_id"], "failed_pairs": m["failed_pairs"]})
            st.session_state[result_key] = m["run_id"]
            remember(project.id,split,m["run_id"])
            st.session_state[k("history",project.id,split)] = m["run_id"]
            bar.progress(1.0,text="추론 및 기록 저장 완료")
        except Exception as exc:
            bar.empty()
            try:
                _event(project.id, "evaluation", "failed", {**details, "error": str(exc)[:500]})
            except Exception as log_exc:
                st.error(f"이벤트 기록도 실패했습니다: {log_exc}")
            if path is not None:
                show_error(f"추론 기록 {path.name}은 저장되었을 수 있지만 완료 확인 또는 이벤트 기록에 실패했습니다.", exc)
            else:
                show_error("새 추론을 완료하지 못했습니다.",exc)
    rid = st.session_state.get(result_key)
    if rid:
        m = load_run(Path(root)/rid)
        message = f"새 실행 기록 생성 · 예정 {m['planned_pairs']:,}회 / 완료 {m['completed_pairs']:,}회 / 실패 {m['failed_pairs']:,}회"
        if m["failed_pairs"]:
            st.warning(message + " · 평가가 불완전합니다.")
        else:
            st.success(message)
        st.code(rid,language=None)
        def view():
            remember(project.id,split,rid)
            st.session_state[k("history",project.id,split)] = rid
            section_jump(project.id,split,"모델 비교")
        st.button("이번 실행 결과 열기",on_click=view,key=k("view-live",rid))
    st.caption("이미 저장한 기록은 ‘교체 판단’에서 엽니다. 그 화면을 여는 것만으로는 새 추론이 실행되지 않습니다.")


def _truth_display(project, meta, original, actual):
    from PIL import Image
    def png(arr):
        data = np.asarray(arr)
        if np.issubdtype(data.dtype,np.floating):
            if np.min(data)<0 or np.max(data)>1:
                raise ValueError("화면에 표시할 입력의 픽셀 범위가 올바르지 않습니다.")
            data = np.rint(data*255).astype("uint8")
        im = Image.fromarray(data)
        im.thumbnail((500,295))
        out = io.BytesIO()
        im.save(out,format="PNG")
        return out.getvalue()
    st.markdown(f"**실제 정답: {label(meta['true_label'])}**")
    st.caption(f"{meta['sample_id']}　·　{meta['source_id']}　·　{cell_label(meta['condition'],meta['seed'])}")
    left,right = st.columns(2)
    for col,arr,title in [(left,original,"원본 이미지"),(right,actual,"두 모델에 전달된 실제 시험 입력")]:
        col.image(png(arr),caption=f"{title} · {arr.shape[1]} × {arr.shape[0]}", width="content")
    preds = []
    for mid in [meta["before_model_id"],meta["after_model_id"]]:
        p = meta["predictions"][mid]
        predicted = p["predicted_label"]
        preds.append({"역할":"기존" if mid==meta["before_model_id"] else "후보","모델":mid,"예측":label(predicted),
                      "정답과 비교":"정답" if p["status"]=="ok" and predicted==meta["true_label"] else ("오류" if p["status"]=="ok" else "실패")})
    st.table(pd.DataFrame(preds),hide_index=True,border="horizontal")
    with st.expander("입력 해시와 모델 버전"):
        st.text(f"sample_id: {meta['sample_id']}\nrun_id: {meta['run_id']}\n입력 SHA-256: {meta['input_sha256']}")
        st.json({"models":meta["models"],"predictions":meta["predictions"]},expanded=False)
        st.caption("점수의 종류와 스케일은 모델마다 다를 수 있습니다.")


def _case_browser(project,split,path,m,case_root):
    from .inspection_cases import build_case, inspect_inputs, save_case
    st.subheader("실패 사례")
    rid = m["run_id"]
    pairs = table_file(path,"pairs",m)
    conds = [("ALL",-1)] + [(str(c),int(s)) for c,s in m["conditions"]]
    kinds = (["new_fn","fixed_fn","persistent_fn","stable_tp","new_fp","fixed_fp","persistent_fp","stable_tn","failed"]
             if project.kind=="binary" else ["new_error","fixed","both_wrong","both_right","failed"])
    if project.kind=="binary":
        from .inspection_reliability import classify_pairs
        pairs = classify_pairs(pairs)
    filters, main, inspector = st.columns([1.2, 2.5, 1.1], gap="medium")
    filters.markdown("#### 사례 목록")
    focus = st.session_state.pop(k("focus",rid),None)
    ck = k("case-condition",rid)
    if focus:
        fc,fs,metric,target = focus
        focus_cell = (str(fc),int(fs) if fs is not None else -1)
        # A criterion over several seeds should show all matching seeds, not just the first.
        if focus_cell not in conds and any(str(c)==str(fc) for c,s in conds):
            focus_cell = (str(fc),-1)
        st.session_state[ck] = focus_cell if focus_cell in conds or focus_cell[1]==-1 and any(c==focus_cell[0] for c,s in conds) else conds[0]
    extra = [(c,-1) for c in dict.fromkeys(v for v,s in conds[1:]) if sum(v==c for v,s in conds)>1]
    options = list(dict.fromkeys(conds[:1]+extra+conds[1:]))
    cond,seed = filters.selectbox("사례 입력 조건",options,format_func=lambda v:("모든 조건·seed" if v[0]=="ALL" else
                                      f"{cname(v[0])} · 모든 seed" if v[1]==-1 else cell_label(*v)),key=ck)
    subset = pairs if cond=="ALL" else pairs[(pairs.condition.astype(str)==cond) & ((pairs.seed==seed) if seed!=-1 else True)]
    tk = k("case-truth",rid,cond,seed)
    if focus:
        st.session_state[tk] = focus[3]
    truth = filters.selectbox("사례의 실제 정답", ["ALL",*project.classes],format_func=lambda v:"모든 정답" if v=="ALL" else label(v),key=tk)
    if truth!="ALL":
        subset = subset[subset.true_label==truth]
    with filters.expander("결함 면적·원본 밝기 필터"):
        st.caption("형태별 정답 라벨은 제공되지 않습니다. 면적은 원본 마스크, 밝기는 원본 RGB 픽셀로 계산합니다.")
        if project.id == "ksdd2" and {"defect_pixels", "width", "height"}.issubset(project.index.columns):
            size = st.selectbox("결함 면적 구간", ["전체", "Small", "Medium", "Large"],
                                key=k("failure-size", rid))
            st.caption("Small < 0.1% · Medium 0.1% 이상 1% 미만 · Large 1% 이상. 정상 원본에는 적용하지 않습니다.")
            if size != "전체":
                index = project.index.set_index("sample_id")
                try:
                    pixel = pd.to_numeric(index["defect_pixels"], errors="raise")
                    width = pd.to_numeric(index["width"], errors="raise")
                    height = pd.to_numeric(index["height"], errors="raise")
                    if ((pixel < 0) | (width <= 0) | (height <= 0) | (pixel > width * height)).any():
                        raise ValueError("유효하지 않은 주석 면적")
                    fraction = pixel / (width * height)
                    mask = ((pixel > 0) & (fraction < .001) if size == "Small" else
                            (fraction >= .001) & (fraction < .01) if size == "Medium" else fraction >= .01)
                    subset = subset[subset.sample_id.astype(str).isin(index.index[mask].astype(str))]
                except (ValueError, KeyError) as exc:
                    show_error("결함 면적을 확인하지 못했습니다.", exc)
                    return
        bright_key = "brightness:" + rid
        if st.button("원본 밝기 측정", key=k("case-brightness", rid)):
            from .inspection_brightness import original_brightness
            st.session_state.pop(bright_key, None)
            try:
                st.session_state[bright_key] = original_brightness(project, split, pairs)
            except Exception as exc:
                show_error("원본 밝기를 측정하지 못했습니다.", exc)
        if bright_key in st.session_state:
            bright = st.selectbox("원본 밝기 구간", ["전체", "낮음", "중간", "높음"],
                                  key=k("case-brightness-group", rid))
            st.caption("탐색용 경계: 낮음 < 0.17, 중간 0.17 이상 0.18 미만, 높음 0.18 이상. 물리 조도가 아닙니다.")
            measured = st.session_state[bright_key]
            if bright != "전체":
                mask = (measured.brightness < .17 if bright == "낮음" else
                        measured.brightness.ge(.17) & measured.brightness.lt(.18) if bright == "중간" else
                        measured.brightness.ge(.18))
                subset = subset[subset.sample_id.astype(str).isin(measured.loc[mask, "sample_id"].astype(str))]
    def matching(kind):
        if project.kind=="binary":
            return subset[subset.transition==kind]
        return subset[subset.status!="ok"] if kind=="failed" else subset[(subset.status=="ok") & (subset.change==kind)]
    frames = {q:matching(q) for q in kinds}
    kk = k("case-kind",rid,cond,seed,truth)
    if focus:
        metric = str(focus[2]).lower()
        priorities = (["new_fp","persistent_fp","new_fn"] if any(x in metric for x in ("fpr","fp","false_positive")) else
                      ["new_fn","persistent_fn","new_fp"]) if project.kind=="binary" else ["new_error","both_wrong"]
        st.session_state[kk] = next((q for q in priorities+kinds if len(frames[q])),kinds[0])
    first = next((i for i,q in enumerate(kinds) if len(frames[q])),0)
    kind = filters.selectbox("판정 변화 유형", kinds,index=0 if kk in st.session_state else first,format_func=lambda q:f"{KINDS[q]} · {len(frames[q]):,}회",key=kk)
    sample_rows = frames[kind]
    if sample_rows.empty:
        filters.info("이 조건과 변화 유형에 해당하는 사례는 0건입니다.")
        return
    choices = sample_rows.reset_index(drop=True)
    idx = filters.selectbox("확인할 실제 사례", range(len(choices)),key=k("case-sample",rid,cond,seed,truth,kind),
                       format_func=lambda i:f"{choices.at[i,'sample_id']} · {label(choices.at[i,'true_label'])} · "
                                            +cell_label(choices.at[i,"condition"],choices.at[i,"seed"]))
    r = choices.iloc[idx]
    token = k(rid,r.sample_id,r.condition,int(r.seed))
    if kind == "failed":
        main.warning("완료되지 않은 입력입니다. 정답이나 재현 성공으로 간주하지 않습니다.")
        main.dataframe(r.to_frame("값").astype(str),width="stretch")
        return
    try:
        data = build_case(project,path,str(r.sample_id),str(r.condition),int(r.seed))
        meta,original,actual = inspect_inputs(data)
    except Exception as exc:
        show_error("사례의 원본·시험 입력을 확인할 수 없습니다.",exc)
        return
    with main:
        _truth_display(project,meta,original,actual)
        if project.id == "ksdd2" and st.checkbox("원본 정답 마스크 표시", key=k("overlay", token)):
            from .inspection_workspace import annotation_overlay
            try:
                overlay = annotation_overlay(project, m, str(meta["sample_id"]), original)
                st.image(overlay, caption="원본 정답 마스크 · 빨강 (모델 예측 위치가 아님)", width="content", clamp=True)
            except Exception as exc:
                show_error("원본 정답 마스크를 표시하지 못했습니다.", exc)
    with inspector:
        st.markdown("#### 사례 Inspector")
        st.write(f"**사례:** {meta['sample_id']}")
        st.write(f"**변화:** {KINDS.get(kind, kind)}")
        st.write(f"**정답:** {label(meta['true_label'])}")
        st.write(f"**조건:** {cell_label(meta['condition'], meta['seed'])}")
        with st.expander("점수와 실행 정보"):
            st.code(meta["run_id"], language=None)
            st.json(meta["predictions"], expanded=False)
    if split!="validation":
        st.info("독립 시험 사례는 최종 결과 설명과 과거 실행 재현에만 사용합니다.")
    with st.container(horizontal=True):
        if st.button("사례 모음에 저장", type="primary", key=k("save-case",token)):
            try:
                rec = save_case(project,path,data,case_root)
                st.success("사례를 저장했습니다." if rec["created"] else "이미 저장한 같은 사례 기록입니다.")
            except Exception as exc:
                show_error("사례를 저장하지 못했습니다.",exc)
        st.download_button("선택한 사례 ZIP",data,f"{meta['project_id']}_{meta['case_id'][:15]}_{meta['history_id'][:10]}.zip",
                           "application/zip",key=k("case-zip",token),on_click="ignore")
    st.button("사례 모음 열기",on_click=section_jump,args=(project.id,split,"사례 모음·재시험"),key=k("case-library",token))


def case_name(row):
    return f"{row['sample_id']} · {label(row['true_label'])} · {cell_label(row['condition'],row['seed'])} · {KINDS.get(row['change'],row['change'])}"


def retest_table(results):
    change = {"unchanged_correct":"여전히 정답","fixed":"고친 오류","still_wrong":"여전히 오답","new_error":"새 오류","incomplete":"평가 불완전"}
    return pd.DataFrame([{"사례":r["sample_id"],"입력 조건":cell_label(r["condition"],r["seed"]),
                          "정답":label(r["true_label"]),"과거 기준 모델":r["reference_model_id"],
                          "당시 예측":label(r["reference_prediction"]), "재시험 모델":r["model_id"],
                          "새 예측":label(r["predicted_label"]) if r["status"]=="ok" else "추론 실패",
                          "기준 대비":change.get(r["change_from_baseline"],r["change_from_baseline"]),
                          "시험 시각 (UTC)":r["created_utc"]} for r in results])


def _history_replay(project,path,m,data,mode,trigger,case_root):
    from .inspection_cases import inspect_inputs, replay_case, save_case
    if data is None:
        return
    rid = m["run_id"]
    marker = hashlib.sha256(data).hexdigest()
    state = k("historical-result",project.id,rid,mode)
    old = st.session_state.get(state)
    if old and old[0]!=marker:
        del st.session_state[state]
        old = None
    if trigger or mode=="upload" and old is None:
        st.session_state.pop(state,None)
        try:
            with st.spinner("저장한 입력으로 당시 두 모델의 추론을 확인하고 있습니다…"):
                record = replay_case(_case_model_view(project,m),path,data)
            st.session_state[state] = (marker,record,None)
        except Exception as exc:
            st.session_state[state] = (marker,None,str(exc))
    old = st.session_state.get(state)
    if old is None:
        return
    _,r,err = old
    if err:
        show_error("사례 ZIP 재현 실패 · 선택한 실행과 입력·모델·기록이 일치하는지 확인하세요.",err)
        return
    st.success("당시 두 모델의 판정과 점수가 재현되었습니다.")
    meta,original,actual = inspect_inputs(data)
    _truth_display(project,r,original,actual)
    st.caption(f"점수 비교 · 절대 허용 오차 {r['atol']} / 상대 허용 오차 {r['rtol']}. 재학습 검사가 아닌 당시 모델의 추론 재현입니다.")
    with st.expander("재현 환경과 점수 오차"):
        st.json({"environment":r["environment"],"max_score_delta":r["max_score_delta"]})
    if mode=="upload" and st.button("재현한 사례를 모음에 저장",key=k("import",marker,rid)):
        try:
            saved = save_case(project,path,data,case_root)
            st.success("사례를 저장했습니다." if saved["created"] else "이미 저장한 같은 사례 기록입니다.")
        except Exception as exc:
            show_error("사례를 저장하지 못했습니다.",exc)


def _library(project,split,path,m,case_root):
    from .inspection_cases import list_cases, retest_collection
    from .inspection_model_registry import list_models
    rid = m["run_id"]
    st.subheader("사례 모음·재시험")
    mode = st.radio("확인할 작업", ["model","historical"],horizontal=True,format_func={"model":"모델 버전 재시험","historical":"과거 실행 재현"}.get,
                    key=k("case-op",rid))
    entries = [r for r in list_cases(case_root,project.id) if r["split"]==split and r["run_id"]==rid]
    count = len({e["case_id"] for e in entries})
    st.caption(f"선택한 실행의 저장 사례 {count:,}개 · 이력 {len(entries):,}건. 같은 입력과 같은 실행의 이력은 중복 저장하지 않습니다.")
    if mode=="historical":
        st.write("원래 모델 두 개로 같은 입력을 다시 추론해 당시 판정과 점수를 확인합니다.")
        source = st.radio("재현할 사례 출처",["사례 ZIP 업로드","저장한 사례"],horizontal=True,key=k("replay-source",rid))
        if source=="사례 ZIP 업로드":
            st.caption("위에서 선택한 실행에서 내보낸 사례 ZIP을 올리면 재현을 시작합니다. 다른 실행에서 만든 ZIP은 해당 실행을 먼저 선택하세요.")
            state_key = k("historical-result",project.id,rid,"upload")
            uploaded = st.file_uploader("재현할 사례 ZIP",type=["zip"],max_upload_size=65,key=k("replay-upload",rid),
                                        on_change=lambda:st.session_state.pop(state_key,None))
            _history_replay(project,path,m,uploaded.getvalue() if uploaded else None,"upload",False,case_root)
        elif entries:
            i = st.selectbox("재현할 저장 사례",range(len(entries)),format_func=lambda x:case_name(entries[x]),key=k("replay-case",rid))
            chosen = entries[i]
            clicked = st.button("당시 모델로 재현",key=k("replay-button",rid,chosen["history_id"]))
            _history_replay(project,path,m,Path(chosen["path"]).read_bytes(),"saved",clicked,case_root)
        else:
            st.info("이 실행에서 저장한 사례가 없습니다. 실패 사례 화면에서 이미지를 선택해 저장하세요.")
        return
    st.write("선택한 모델을 저장한 입력에 실행해 정답과 비교합니다. 판정이 바뀌어도 정답을 맞혔다면 개선으로 기록합니다.")
    if split!="validation":
        st.info("독립 시험 사례는 새 모델 재시험 대상에 포함하지 않습니다. 과거 실행 재현은 이용할 수 있습니다.")
        return
    if not entries:
        st.info("이 실행에서 저장한 사례가 없습니다. 실패 사례 화면에서 이미지를 선택해 저장하세요.")
        st.button("실패 사례에서 선택",on_click=section_jump,args=(project.id,split,"실패 사례"),key=k("go-cases",rid))
        return
    by_id = {row["case_id"]:row for row in entries}
    sel = st.multiselect("재시험할 저장 사례",list(by_id),default=[next(iter(by_id))],format_func=lambda v:case_name(by_id[v]),key=k("reg-cases",rid))
    mids = list(dict.fromkeys(_ids(project) + ([m["after_model_id"]] if project.kind=="binary" else [])
                              + (list_models(_model_root()) if project.kind=="binary" else [])))
    model = st.selectbox("재시험할 모델 버전",mids,index=mids.index(m["after_model_id"]),key=k("reg-model",rid))
    st.caption(f"당시 기준 모델 {m['before_model_id']} → 선택한 모델 {model} · 개발용 검증 사례 · 전체 분할 평가를 대신하지 않습니다.")
    selections = sorted(sel)
    histories = {by_id[s]["history_id"] for s in sel}
    folder = Path(case_root)/"retests"
    if st.button("선택한 사례 재시험",type="primary",disabled=not sel,key=k("reg-exec",rid,model,*selections)):
        details = {"run_id": rid, "model_id": model, "case_count": len(sel)}
        try:
            _event(project.id, "case_retest", "started", details)
            with st.spinner("선택한 모델로 재시험하고 결과를 기록하고 있습니다…"):
                results = retest_collection(_case_model_view(project,m,model),path,[by_id[s]["path"] for s in sel],model,folder)
            nok = sum(r["status"]=="ok" for r in results)
            _event(project.id, "case_retest", "completed", {**details, "failed": len(results)-nok})
            if nok==len(results):
                st.success(f"새 재시험 기록 저장 · 완료 {nok}건 / 실패 0건")
            else:
                st.warning(f"새 재시험 기록 저장 · 완료 {nok}건 / 실패 {len(results)-nok}건")
        except Exception as exc:
            try:
                _event(project.id, "case_retest", "failed", {**details, "error": str(exc)[:500]})
            except Exception as log_exc:
                st.error(f"이벤트 기록도 실패했습니다: {log_exc}")
            show_error("재시험을 완료하지 못했습니다.",exc)
    history = []
    for file in folder.glob("*.json"):
        r = jread(file)
        if r.get("run_id")==rid and r.get("project_id")==project.id and r.get("model_id")==model and r.get("history_id") in histories:
            history.append(r)
    if history:
        history.sort(key=lambda r:r["created_utc"], reverse=True)
        st.markdown("**선택한 사례·모델의 재시험 기록**")
        st.dataframe(retest_table(history),hide_index=True,width="stretch")
        csv = pd.DataFrame(history).to_csv(index=False).encode("utf-8-sig")
        st.download_button("재시험 기록 CSV",csv,f"{rid}_{model}_retests.csv","text/csv",
                           key=k("retest-csv",rid,model,*selections),on_click="ignore")


def _settings(project,split):
    from .inspection_projects import register_zip
    st.subheader("프로젝트 설정")
    binary = project.kind=="binary"
    if binary:
        st.write("검사 지점에서 불량을 놓치면 뒤에서 찾아 다시 확인해야 합니다. 모델을 교체할 때 어떤 불량을 새로 놓치는지, 정상에 대한 오경보도 함께 확인합니다.")
    else:
        st.write("여섯 결함 종류를 구분하는 모델을 바꿀 때, 잡음에서 고친 오분류와 새로 생긴 오분류를 함께 확인합니다.")
    models = _ids(project)
    rows = project.rows(split)
    conditions = project.conditions(split)
    field_table([
        ("검사 대상",project.title),
        ("정답 출처", project.source + (" · 공식 마스크의 결함 픽셀 유무" if binary and project.official else "")),
        ("정답 클래스"," · ".join(f"{label(c)} ({c})" for c in project.classes)),
        ("모델 교체", f"기존 {models[0]} → 후보 {models[1]}"),
        ("평가 분할", f"{SPLITS.get(split,split)} · 원본 {len(rows):,}장 · " + " / ".join(f"{label(c)} {sum(rows.true_label==c):,}장" for c in project.classes)),
        ("시험 입력", " / ".join(cell_label(c,s) for c,s in conditions)),
    ])
    if binary:
        st.write("**시제품 기준** · 각 조건에서 불량 재현율 80% 이상, 정상 오경보율 10% 이하입니다. 원본에서 각 지표의 악화는 2%p, 교란 조건에서는 5%p까지 허용합니다. 전체 조건에서 한 오류가 줄 때 다른 오류는 늘지 않아야 합니다.")
        if project.official:
            st.caption("검증 원본에서 정상 오경보 한도 안의 임계값을 선택해 모든 조건에 고정합니다. 전체 검증에서 미달하면 공식 시험 분할을 열지 않습니다.")
    else:
        st.write("**시제품 기준** · 동결된 NEU 규약에 따라 원본·잡음 조건의 macro F1과 종류별 재현율 변화를 확인합니다.")
    if not project.official:
        st.info("사용자 등록 자료는 개발용 검증입니다. 이 자료의 결과로 공식 모델의 교체 판정을 바꾸지 않습니다.")
    with st.expander("모델 식별 정보와 전체 시험 규약"):
        st.json({"models":{name:project.models[name].identity for name in models},"protocol":project.protocol},expanded=False)
    if binary:
        from .inspection_model_registry import list_models, register_model
        st.divider()
        st.markdown("### 로컬 모델 등록")
        st.warning("joblib 파일을 읽으면 임의 코드가 실행될 수 있습니다. 출처를 신뢰하는 로컬 파일만 등록하세요. SHA-256은 변경 감지용이며 안전성을 증명하지 않습니다.")
        st.caption("현재 K0/K1과 같은 고정 ResNet18 전처리·1536차원 특징·이진 StandardScaler/LogisticRegression 모델만 지원합니다. 등록 후보는 검증 분할에서만 사용합니다.")
        registered = list_models(_model_root())
        st.write("등록된 모델: " + (", ".join(registered) if registered else "없음"))
        with st.form(k("model-form",project.id)):
            model_id = st.text_input("새 모델 ID (예: K2)", max_chars=40)
            model_path = st.text_input("신뢰한 로컬 .joblib 절대 경로")
            trusted = st.checkbox("이 파일을 신뢰하며 로드 시 코드가 실행될 수 있음을 확인했습니다.")
            submit = st.form_submit_button("모델 등록")
        if submit:
            details = {"model_id": model_id}
            try:
                _event(project.id, "model_registration", "started", details)
                metadata = register_model(_model_root(), model_path, model_id, trusted=trusted,
                                          auth_token=st.session_state.get("security-token"))
                _event(project.id, "model_registration", "completed",
                       {**details, "artifact_sha256": metadata["artifact_sha256"]})
            except Exception as exc:
                try:
                    _event(project.id, "model_registration", "failed", {**details, "error": str(exc)[:500]})
                except Exception as log_exc:
                    st.error(f"이벤트 기록도 실패했습니다: {log_exc}")
                show_error("모델 등록 또는 이벤트 기록에 실패했습니다. 이미 등록되었을 수 있으니 목록을 확인하세요.", exc)
            else:
                st.success(f"{model_id} 등록 완료 · 개발용 검증에서 선택할 수 있습니다.")
                st.json(metadata, expanded=False)
    st.divider()
    st.markdown("### 새 검사 데이터 등록")
    st.write("정답 CSV와 이미지를 한 ZIP으로 올리고, 클래스 구성이 맞는 로컬 모델 두 개를 선택합니다.")
    st.caption("ZIP 루트의 manifest.csv: sample_id, image_path, true_label 필수 · group_id, source_id 선택. "
               "image_path는 ZIP 안의 상대 경로입니다. PNG·JPEG·BMP, 최대 500장 / ZIP 100 MiB.")
    template = st.selectbox("등록할 모델 조합",["neu_cls6"],format_func=lambda x:
                            ("K0 → K1 · 정상 / 불량" if x=="ksdd2" else "V0 → G1 · NEU 결함 6종"),key=k("register-adapter",project.id))
    labels = ["normal","defect"] if template=="ksdd2" else list(LABELS)[2:]
    st.caption("허용된 true_label: " + ", ".join(labels) + (" · NEU 입력은 200×200 회색조입니다." if template=="neu_cls6" else " · RGB 또는 회색조, 이미지당 최대 2,795,520픽셀 / 한 변 최대 4,096픽셀."))
    title = st.text_input("새 프로젝트 이름",max_chars=80,key=k("register-title",project.id))
    uploaded = st.file_uploader("이미지와 정답 CSV가 든 ZIP",type=["zip"],max_upload_size=100,key=k("registration",project.id))
    if st.button("새 프로젝트 등록",disabled=not title.strip() or not uploaded,key=k("register",project.id),type="primary"):
        try:
            _event(project.id, "dataset_registration", "started", {"template_id": template})
            pid = register_zip(uploaded.getvalue(),title.strip(),template)
            _event(project.id, "dataset_registration", "completed", {"registered_project_id": pid, "template_id": template})
        except Exception as exc:
            try:
                _event(project.id, "dataset_registration", "failed", {"template_id": template, "error": str(exc)[:500]})
            except Exception as log_exc:
                st.error(f"이벤트 기록도 실패했습니다: {log_exc}")
            show_error("데이터 등록 또는 이벤트 기록에 실패했습니다. 프로젝트가 이미 생성되었을 수 있으니 목록을 확인하세요.",exc)
        else:
            st.session_state["inspection_next_project"] = pid
            st.session_state[k("section",pid,"validation")] = "새 검증 실행"
            st.rerun()


def workbench():
    from .inspection_projects import CASES, RUNS, load_project
    from .inspection_workspace import ACTIVE, active_model, decision_workspace, introduction, overview, registry, reports
    projects = _steel_projects()
    if not projects:
        st.warning("등록된 검사 프로젝트가 없습니다.")
        return
    if "inspection_next_project" in st.session_state:
        pid = st.session_state.pop("inspection_next_project")
        if pid in projects:
            st.session_state["inspection_project"] = pid
    if st.session_state.get("inspection_project") not in projects:
        st.session_state.pop("inspection_project", None)
    left,right = st.columns([2,1])
    pid = left.selectbox("검사 프로젝트", list(projects),format_func=projects.get,key="inspection_project")
    project = load_project(pid)
    splits = project.available_splits()
    if not splits:
        st.warning("지금 평가할 수 있는 분할이 없습니다.")
        return
    split = right.selectbox("평가 분할",splits,format_func=lambda s:SPLITS.get(s,s),key=k("split",pid))
    section_key = k("section",pid,split)
    section = st.session_state.get(section_key, "프로젝트 소개")
    if section not in SECTIONS:
        section = "개요"
    with st.sidebar:
        for group, items in NAVIGATION.items():
            st.html(f'<div class="workspace-nav-label">{escape(group)}</div>')
            for name in items:
                st.button(name, type="primary" if name == section else "secondary",
                          width="stretch", on_click=section_jump, args=(pid,split,name),
                          key=k("navigation",pid,split,name))
    st.html('<span class="steel-scope" style="margin-bottom:0">SteelTrust　 /　 OPERATIONAL AI VALIDATION</span>')
    if section != "프로젝트 소개":
        heading(section, "검사 AI 모델, 데이터와 검증 근거를 연결하는 작업 공간")
    try:
        active = active_model(ACTIVE, project)
        st.caption(f"프로젝트 · {project.title}　 |　 데이터 · {SPLITS.get(split,split)}　 |　 로컬 활성 모델 · {active}　 |　 저장 기록 / 새 추론 구분")
        activated = st.session_state.pop(k("activation-message", pid), None)
        if activated:
            st.success(f"로컬 활성 모델을 {activated}로 변경했습니다. 생산 설비는 변경되지 않았습니다.")
    except Exception as exc:
        show_error("로컬 활성 모델을 확인하지 못했습니다.", exc)
        return
    if not project.official:
        st.caption("사용자 등록 개발용 검증 · 공식 프로젝트의 교체 판정과 별도로 기록합니다.")
    if section=="프로젝트 설정":
        _settings(project,split)
        return
    if section=="이벤트 기록":
        from .inspection_event_log import read_events
        from .surface_data import ROOT
        st.subheader("로컬 이벤트 기록")
        st.warning("OS 계정은 인증된 작성자 신원이 아닙니다. 로컬 파일은 수정·삭제할 수 있으므로 공식 감사 로그나 접근 통제가 아닙니다. UI에서 수행한 작업만 기록합니다.")
        try:
            events = read_events(ROOT/"inspection_events", project_id=pid)
        except Exception as exc:
            show_error("이벤트 기록을 읽지 못했습니다.", exc)
        else:
            if events:
                st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
                st.download_button("이벤트 JSON", json.dumps(events, ensure_ascii=False, indent=2).encode("utf-8"),
                                   f"{pid}_events.json", "application/json", on_click="ignore")
            else:
                st.info("이 프로젝트의 기록된 이벤트가 없습니다.")
        return
    if section=="평가 설계 기록":
        from .inspection_reliability_ui import governance
        from .surface_data import ROOT
        governance(project, split, ROOT/"inspection_governance"/project.id)
        return
    if section=="새 검증 실행":
        _run(project,split,RUNS)
        return
    records = runs_for(RUNS,pid,split)
    if section=="모델 레지스트리":
        registry(project,split,records)
        return
    path, m = pick_run(project,split,records,RUNS,interactive=True)
    if section=="프로젝트 소개":
        introduction(project,split,path,m)
        return
    if section=="개요":
        overview(project,split,path,m)
        return
    if section=="교체 판단":
        decision_workspace(project,split,path,m)
        return
    if section=="보고서":
        reports(project,split,path,m)
        return
    if m is None:
        st.info("이 분할에 저장된 실행 기록이 없습니다. 새 추론으로 첫 검증 결과를 만드세요.")
        st.button("새 검증 실행으로 이동",type="primary",on_click=section_jump,args=(pid,split,"새 검증 실행"),key=k("go-run",pid,split))
        return
    if section=="전이·통계 분석":
        if project.kind != "binary":
            _multiclass_analysis(project, split, path, m)
        else:
            from .inspection_reliability_ui import reliability
            reliability(table_file(path,"pairs",m), m, project)
        return
    if section=="모델 비교":
        buttons = st.columns(3)
        buttons[0].button("새 비교 구성", on_click=section_jump, args=(pid,split,"새 검증 실행"), key=k("compare-run",m["run_id"]))
        buttons[1].button("실패 사례 열기", on_click=section_jump, args=(pid,split,"실패 사례"), key=k("compare-cases",m["run_id"]))
        buttons[2].button("교체 조건 확인", on_click=section_jump, args=(pid,split,"교체 판단"), key=k("compare-decision",m["run_id"]))
        _result(project,split,path,m)
    elif section=="실패 사례":
        _case_browser(project,split,path,m,CASES)
    else:
        _library(project,split,path,m,CASES)
