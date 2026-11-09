"""Adapters for the existing record-comparison view, using the held-out fold only."""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from .detector import RUN, model_bundle, predict_record
from .hi_study import make_wave

LABELS = {"selected":"내부 검증 선택 절차", "A1":"A1 · 기존 특징 RF", "H":"H · 기존 특징 SVC",
          "I":"I · 강건 특징 SVC", "R":"R · 강건 특징 RF", "J":"J · 세부 대역 SVC", "K":"K · 최근 2초 SVC"}


@st.cache_data
def results():
    p = pd.read_csv(RUN/"predictions.csv", float_precision="round_trip")
    stats = pd.read_csv(RUN/"summary.csv")
    verdict = json.loads((RUN/"verdict.json").read_text())
    return p[p.budget==.15], stats[stats.budget==.15], verdict


@st.cache_resource
def bundle(fold):
    return joblib.load(RUN/f"models/fold_{fold}.joblib")


def jump(file, idx):
    st.session_state.update(hi_file=file, hi_index=int(idx))


def case_buttons(rows):
    slag = rows.y_true.eq(1)
    new = rows.prediction.eq(1)
    old = rows.A1.eq(1)
    masks = {"회복한 미검출":slag & new & ~old, "새로 놓친 S":slag & ~new & old,
             "새 오경보":~slag & new & ~old}
    for col, (label, mask) in zip(st.columns(3), masks.items()):
        examples = rows.loc[mask].sort_values(["group_id","file_id","window_idx"])
        target = () if examples.empty else (examples.iloc[0].file_id, examples.iloc[0].window_idx)
        col.button(f"{label} · {len(examples)}건", disabled=not len(target),
                   on_click=jump, args=target, key="det_jump_"+label)


def replay(record, idx, condition, subset, model, outcome):
    ref = subset.set_index("model_id")
    row = ref.loc[model]
    b = bundle(int(row.fold_id))
    if row.group_id in b["train_groups"] or row.group_id not in b["test_groups"]:
        raise ValueError("Selected group is not held out")
    history = []
    source = row.copy()
    for k in range(idx+1):
        source["window_idx"] = k
        history.append(make_wave(record[k], b["scale"], source, condition))
    display = []
    for name in dict.fromkeys(["A1","I",model]):
        saved = ref.loc[name]
        config = b["winners"][.15] if name == "selected" else name
        if saved.configuration != config:
            raise ValueError("저장된 구성과 학습 그룹의 선택이 다릅니다.")
        candidate = model_bundle(b, config, .15)
        result = list(predict_record(history, candidate))[-1]
        s, p = result["score"], result["prediction"]
        if p != saved.prediction or not np.isclose(s, saved.score, atol=1e-12, rtol=0):
            raise ValueError("저장된 평가 결과와 재추론 불일치")
        caption = LABELS[name] if name!="selected" else f"내부 선택 · {config}"
        display.append({"방식":caption, "원래 판정":"슬래그로 분류" if p else "비슬래그로 분류",
                        "보류 여부":"보류 없음", "점수 종류":"RF 슬래그 점수" if config in ("A1","R") else "SVC 결정 점수",
                        "모델 점수":f"{s:.4f}", "임계값":f"{candidate['threshold']:.4f}",
                        "정답과 비교":outcome(int(saved.y_true),p)})
    return display


def totals(stats, model, scenario, verdict):
    clean_ok = verdict["targets_pass"]["clean"]
    st.info("원본·교란 목표 통과 · 기본 시연 후보" if verdict["adopt"] else
            ("원본 조건 개선 후보 · 교란 목표 미달" if clean_ok else "핵심 검출 목표 미달 · 기존 기본 검출기 A0 유지"))
    st.caption("15%는 내부 원본의 오경보 예산입니다. 선택 절차 행은 각 평가 fold의 학습 그룹에서 고른 구성을 사용합니다.")
    current = stats[stats.scenario==scenario].set_index("model_id")
    cols = {"tp":"S 검출","fn":"S 미검출","fp":"오경보","tn":"비슬래그 정답","fp_E":"E 오경보","fp_B":"B 오경보",
            "recall":"S 검출률","fpr":"오경보율"}
    table = current.loc[list(LABELS),list(cols)].copy()
    for k in ["recall","fpr"]:
        table[k] = table[k].map("{:.1%}".format)
    st.dataframe(table.rename(index=LABELS,columns=cols),width="stretch")
    c = current.loc[model]
    st.write(f"A1과 비교해 이전 미검출 {int(c.rescued_vs_A1)}개를 검출하고, 이전 검출 {int(c.lost_vs_A1)}개를 놓쳤습니다. "
             f"새 오경보 {int(c.added_fp_vs_A1)}개, 사라진 오경보 {int(c.removed_fp_vs_A1)}개입니다.")
    st.caption("합성 교란에서도 동일 fold의 원본 임계값을 사용합니다. K는 현재 포함 같은 파일에서 관측한 최대 2초의 특징을 사용합니다.")
    a,b = st.columns(2)
    a.download_button("전체 비교 CSV · 10%와 15%", (RUN/"summary.csv").read_bytes(),
                      "detector_summary.csv", "text/csv", key="det_summary")
    b.download_button("구간별 예측 CSV", (RUN/"predictions.csv").read_bytes(),
                      "detector_predictions.csv", "text/csv", key="det_predictions")
