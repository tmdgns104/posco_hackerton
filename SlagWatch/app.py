import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data import HZ, SIZE, recording_paths, read_recording
from src.faults import seed_for
from src.inference import classify_window


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
LABELS = dict(clean="원본", noise="잡음", spike="스파이크", missing="값 누락", freeze="값 고정")
STATUS = dict(SLAG="유동 감지", NON_SLAG="유동 미감지", INPUT_CHECK="입력 확인 필요")
REASONS = dict(NONFINITE="누락 또는 비유한 값", CONST_AXIS="한 축의 값이 구간 전체에서 고정됨", BAD_SHAPE="구간 크기 오류")
BINARY = {1:"유동 감지",0:"유동 미감지"}
MODELS = {"spectral":"진폭 + 주파수 RF","rf":"진폭 RF","baseline":"단일 RMS"}
def model_cell(value, model):
    pred = value[model + "_pred"]
    if pred is None:
        return "보류"
    if model == "baseline":
        return BINARY[pred]
    return f"{BINARY[pred]} ({value[model + '_score']:.3f})"



@st.cache_resource
def resources():
    fold_map = json.loads((ROOT/"artifacts/fold_map.json").read_text())
    models = {f:joblib.load(ROOT/f"artifacts/fold_{f}.joblib") for f in set(fold_map.values())}
    return fold_map, models


@st.cache_data
def results():
    return (pd.read_csv(REPORTS/"predictions.csv"), json.loads((REPORTS/"metrics.json").read_text(encoding="utf-8")),
            recording_paths(ROOT/"data/raw"))


@st.cache_data
def recording(path):
    return read_recording(path)


def waveform(original, changed, start_s, scenario):
    x = np.arange(SIZE)/HZ + start_s
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for axis, letter in enumerate("xyz"):
        for label, color, values in [("원본", "#166b88", original), ("시험 입력", "#d24f27", changed)]:
            if scenario == "clean" and label == "시험 입력":
                continue
            fig.add_trace(go.Scatter(x=x, y=values[:,axis], name=label, mode="lines",
                           line=dict(color=color,width=1), opacity=0.82, legendgroup=label,
                           showlegend=axis==0, connectgaps=False),row=axis+1,col=1)
        fig.update_yaxes(title_text=letter.upper(), row=axis+1,col=1)
    if scenario == "missing" and np.isnan(changed).any():
        fig.add_vrect(x0=start_s+0.2,x1=start_s+0.3, fillcolor="#d24f27",opacity=0.13,line_width=0,row="all",col=1)
    fig.update_xaxes(title_text="기록 시작 후 시간 (초)",row=3,col=1)
    fig.update_layout(height=410,margin=dict(l=15,r=10,t=10,b=12),
                      legend=dict(orientation="h",y=1.13),hovermode="x unified")
    return fig


st.set_page_config(page_title="SteelTrust · 철강 AI 검증",layout="wide")
section=st.sidebar.radio("화면 선택",["철강 표면 결함","Adaptive Red Team","슬래그 G3 개선 실험","슬래그 G5·G6 모델 검증","실패 재검증","슬래그 유동 판정","안전 지원 검증","검증 도구"],key="hi_workspace")
if section=="철강 표면 결함":
    from src.surface_ui import workbench
    try:
        workbench()
    except (OSError,KeyError,ValueError,RuntimeError) as err:
        st.error(f"영상 사례 재현 중단 · {err}")
    st.stop()
if section=="Adaptive Red Team":
    from src.adaptive_ui import workbench as adaptive_workbench
    adaptive_workbench()
    st.stop()
if section=="슬래그 G5·G6 모델 검증":
    from src.slag_compare_ui import workbench
    workbench()
    st.stop()
if section=="슬래그 G3 개선 실험":
    from src.slag_g3_ui import workbench as g3_workbench
    g3_workbench()
    st.stop()
if section=="실패 재검증":
    from src.pair_ui import workbench
    try:
        workbench()
    except (OSError, KeyError, ValueError, RuntimeError) as err:
        st.error(f"재현 중단 · {err}")
    st.stop()
if section=="안전 지원 검증":
    from src.safety_ui import safety_home
    safety_home()
    st.stop()
if section=="슬래그 유동 판정":
    from src.hi_ui import home
    home()
    with st.expander("5초 파일 전체 평가 · 48파일"):
        st.caption("0.5초 창 10개가 끝난 뒤 파일당 한 번 판정한 개발 평가입니다. S 16개, E/B 32개입니다.")
        t=pd.read_csv(ROOT/"experiments/temporal5s/summary.csv")
        st.table(t[["model_id","tp","fn","fp","tn","fp_E","fp_B"]].set_index("model_id"))
        st.caption("T2 원본 검출 9개로 최소 12개에 미달해 5초 교란 평가는 중단했습니다.")
    st.stop()
st.title("판정 검증 도구")
st.caption("슬래그 유동 검증  ·  SSFD v3 공개 주조 진동  ·  실패 발견 · 모델 수정 · 동일 조건 회귀 검증")
try:
    pred, stats, paths = results()
    fold_map, models = resources()
except (OSError, ValueError, KeyError) as err:
    st.error(f"데이터 또는 학습 결과를 확인하세요. {err}")
    st.code("python -m src.train_eval")
    st.stop()

improved, review, discovery, policy_tab, evaluation = st.tabs(["개선 전후 비교","구간 검사", "자동 실패 탐색", "신뢰성 게이트", "모델 전체 평가"],default="개선 전후 비교")
def jump_to(file, window, scenario):
    st.session_state.update(record_id=file, window_id=window, scenario_id=scenario)


with review:
    with st.container(horizontal=True):
        st.button("미검출 개선 사례", on_click=jump_to, args=("S-1.csv",9,"clean"))
        st.button("스파이크 오경보 사례", on_click=jump_to, args=("B-12.csv",0,"spike"))
        st.button("입력 누락 사례", on_click=jump_to, args=("B-12.csv",0,"missing"))
    a,b,c=st.columns([1,1,1.4])
    file_id=a.selectbox("측정 기록", sorted(paths, key=lambda n:("SEB".index(n[0]), int(n[2:-4]))),key="record_id")
    scenario=b.selectbox("시험 조건", list(LABELS),format_func=LABELS.get,key="scenario_id")
    idx=c.slider("0.5초 구간 번호", 0, 9, 0,key="window_id")
    phase, number = file_id[:-4].split("-")
    fold = fold_map[number]
    group = int(number)
    w = recording(paths[file_id])[idx]
    seed = seed_for(group,phase,idx,scenario)
    changed, observed = classify_window(w,models[fold],scenario,idx,seed)
    _, reference = classify_window(w,models[fold],"clean",idx,seed_for(group,phase,idx,"clean"))
    st.write(f"**{file_id}**　{idx/2:.1f} – {(idx+1)/2:.1f}초　·　기록 정답 **{BINARY[int(phase=='S')]}**　·　평가 그룹 {group}")
    left,right = st.columns([2.2,1])
    with left:
        st.plotly_chart(waveform(w,changed,idx/2,scenario),width="stretch",config={"displaylogo":False})
    with right:
        st.subheader("입력 확인 필요" if observed["reason_codes"] else "모델별 판정")
        descriptions = {"noise":"세 축에 가우시안 잡음을 추가했습니다.", "spike":"X축 16개 샘플에 순간 충격을 추가했습니다.",
                        "missing":"짝수 구간의 중앙 0.1초를 누락 처리합니다.", "freeze":"짝수 구간의 X축을 첫 값으로 고정합니다."}
        if scenario in descriptions:
            st.caption(descriptions[scenario])
        if observed["reason_codes"]:
            st.warning(REASONS[observed["reason_codes"]])
        elif scenario in ("missing","freeze") and idx%2:
            st.info("이 시험은 짝수 구간에만 이상을 주입합니다. 현재 구간은 원본입니다.")
        st.table(pd.DataFrame([
            {"모델": label, "원본": model_cell(reference, model), "시험 입력": model_cell(observed, model)}
            for model, label in MODELS.items()
        ]).set_index("모델"))
        st.caption(f"그룹 {group}은 세 모델의 학습에서 제외했습니다. 괄호 안은 RF 유동 점수이며 0.5 이상을 감지로 표시합니다.")
    current=pred[(pred.file_id==file_id)&(pred.scenario==scenario)].sort_values("window_idx")
    with st.expander("이 기록의 10개 구간"):
        st.dataframe(current[["window_idx","start_s","end_s","y_true","baseline_pred","rf_pred","spectral_pred","reason_codes"]],hide_index=True)
    st.download_button("선택 기록 CSV",current.to_csv(index=False).encode("utf-8-sig"),
                       f"{file_id[:-4]}_{scenario}.csv","text/csv")

with evaluation:
    st.subheader("학습에서 제외한 16개 그룹의 평가")
    st.caption("480개 구간을 조건별로 평가했습니다. 검출률은 전체 S 160개, 오경보율은 전체 E/B 320개를 분모로 사용합니다. 보류는 별도로 셉니다.")
    m=pd.DataFrame(stats["metrics"])
    overview=m.assign(조건=m.scenario.map(LABELS),모델=m.model.map(MODELS),
          검출률=(m.detection_rate*100).round(1),오경보율=(m.false_alarm_rate*100).round(1)).rename(
          columns={"fp":"오경보","fn":"미검출","held":"보류","eligible":"판정","macro_f1":"Macro F1"})
    st.dataframe(overview[["조건","모델","검출률","오경보율","오경보","미검출","보류","판정","Macro F1"]].rename(
        columns={"검출률":"검출률 (%)","오경보율":"오경보율 (%)"}),hide_index=True,width="stretch")
    with st.expander("36개 입력 특징과 평가 방법"):
        st.write("진폭 RF는 X·Y·Z축의 RMS, 표준편차, 범위, 절댓값 최댓값 12개를 사용합니다. "
                 "진폭 + 주파수 RF는 각 축의 8개 주파수 대역별 에너지 비율 24개를 더 사용합니다. "
                 "모델 설정·판정 기준·평가 그룹은 동일합니다. 같은 16개 그룹에서 개발 방법을 비교한 결과입니다.")
    st.subheader("오판 구간 찾기")
    model=st.radio("비교 모델",list(MODELS),format_func=MODELS.get,horizontal=True)
    wrong=pred[pred[model+"_pred"].notna() & pred[model+"_pred"].ne(pred.y_true)]
    st.dataframe(wrong[["file_id","window_idx","scenario","y_true",model+"_pred"]],hide_index=True,height=230,width="stretch")
    st.caption("구간 검사에서 같은 기록·구간·시험 조건을 선택하면 해당 오판을 재현할 수 있습니다.")
    c1,c2=st.columns(2)
    c1.download_button("전체 예측 CSV",(REPORTS/"predictions.csv").read_bytes(),"predictions.csv","text/csv")
    c2.download_button("평가 결과 JSON",(REPORTS/"metrics.json").read_bytes(),"metrics.json","application/json")
st.caption("자료: Sehri 외, Steel Slag Flow Dataset v3, CC BY 4.0 · DOI 10.17632/mjnpsbs6fs.3. 잡음·스파이크·누락·고정은 합성 시험입니다.")

from src.steel_ui import search_tab, gate_tab
with discovery:
    search_tab()
with policy_tab:
    gate_tab()

from src.comparison_ui import comparison
with improved:
    comparison()
