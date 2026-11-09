"""Recorded SSFD inspection; primary predictions never use a gate or the label."""
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .data import recording_paths,read_recording,HZ
from .hi_models import predict_signal
from .hi_study import DEST,ROOT,read_config,make_wave,summarize
from .score_audit import AUDIT
from . import detector_comparison as current

LABEL={"A0":"A0 · 기존 RF / 0.5","A1":"A1 · 기존 RF / 내부 조정","H":"H · 기존 특징 + SVC","I":"I · 강건 특징 + SVC"}
SIGNALS={"clean":"실측 원본","noise":"합성 잡음","spike":"합성 X축 스파이크","new_y_spike":"합성 Y축 스파이크","new_z_spike":"합성 Z축 스파이크"}


@st.cache_resource
def inputs():
    return (read_config(),pd.read_csv(DEST/"predictions.csv",float_precision="round_trip"),recording_paths(ROOT/"data/raw"),
            json.loads((DEST/"adoption.json").read_text()))


@st.cache_resource
def fold_bundle(fold):
    return joblib.load(DEST/f"models/fold_{fold}.joblib")


@st.cache_data
def recording(path):
    return read_recording(path)


def outcome(truth,p):
    if p is None:return "입력 확인"
    return ("검출" if p else "미검출") if truth else ("오경보" if p else "비슬래그 정상 판단")


def home():
    protocol,results,paths,decision=inputs()
    st.title("슬래그 유동 구간 판정")
    st.caption("SteelTrust · 공정 AI 개발·검증 담당자를 위한 안전 지원 도구. 주변 진동에 따른 오판과 필요한 슬래그 경보의 손실을 재현·검증합니다.")
    st.write("**기본 경보 처리 · 보류 없음**　유효한 입력은 원래 판정을 그대로 표시합니다. 누락·비유한 값·고정 축은 판정 전에 검사합니다.")
    st.write("철강 주조 진동에서 슬래그를 놓치는 구간을 줄이면서, 비슬래그 구간의 불필요한 경보가 늘어나는지 확인합니다.")
    st.caption("SSFD 공개 주조 기록 재생 · 0.5초 단위 · 출강 사례에서 출발했으며 잡음·스파이크는 문헌의 진동 간섭 문제를 참고한 합성 교란입니다. 출강 종료 시각이나 실제 유출량은 이 데이터로 평가할 수 없습니다.")
    left,middle,right=st.columns(3)
    names=sorted(paths,key=lambda n:("SEB".index(n[0]),int(n[2:-4])))
    file_id=left.selectbox("측정 기록",names,key="hi_file")
    condition=middle.selectbox("시험 입력",list(SIGNALS),format_func=SIGNALS.get,key="hi_condition")
    choices = {"new:"+key:label+" · 15%" for key,label in current.LABELS.items()}
    choices.update({key:LABEL[key]+" · 이전 10%" for key in ("I","H")})
    model=right.selectbox("비교할 실험 후보",list(choices),format_func=choices.get,key="hi_model")
    active = model.startswith("new:")
    if active:
        model = model[4:]
        fresh, stats, status = current.results()
        fresh = fresh[fresh.scenario==condition]
        current.case_buttons(fresh[fresh.model_id==model])
    idx=st.slider("구간 번호 (한 기록에 10구간)",0,9,0,key="hi_index")
    subset=results[results.scenario==condition]
    saved=subset[(subset.file_id==file_id)&(subset.window_idx==idx)].set_index("model_id")
    row=saved.loc["A0" if active else model]
    bundle=fold_bundle(int(row.fold_id))
    if row.group_id in bundle['train_groups'] or row.group_id not in bundle['test_groups']:
        st.error("평가 그룹과 모델의 분할이 일치하지 않습니다.");st.stop()
    raw=recording(paths[file_id])[idx]
    wave=make_wave(raw,bundle["scale"],row,condition)
    st.caption(f"{file_id} · {idx/2:.1f}~{(idx+1)/2:.1f}초 · {SIGNALS[condition]} · 그룹 {row.group_id}은 fold {row.fold_id} 학습에서 제외")
    fig=go.Figure()
    x=idx/2+np.arange(len(wave))/HZ
    for j,(axis,color) in enumerate(zip("XYZ",("#08756e","#c67429","#6969a6"))):
        fig.add_scatter(x=x,y=wave[:,j],name=axis,line=dict(width=1,color=color))
    fig.update_layout(height=235,margin=dict(t=0,b=5),xaxis_title="기록 시작 후 시간 (초)",yaxis_title="가속도 (m/s²)",legend_orientation="h")
    st.plotly_chart(fig,width="stretch")
    truth=int(row.y_true)
    stage={"S":"슬래그 유동","E":"초기 슬래그 없음","B":"슬래그 유동 전"}[row.phase]
    st.write(f"**기록 정답　{row.phase} · {stage}**")
    if active:
        chosen = fresh[(fresh.file_id==file_id) & (fresh.window_idx==idx)]
        compared = current.replay(recording(paths[file_id]),idx,condition,chosen,model,outcome)
        # The previous default detector is shown with its original operating point.
        a0 = saved.loc["A0"]
        score, pred, reason = predict_signal(wave,bundle["models"]["A0"],protocol["features_I"])
        if reason or pred != a0.prediction or not np.isclose(score,a0.score,atol=1e-12,rtol=0):
            st.error("기존 기본 검출기의 입력 또는 점수 불일치"); return
        compared.insert(0, {"방식":"A0 · 기존 0.5", "원래 판정":"슬래그로 분류" if pred else "비슬래그로 분류",
                            "보류 여부":"보류 없음","점수 종류":"RF 슬래그 점수","모델 점수":f"{score:.4f}",
                            "임계값":"0.5000","정답과 비교":outcome(truth,pred)})
        st.dataframe(pd.DataFrame(compared),hide_index=True,width="stretch")
        st.subheader(f"{SIGNALS[condition]} 전체 480구간 · 보류 없이 비교")
        current.totals(stats,model,condition,status)
        return
    predicted=[]
    for name in ["A0","A1",model]:
        m=bundle["models"][name]
        score,p,reason=predict_signal(wave,m,protocol["features_I"])
        if reason:
            st.warning(f"입력 확인 필요 · {reason}");return
        s=saved.loc[name]
        if p!=s.prediction or not np.isclose(score,s.score,rtol=0,atol=1e-12):
            st.error("동일 입력의 재추론 값과 저장 결과가 달라 표시를 중단했습니다.");return
        predicted.append({"방식":LABEL[name],"원래 판정":"슬래그로 분류" if p else "비슬래그로 분류",
                          "보류 여부":"보류 없음","점수 종류":"RF 슬래그 점수" if name in ("A0","A1") else "SVC 결정 점수",
                          "모델 점수":f"{score:.4f}","임계값":f"{m['threshold']:.4f}","정답과 비교":outcome(truth,p)})
    st.dataframe(pd.DataFrame(predicted),hide_index=True,width="stretch")
    st.caption("점수가 임계값 이상이면 유동으로 판정합니다. RF 점수와 SVC의 결정 점수는 척도가 달라 각 모델의 임계값과 비교합니다.")
    is_adopted=decision["adopt"]==model
    if not is_adopted:
        st.info(f"{model} · 실험 후보 / 범용 개선 미입증. 기존 기본 모델 A0를 유지합니다. A1은 동일한 임계값 선택 절차로 비교하기 위한 기준선입니다." if decision["adopt"] is None else f"{model} · 실험 후보 / 범용 개선 미입증")
    st.subheader(f"{SIGNALS[condition]} 전체 480구간 · 보류 없이 비교")
    labels={"model_id":"모델","tp":"검출 TP","fn":"미검출 FN","fp":"오경보 FP","tn":"비슬래그 정답 TN","fp_E":"E 오경보","fp_B":"B 오경보",
            "recall":"재현율","fpr":"오경보율","precision":"정밀도","balanced_accuracy":"균형 정확도","fixed_vs_A1":"고친 오류","new_error_vs_A1":"새 오류"}
    totals=pd.DataFrame([dict(model_id=k,**summarize(g)) for k,g in subset.groupby("model_id")]).set_index("model_id")
    st.dataframe(totals[["tp","fn","fp","tn","fp_E","fp_B"]].rename(columns=labels),width="stretch")
    c,b=totals.loc[model],totals.loc["A1"]
    st.write(f"**{model} − A1**　검출 {int(c.tp-b.tp):+d}구간　·　오경보 {int(c.fp-b.fp):+d}구간")
    with st.expander("비율·fold별 결과와 채택 판정"):
        rates=totals[["recall","fpr","precision","balanced_accuracy"]].map(lambda v:f"{v:.2%}" if pd.notna(v) else "N/A")
        st.dataframe(rates.rename(columns=labels),width="stretch")
        st.caption("오경보율 10%는 학습 그룹 내부의 개발 목표입니다. 표에 표시한 오경보율은 평가 그룹에서 관측된 결과입니다.")
        v=decision["candidates"][model]
        st.write("원본에서 fold별 추가 검출 수 (A1 대비): "+", ".join(f"fold {k}: {n:+d}" for k,n in v["fold_tp_delta"].items()))
        violations=v["bad_recall_conditions"]+v["bad_fp_conditions"]
        if violations:st.write("교란 허용 폭을 벗어난 조건: "+", ".join(SIGNALS[k] for k in dict.fromkeys(violations)))
        st.caption("48개 파일의 상태 라벨을 480개 구간에 상속했습니다. 같은 번호의 E/B/S를 함께 나눈 개발 평가이며, 독립적인 480회 조업 실험이 아닙니다.")
        st.download_button("전체 모델·조건 평가 CSV", (DEST/"summary.csv").read_bytes(),"hi_summary.csv","text/csv")
    if model == "I":
        with st.expander("후보 I의 임계값별 검출률과 오경보율 · 내부 원본 검증"):
            scores = pd.read_csv(AUDIT/"inner_points.csv")
            scores["기준"] = scores.point.map({"native": "SVC 기본 경계", "selected": "학습 그룹에서 선택"})
            scores["S 검출률"] = scores.recall.map("{:.1%}".format)
            scores["오경보율"] = scores.fpr.map("{:.1%}".format)
            view = scores.rename(columns={"fold_id":"fold", "threshold":"임계값", "roc_auc":"AUC"})
            st.dataframe(view[["fold","기준","임계값","S 검출률","오경보율","AUC"]].round(4),
                         hide_index=True, width="stretch")
            st.caption("각 fold의 학습 그룹 내부 원본 예측(S 120개, E/B 240개)을 사용했습니다. "
                       "AUC는 실제 S의 점수가 비슬래그보다 높은 순서로 놓이는 정도입니다. "
                       "한 fold의 두 행은 같은 점수에 경계만 달리 적용합니다. "
                       "여러 fold의 내부 검증에는 같은 파일이 다시 들어갑니다.")
