import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .improve import ROOT,OUT,CONDITIONS,design,signal,vector
from .impulse import repair
from .shape import shape
from .multi_axis import z_check
from .faults import inject_fault
from .steel_ui import context

FAMILY={"rf":"진폭 RF","spectral":"진폭 + 주파수 RF"}
METHOD={"A":"A · 기존 모델","B":"B · 잡음·충격 증강 학습","C":"C · 강건 특징 추가","D":"D · 증강 학습 + 강건 특징","E":"E · 고립 충격 보정 + 기존 모델","F":"F · 주파수 분포·축 관계","G":"G · F + 다축 충격 증강"}
CONDITION={"clean":"원본","noise":"잡음","spike":"X축 충격","new_y_spike":"다른 충격 (Y축)","new_z_spike":"추가 시험 (Z축)"}


@st.cache_data
def tables():
    fixed=pd.concat([pd.read_csv(OUT/n) for n in ("paired_scores.csv","impulse_scores.csv","shape_scores.csv","multi_scores.csv")],ignore_index=True)
    return fixed,pd.read_csv(OUT/"regression_2000.csv"),pd.read_csv(OUT/"gate_summary.csv")


@st.cache_resource
def learned(fold):
    return joblib.load(OUT/f"models/fold_{fold}.joblib")


@st.cache_resource
def shape_model(fold,variant):
    prefix="multi" if variant=="G" else "shape"
    return joblib.load(OUT/f"models/{prefix}_{fold}.joblib")


def candidate_score(changed,b,fold,family,variant):
    if variant in ("F","G"):
        return float(shape_model(fold,variant).predict_proba(shape(changed)[None,:])[0,1])
    applied=repair(changed) if variant=="E" else changed
    model=b[family] if variant=="E" else learned(fold)[family,variant]
    return float(model.predict_proba(design(vector(applied)[None,:],family,variant))[0,1])


def replay_comparison(r,family,variant,suite):
    ctx=context()
    w,fold=ctx["items"][r.file_id,int(r.window_idx)]
    b=ctx["bundles"][fold]
    if suite:
        changed,_,_=inject_fault(w,r.fault_type,r.axis,r.scale,r["count"],r.seed,b["scale"],
                                 json.loads(r.positions_json),json.loads(r.signs_json))
        if hashlib.sha256(changed.tobytes()).hexdigest()!=r.changed_window_hash:raise ValueError("시험 입력 해시 불일치")
    else:
        changed=z_check(w,b["scale"],r) if r.scenario=="new_z_spike" else signal(w,b["scale"],r,r.scenario)
    applied=repair(changed) if variant=="E" else changed
    a=b[family].predict_proba(design(vector(changed)[None,:],family,"A"))[0,1]
    v=candidate_score(changed,b,fold,family,variant)
    if not np.allclose([a,v],[r.A,r[variant]],atol=1e-12,rtol=0):raise ValueError("실행 점수와 저장 결과 불일치")
    return w,changed,applied,a,v


def comparison():
    fixed,reg,gates=tables()
    st.header("찾아낸 오판을 실제로 줄였을까?")
    st.write("공정 AI 검증 엔지니어가 같은 진동 입력을 기존 방식과 수정 방식에 넣어, 고친 오류와 새로 생긴 오류를 확인합니다.")
    pop=st.radio("평가 범위",["기존 2,000회 재시험","전체 480구간 평가"],horizontal=True,key="v4_scope")
    suite=pop.startswith("기존")
    a,b,c=st.columns(3)
    family=a.selectbox("비교할 모델",["rf"] if suite else ["spectral","rf"],format_func=FAMILY.get,key="v4_family")
    variant=b.selectbox("수정 방식",(["E","B","C","D"] if suite or family=="rf" else ["G","F","B","C","D","E"]),format_func=METHOD.get,key="v4_variant")
    if suite:
        fault=c.selectbox("탐색 교란",["전체","spike","noise"],format_func=lambda x:{"전체":"전체","spike":"스파이크","noise":"잡음"}[x])
        rows=reg.copy() if fault=="전체" else reg[reg.fault_type==fault].copy()
        st.caption("기존 진폭 RF가 원본을 맞힌 40개 구간에서 만든 회귀 시험입니다. 각 교란 배열의 SHA256을 확인했습니다. 모든 분류 결과를 보류 없이 비교합니다.")
    else:
        cond=c.selectbox("시험 신호",list(CONDITION) if variant in ("F","G") else list(CONDITION)[:-1],format_func=CONDITION.get,key="v4_signal")
        selected=fixed[(fixed.family==family)&(fixed.scenario==cond)]
        keys=["file_id","window_idx","y_true","group_id","phase","fold_id","scenario"]
        rows=selected.pivot(index=keys,columns="variant",values="score").reset_index()
        st.caption("학습과 그룹이 겹치지 않는 16개 그룹의 모든 구간입니다. S 160개, 비S 320개입니다.")
    old_err=(rows.A>=.5)!=rows.y_true;new_err=(rows[variant]>=.5)!=rows.y_true
    fix=old_err&~new_err;regress=~old_err&new_err
    a,b,c,d=st.columns(4)
    a.metric("기존 오류",f"{old_err.sum()} / {len(rows)}")
    b.metric("수정 후 오류",f"{new_err.sum()} / {len(rows)}",delta=int(new_err.sum()-old_err.sum()),delta_color="inverse")
    c.metric("고친 오류",int(fix.sum()));d.metric("새 오류",int(regress.sum()))
    lines=[]
    for key in ("A",variant):
        pos=rows.y_true==1; alert=rows[key]>=.5
        lines.append({"방식":METHOD[key],"슬래그 검출":f"{(alert&pos).sum()} / {pos.sum()}",
                      "미검출":int((~alert&pos).sum()),"오경보":int((alert&~pos).sum()),
                      "비슬래그 구간":int((~pos).sum()),"균형 정확도":f"{(alert[pos].mean()+(~alert[~pos]).mean())/2:.1%}"})
    st.dataframe(pd.DataFrame(lines),hide_index=True,width="stretch")
    if suite and variant=="E":
        st.caption("E의 개선은 이 회귀 세트의 순간 충격에서 발생했습니다. 전체 구간 평가에서 원본 진폭 RF는 여전히 슬래그 160개 중 110개를 놓칩니다.")
    st.subheader("같은 신호로 판정 재현")
    types={"고친 오류":fix,"새 오류":regress,"여전히 틀림":old_err&new_err,"둘 다 정답":~old_err&~new_err}
    available=[k for k,mask in types.items() if mask.any()]
    kind=st.radio("사례 유형",available,horizontal=True)
    sub=rows[types[kind]]
    def label(i):
        r=sub.loc[i];extra=f" · {r.fault_type}/{r.axis} ×{r.scale:g} · 시험 {r.trial_order}" if suite else ""
        return f"{r.file_id} · 구간 {r.window_idx}"+extra
    idx=st.selectbox("재현할 사례",sub.index,format_func=label)
    r=sub.loc[idx]
    w,changed,applied,old,new=replay_comparison(r,family,variant,suite)
    txt=lambda pred:"유동" if pred else "비유동"
    st.write(f"기록 정답 **{txt(r.y_true)}**　 기존 **{txt(old>=.5)} ({old:.3f})** → 수정 후 **{txt(new>=.5)} ({new:.3f})**")
    if suite:st.caption(f"원본 점수 {r.clean_score:.3f} · 그룹 {r.group_id} 평가 전용 fold {r.fold_id} · seed {r.seed} · {r.scenario_id[:16]}")
    else:st.caption(f"그룹 {r.group_id} 평가 전용 fold {r.fold_id} · 판정 경계 0.5")
    axis=st.radio("표시 축",["X","Y","Z"],horizontal=True,key="v4_axis");j="XYZ".index(axis)
    zoom=st.checkbox("변화가 가장 큰 주변 96샘플 확대",value=True)
    center=int(np.argmax(np.abs(changed[:,j]-w[:,j]))) if np.any(changed[:,j]!=w[:,j]) else int(np.argmax(np.abs(w[:,j])))
    left=max(0,min(3200-96,center-48)); sl=slice(left,left+96) if zoom else slice(None)
    x=(np.arange(3200)/6400+r.window_idx/2)[sl]
    traces=[("실측 원본",w,"#657588"),("시험 입력",changed,"#d96f23")]
    if variant=="E":traces.append(("보정 후 입력",applied,"#007d70"))
    fig=go.Figure()
    for name,z,color in traces:
        fig.add_scatter(x=x,y=z[sl,j],name=name,line=dict(color=color,width=2,dash="dot" if name=="보정 후 입력" else "solid"))
    fig.update_layout(height=270,margin=dict(t=10,b=15),legend_orientation="h",xaxis_title="기록 시작 후 시간 (초)",yaxis_title=axis+"축 · 원본 단위")
    st.plotly_chart(fig,width="stretch")
    with st.expander("실험 설정과 조건별 전수 결과"):
        st.write("각 fold의 학습 그룹 12개와 평가 그룹 4개를 고정했습니다. B·D는 학습 구간 360개에 잡음·충격 사본을 하나씩 더해 1,080개로 학습했습니다. C·D는 축별 MAD, IQR, 중앙 90% RMS를 추가했습니다. E는 이웃 두 샘플을 기준으로 고립 피크를 보정하고 기존 모델을 사용합니다. F는 대역별 상대 에너지와 축 간 관계를 로지스틱 회귀로 학습하고 G는 훈련 충격을 여러 축과 세기로 분산합니다.")
        show=pd.concat([pd.read_csv(OUT/"scores_summary.csv"),pd.read_csv(OUT/"impulse_summary.csv"),pd.read_csv(OUT/"shape_summary.csv"),pd.read_csv(OUT/"multi_summary.csv")])
        st.dataframe(show[show.family==family][["variant","scenario","tp","fn","fp","balanced_accuracy","macro_f1"]],hide_index=True)
    if not suite:
        with st.expander("그룹별로 고친 오류와 새 오류"):
            grouped=rows.assign(기존오류=old_err.astype(int),수정후오류=new_err.astype(int),
                                고친오류=fix.astype(int),새오류=regress.astype(int)).groupby("group_id")[
                                    ["기존오류","수정후오류","고친오류","새오류"]].sum()
            st.dataframe(grouped,width="stretch")
    with st.expander("보류 규칙을 같은 자동 처리량에서 비교"):
        gate_target=st.radio("보류를 적용할 분류기",["E","G"],format_func={"E":"E · 충격 보정 진폭 RF","G":"G · 다축 증강 모델"}.get,horizontal=True)
        if gate_target=="G":gates=pd.read_csv(OUT/"gate_summary_G.csv")
        st.caption(" 각 바깥 학습 그룹 안에서 3회 그룹 분할로 별도 예측을 만들어 보류 기준을 학습했습니다. 정답과 파형 품질은 평가 용도가 다릅니다.")
        mode=st.radio("비교 기준",["matched","calibrated"],format_func=lambda x:{"matched":"정확히 80% 전달 · 배치 비교","calibrated":"학습 그룹에서 정한 80% 목표 기준 · 실제 전달률"}[x],horizontal=True)
        show=gates[gates["mode"]==mode].copy()
        show["policy"]=show.policy.map({"margin80":"점수 경계 거리","risk80":"오류 위험 추정"})
        st.dataframe(show[["scenario","policy","passed","errors_pass","conditional_error","fn","fp","correct_hold"]].rename(columns={"scenario":"조건","policy":"정책","passed":"전달","errors_pass":"오판 전달","conditional_error":"전달 중 오류율","fn":"미검출 전달","fp":"오경보 전달","correct_hold":"정답 보류"}),hide_index=True)
        st.caption("정확히 80% 비교는 조건별 평가 120건 묶음에서 96건을 점수 순으로 고릅니다. 온라인 개별 판정 결과는 학습 그룹에서 정한 기준으로 확인합니다.")
