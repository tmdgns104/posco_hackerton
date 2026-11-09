"""Streamlit workbench for deterministic failure discovery and policy review."""
import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .search import ROOT,load_context,replay,search_failures
from .metrics_gate import evaluate_rows
from .gate import MARGIN

@st.cache_resource
def context():
    return load_context()

MODELS={"rf":"진폭 RF (12개 특징)","spectral":"진폭 + 주파수 RF (36개 특징)"}
OUTCOMES={"gate_caught_error":"오판 보류","gate_missed_error":"오판 전달","unnecessary_abstention":"정답 보류","correct_pass":"정답 전달"}
FAILURES={"induced_false_positive":"새 오경보","induced_false_negative":"새 미검출","none":"정답 유지"}
CONDS={"clean":"원본","noise":"잡음","spike":"순간 충격","missing":"입력 누락","freeze":"값 고정"}


def search_tab():
    st.header("어떤 교란이 맞았던 판정을 뒤집을까?")
    st.caption("같은 0.5초 구간에서 입력만 바꿉니다. 원본과 변형 신호 모두 해당 그룹을 학습에서 제외한 모델로 판정합니다.")
    with st.expander("탐색 설정"):
        a,b=st.columns(2)
        model=a.selectbox("시험 모델",list(MODELS),format_func=MODELS.get)
        direction=b.selectbox("찾을 오류",["both","0","1"],format_func={"both":"오경보와 미검출","0":"오경보","1":"미검출"}.get)
        a,b=st.columns(2)
        faults=a.multiselect("교란 종류",["noise","spike"],["noise","spike"],format_func=CONDS.get)
        budget=b.number_input("최대 시험 횟수",min_value=1,max_value=2000,value=400)
    a,b=st.columns([1,1])
    run=a.button("실패 조건 자동 탐색",type="primary",disabled=not faults)
    if b.button("저장된 탐색 결과 열기"):st.session_state.pop("steel_live",None)
    if run:
        bar=st.progress(0,text="OOF 데이터 준비 중")
        def update(n,h,t):
            bar.progress(min(1.,n/int(budget)),text=f"{n:,}회 시험 · 새 오판 {h}건 · {t:.1f}초")
        st.session_state.steel_live=search_failures(context(),model,direction,tuple(faults),int(budget),90,update)
    live="steel_live" in st.session_state
    trials=st.session_state.steel_live if live else pd.read_csv(ROOT/"reports/search_results.csv")
    st.caption("이번 버튼으로 수행한 탐색" if live else "저장된 탐색 결과 · 실제 사전 실행")
    if trials.empty:
        st.info("시간 예산 내에 완료한 시험이 없습니다.");return
    failed=trials[trials.failure_type!="none"]
    a,b,c,d=st.columns(4)
    for box,title,value in [(a,"시험 조건",len(trials)),(b,"유발된 오판",len(failed)),(c,"오판 보류",(failed.gate_decision=="HOLD").sum()),(d,"오판 전달",(failed.gate_decision=="PASS").sum())]:
        box.metric(title,f"{value:,}")
    st.caption(f"시험 모델 {MODELS[trials.model_id.iloc[0]]} · 원본 정답 구간 {len(trials.drop_duplicates(['file_id','window_idx']))}개 · 발견률 {len(failed)/len(trials):.1%} (시험 조건 기준) · G2 마진 0.10")
    if failed.empty: st.info("현재 예산과 그리드에서는 새 오판을 발견하지 못했습니다.")
    with st.expander("시험 내역과 최소 관측 실패 강도"):
        if not failed.empty:
            st.dataframe(failed.groupby(["fault_type","axis","failure_type"]).scale.min().rename("최소 관측 강도").reset_index(),hide_index=True)
        st.download_button("전체 시도 CSV",trials.to_csv(index=False).encode("utf-8-sig"),"search_results.csv","text/csv")
        pre=context()["clean"]; n=trials.model_id.iloc[0]+"_pred"
        st.caption(f"이 모델의 원본 오판 {int((pre[n]!=pre.y_true).sum())}구간은 탐색 대상에서 제외했습니다. 발견률은 현장 발생률과 다릅니다.")
    st.subheader("발견 조건 재현")
    category=st.radio("볼 사례",list(OUTCOMES),format_func=OUTCOMES.get,horizontal=True)
    subset=trials[trials.gate_outcome==category]
    if subset.empty:st.info("해당 유형의 사례가 없습니다.");return
    def display(i):
        r=subset.loc[i]
        return f"{r.file_id} · 구간 {r.window_idx} · {FAILURES[r.failure_type]} · {r.fault_type}/{r.axis} ×{r.scale:g} · #{r.trial_order}"
    choice=st.selectbox("사례를 선택하면 동일 조건을 재실행합니다",subset.index,format_func=display)
    row=subset.loc[choice]
    original,changed,result=replay(context(),row)
    if result["decision"]=="HOLD":
        st.warning(f"검토 요청 · {', '.join(result['reasons'])} · 원본 점수 {row.clean_score:.3f} → 교란 후 {result['score']:.3f}")
    else:
        st.info(f"판단 전달 · 원본 점수 {row.clean_score:.3f} → 교란 후 {result['score']:.3f}")
    label=lambda v:"유동" if int(v) else "비유동"
    st.write(f"실제 라벨 **{label(row.y_true)}**　 원본 예측 **{label(row.clean_pred)}**　 교란 후 예측 **{label(result['pred'])}**")
    with st.expander("이 실패를 수정 후보로 다시 시험"):
        from .comparison_ui import candidate_score, METHOD
        v=st.selectbox("다시 시험할 수정",["E","B","C","D","F","G"],format_func=METHOD.get)
        score=candidate_score(changed,context()["bundles"][int(row.fold_id)],int(row.fold_id),row.model_id,v)
        good=result["pred"]==row.y_true
        new_good=int(score>=.5)==row.y_true
        status=("둘 다 정답" if good else "고친 오류") if new_good else ("새 오류" if good else "여전히 틀림")
        st.write(f"**{status}** · 같은 시험 입력 · 기존 점수 {result['score']:.3f} → 수정 후 {score:.3f}")
        st.caption("수정 후보의 판정은 보류 없이 비교합니다. F·G는 별도 분류기입니다.")
    ax=st.radio("파형 축",["x","y","z"],horizontal=True);i="xyz".index(ax)
    x=np.arange(len(original))/6400+row.window_idx/2
    fig=go.Figure()
    for title,array,color in [("실측 원본",original,"#007d72"),("시험 입력",changed,"#e17736")]:
        fig.add_scatter(x=x,y=array[:,i],name=title,line=dict(width=1,color=color))
    pos=json.loads(row.positions_json)
    if pos and (row.axis=="all" or ax=="x"):
        fig.add_scatter(x=x[pos],y=changed[pos,i],mode="markers",name="주입 위치",marker=dict(color="#a9352d",size=6))
    fig.update_layout(height=300,margin=dict(t=10,b=18),xaxis_title="기록 시작 후 시간 (초)",yaxis_title=ax+"축 · 원본 단위",legend_orientation="h")
    st.plotly_chart(fig,width="stretch")
    with st.expander("재현 근거 · 입력·교란 배열 해시와 모델 확인 완료"):
        st.json(row.to_json(force_ascii=False))
    st.download_button("선택 사례 JSON",row.to_json(force_ascii=False,indent=2),"failure_case.json","application/json")


def gate_tab():
    st.header("오판을 보류하는 대가로 얼마나 자주 멈추는가?")
    st.caption("각 조건의 전체 480구간 평가입니다. 탐색에서 발견한 사례만 골라 계산하지 않습니다.")
    metrics=pd.DataFrame(json.loads((ROOT/"reports/gate_metrics.json").read_text()))
    pred=pd.read_csv(ROOT/"reports/predictions.csv")
    a,b,c=st.columns(3)
    model=a.selectbox("평가 모델",list(MODELS),format_func=MODELS.get)
    cond=b.selectbox("고정 시험 조건",list(CONDS),format_func=CONDS.get)
    pol=c.selectbox("판단 게이트",["G2","G0","G1","G3"],format_func={"G0":"G0 · 모델 단독","G1":"G1 · 입력 검사","G2":"G2 · 입력 + 점수 마진","G3":"G3 · 마진 + 두 모델 불일치"}.get)
    base=metrics[(metrics.scenario==cond)&(metrics.model==model)]
    m=base[(base.policy==pol)&(base.margin==MARGIN)].iloc[0]
    a,b,c,d=st.columns(4)
    for box,k,title in [(a,"error_pass_rate","전체 중 오판 전달"),(b,"block_rate","오판 중 보류"),(c,"unnecessary_rate","정답 중 보류"),(d,"coverage","자동 전달률")]:
        box.metric(title,"N/A" if pd.isna(m[k]) else f"{m[k]:.1%}")
    st.write(f"잘못 전달한 판정 **{int(m.FP+m.FN)}건** (오경보 {int(m.FP)}, 미검출 {int(m.FN)}) · 판단이 맞았지만 보류 **{int(m.unnecessary_hold)}건** · 입력 품질 보류 **{int(m.quality_held)}건**")
    st.caption(f"전달한 {int(m.passed)}건 중 오류율 {m.conditional_error:.1%}. G0에서도 손상 입력은 모델에 넣지 않으며 출력 불가로 기록합니다.")
    comp=base[base.margin==MARGIN]
    rates={"error_pass_rate":"전체 중 오판 전달","unnecessary_rate":"정답 중 보류"}
    st.plotly_chart(px.bar(comp.rename(columns=rates),x="policy",y=list(rates.values()),barmode="group",labels={"value":"비율","policy":"정책","variable":"지표"}).update_yaxes(tickformat=".0%").update_layout(height=285,margin=dict(t=8,b=5)),width="stretch")
    with st.expander("마진 민감도 · 자동 전달률과 오판 전달률"):
        curve=base[base.policy=="G2"].sort_values("margin")
        st.plotly_chart(px.line(curve,x="coverage",y="error_pass_rate",text="margin",markers=True,labels={"coverage":"자동 전달률","error_pass_rate":"전체 중 오판 전달률"}).update_traces(textposition="top center").update_xaxes(tickformat=".0%").update_yaxes(tickformat=".0%"),width="stretch")
        st.caption("0, 0.05, 0.10, 0.15, 0.20을 같은 OOF 데이터에서 비교한 민감도 분석입니다. 기본값 0.10은 비교 전에 고정했습니다.")
    with st.expander("원본·잡음·충격에서 두 RF의 오경보와 미검출"):
        fixed=metrics[(metrics.policy=="G0") & metrics.scenario.isin(["clean","noise","spike"])].copy()
        fixed["조건"]=fixed.scenario.map(CONDS);fixed["모델"]=fixed.model.map(MODELS)
        st.plotly_chart(px.bar(fixed,x="조건",y=["FP","FN"],facet_col="모델",barmode="group").update_layout(height=300),width="stretch")
    with st.expander("게이트가 놓친 오류와 그룹별 편차"):
        selected=evaluate_rows(pred[pred.scenario==cond],model,pol,MARGIN)
        missed=selected[(selected.decision=="PASS")&(selected.forwarded_pred!=selected.y_true)]
        st.dataframe(missed[["file_id","window_idx","y_true",model+"_score","forwarded_pred"]],hide_index=True)
        groups=pd.read_csv(ROOT/"reports/gate_by_group.csv")
        st.dataframe(groups[(groups.model==model)&(groups.scenario==cond)&(groups.policy==pol)][["group_id","N","FP","FN","caught","unnecessary_hold","coverage"]],hide_index=True)
