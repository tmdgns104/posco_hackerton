"""Small offline inspection screen for the cost of holding a correct slag alarm."""
import numpy as np
import pandas as pd
import streamlit as st

from .safety_eval import SAFETY,SEEDS,KEY,random_decisions
from .hi_ui import LABEL,SIGNALS
from .score_audit import AUDIT

MODELS={**LABEL,"E":"E · 고립 피크 보정 / 진폭 RF","G":"G · 다축 증강 모델"}
METRICS={"TP_pass":"슬래그 경보 전달","FN_pass":"미검출 전달","H_TP":"정확한 경보 보류","H_FN":"기존 미검출 보류",
         "FP_pass":"오경보 전달","hold_count":"전체 보류","pass_count":"전체 전달",
         "tp_pass_rate":"전달된 슬래그 검출률","fn_pass_rate":"미검출 통과율","fp_pass_rate":"오경보 통과율",
         "coverage":"전체 전달률","coverage_S":"S 전달률","coverage_E":"E 전달률","coverage_B":"B 전달률","coverage_nonS":"E/B 합계 전달률"}
POLICY={"none":"보류 없음","margin":"기존 경계 거리 · 실험","proposed":"제안 정책 · 실험","random":"무작위 보류"}


@st.cache_data
def safety_data():
    return tuple(pd.read_csv(SAFETY/f,float_precision="round_trip") for f in
                 ["policies.csv","group_runs.csv.gz","random_summary.csv","random_groups.csv.gz","cases.csv.gz"])


def display_number(x,k,average=False):
    if pd.isna(x):return "N/A"
    if "rate" in k or "coverage" in k:return f"{x:.1%}"
    return f"{x:.1f}" if average else str(int(x))


def comparison_table(tab,rnd,columns):
    table=[]
    for row in tab.to_dict("records"):
        table.append({"정책":POLICY[row["policy"]],**{METRICS[k]:display_number(row[k],k) for k in columns}})
    table.append({"정책":f"무작위 · {len(SEEDS)}개 seed 평균 ± 표준편차",
                  **{METRICS[k]:display_number(rnd[k+"_mean"],k,True)+" ± "+display_number(rnd[k+"_std"],k,True) for k in columns}})
    return pd.DataFrame(table)


def case_masks(cases,policy,mode,seed):
    rows=cases[cases.policy==("margin" if policy in ("none","random") else policy)].sort_values(KEY).reset_index(drop=True).copy()
    if policy=="none":rows["passed"]=True
    elif policy=="random":
        rows["passed"]=False
        for fold,g in rows.groupby("fold_id"):
            rows.loc[g.index,"passed"]=random_decisions(len(g),fold,seed,mode)
    else:rows["passed"]=rows["pass_"+mode]
    return rows


def safety_home():
    totals,group,rnd,rgroup,case=safety_data()
    st.title("안전 지원 검증")
    st.write("공정 AI 검증 담당자가 한 검출기의 같은 예측을 놓고, 잘못 전달한 미검출과 검토로 넘어간 정확한 슬래그 경보를 함께 확인합니다.")
    a,b,c=st.columns([1.2,1,1.5])
    model=a.selectbox("검출기",["I","A1","A0","H","E","G"],format_func=MODELS.get,key="safety_model")
    scenario=b.selectbox("입력 조건",list(SIGNALS),format_func=SIGNALS.get,key="safety_condition")
    mode=c.selectbox("검토량 비교",["matched","calibrated"],format_func={"matched":"오프라인 배치 · 정확히 80% 전달","calibrated":"고정 기준 · 실제 전달률 확인"}.get,key="safety_mode")
    if mode=="matched":st.caption("각 시험을 별도 배치로 재생해 fold당 120개 중 96개를 전달합니다. 고정 기준의 평가 데이터 전달률은 ‘고정 기준’에서 확인합니다.")
    else:st.caption("학습 그룹에서 정한 기준을 모든 평가 조건에 그대로 적용합니다. 무작위 기준선은 고정 확률 0.8을 사용하며 실제 전달 건수를 표시합니다.")
    if model in ("E","G"):st.caption("기존 v4 점수·정책 재사용 · 제안 정책은 저장된 오류 위험 추정, 기존 정책은 원래 모델 점수와 판정 경계의 거리입니다.")
    else:st.caption("H/I 실험 점수 재사용 · 제안 정책은 비교 모델과의 판정 불일치를 먼저, 그다음 경계에 가까운 순으로 검토합니다. A0/A1의 비교 모델은 H, H/I의 비교 모델은 A1입니다.")
    fil=lambda df:df[(df.model_id==model)&(df.scenario==scenario)]
    ref=fil(totals)
    table=ref[ref["mode"].isin(["reference",mode])].copy()
    table["order"]=table.policy.map({"none":0,"margin":1,"proposed":2})
    table=table.sort_values("order")
    random=fil(rnd).query("mode==@mode").iloc[0]
    base=table[table.policy=="none"].iloc[0];candidate=table[table.policy=="proposed"].iloc[0]
    st.subheader("원래 경보 중 몇 건이 전달됐나")
    existing = table[table.policy == "margin"].iloc[0]
    st.write(f"기존 경계 거리 정책은 정확한 슬래그 경보 **{int(base.TP_pass)}건 중 "
             f"{int(existing.TP_pass)}건 전달 · {int(existing.H_TP)}건 보류**.")
    st.write(f"제안 정책에서는 정확한 슬래그 경보 **{int(base.TP_pass)}건 중 {int(candidate.TP_pass)}건 전달 · {int(candidate.H_TP)}건 보류**. "
             f"원래 미검출 **{int(base.FN_pass)}건 중 {int(candidate.FN_pass)}건 전달 · {int(candidate.H_FN)}건 보류**.")
    st.caption(f"실제 슬래그 {int(base.N_S)}구간, 비슬래그 {int(base.N_N)}구간. 검토 대상으로 보류된 미검출도 모델의 미검출로 남습니다.")
    st.dataframe(comparison_table(table,random,list(METRICS)[:7]),hide_index=True,width="stretch")
    st.dataframe(comparison_table(table,random,["tp_pass_rate","fn_pass_rate","fp_pass_rate","coverage","coverage_S","coverage_E","coverage_B","coverage_nonS"]),
                 hide_index=True,width="stretch")
    st.info("검출 시연의 기본은 보류 없음입니다. 기존·제안 정책은 모두 실험용입니다.")
    if model == "I":
        st.subheader("실제 슬래그에서 A1과 I의 판정 비교")
        cross = pd.read_csv(AUDIT/"paired_s.csv")
        current = cross[(cross.scenario == scenario) & (cross["mode"] == mode)]
        labels = {"category":"A1과 I의 원래 결과", "n":"실제 S 구간", "margin_held":"기존 정책 보류",
                  "proposed_held":"불일치 우선 보류"}
        st.dataframe(current[list(labels)].rename(columns=labels), hide_index=True, width="stretch")
        st.caption("두 보류 열 모두 I의 판정에 적용한 결과입니다. 보류 여부를 정한 뒤 실제 S만 골라 셌습니다.")
    st.subheader("한 구간의 원래 판정과 보류 결과")
    left,right=st.columns(2)
    policy=left.selectbox("확인할 정책",["none","margin","proposed","random"],format_func=POLICY.get,key="case_policy")
    seed=right.selectbox("사례 재현용 무작위 seed",SEEDS,disabled=policy!="random")
    rows=case_masks(fil(case),policy,mode,seed)
    options={"전체":np.ones(len(rows),bool),"정확한 슬래그 경보 보류":(rows.y_true==1)&(rows.prediction==1)&~rows.passed,
             "미검출 판정 보류":(rows.y_true==1)&(rows.prediction==0)&~rows.passed,
             "미검출 판정 전달":(rows.y_true==1)&(rows.prediction==0)&rows.passed}
    if model == "I":
        peer = case[(case.model_id=="A1") & (case.scenario==scenario) & (case.policy=="margin")]
        rows = rows.merge(peer[KEY+["prediction"]].rename(columns={"prediction":"A1"}),
                          on=KEY, validate="one_to_one", how="left")
        if rows.A1.isna().any():
            raise ValueError("Missing A1 prediction for the same window")
        options["A1이 놓치고 I가 검출"] = (rows.y_true==1) & (rows.A1==0) & (rows.prediction==1)
    kind=st.radio("사례 유형",list(options),horizontal=True,key="safety_case_type")
    selected=rows[options[kind]]
    if not selected.empty:
        idx=st.selectbox("측정 파일 · 구간",selected.index,format_func=lambda i:f"{rows.loc[i,'file_id']} · 구간 {rows.loc[i,'window_idx']}")
        r=rows.loc[idx];pred="슬래그로 분류" if r.prediction else "비슬래그로 분류"
        st.write(f"**기록 정답 {r.phase}**　|　**원래 예측 {pred}**　|　**정책 처리 {'전달' if r.passed else '검토 대상으로 보류'}**")
        st.caption(f"모델 점수 {r.score:.4f} · fold {r.fold_id} 검출 임계값 {r.threshold:.4f} · "
                   f"경계까지의 점수 차이 {abs(r.score-r.threshold):.4f} · 학습 제외 그룹 {r.group_id}")
        st.caption(f"자동 전달값 · {pred if r.passed else '없음 (검토 대상으로 보류)'}")
        if model == "I":
            st.caption(f"같은 구간의 A1 원래 예측 · {'슬래그로 분류' if r.A1 else '비슬래그로 분류'}")
    else:st.caption("이 조합에서 해당 사례는 0건입니다.")
    with st.expander("그룹별 결과와 무작위 변동 확인"):
        cols=["group_id","policy","TP_pass","FN_pass","FP_pass","H_TP","H_FN","hold_count","coverage_S","coverage_E","coverage_B"]
        v=fil(group); v=v[v["mode"].isin([mode,"reference"]) & (v.policy!="random")]
        st.dataframe(v[cols].rename(columns=METRICS),hide_index=True,width="stretch")
        rv=fil(rgroup); cols2=["TP_pass","FN_pass","FP_pass","H_TP","H_FN","hold_count"]
        st.dataframe(rv[rv["mode"]==mode][["group_id"]+[k+"_"+s for k in cols2 for s in ("mean","std")]],
                     hide_index=True,width="stretch")
        st.caption("±는 고정 seed 0~31에서의 표본 표준편차입니다. 그룹과 창의 현장 발생 빈도 또는 사고 확률을 뜻하지 않습니다. 지표 분모가 0이면 N/A입니다.")
        st.download_button("전체 조건·정책 CSV",(SAFETY/"policies.csv").read_bytes(),"safety_policies.csv","text/csv")
        st.download_button("무작위 기준선 평균·변동 CSV",(SAFETY/"random_summary.csv").read_bytes(),"safety_random.csv","text/csv")
    with st.expander("산업적 근거와 검증 범위"):
        st.markdown("""- [포스코 공개특허 KR20040056051A, 종래기술 및 발명의 목적](https://patents.google.com/patent/KR20040056051A/ko) · 래들→턴디시에서 강종·슬라이딩 게이트 개도에 따른 패턴 차이와 조업자·크레인 이동 진동의 간섭을 설명하고 개선 방법을 제시했습니다.
- [이영진 명장 인터뷰, 2024.12.10](https://newsroom.posco.com/kr/포스코명장-특별인터뷰㉓-포항-제강부-이영진-명장/) · 전로→래들 출강 자동화가 작업자별 품질 편차와 안전에 미친 영향을 소개한 과거 사례입니다.
- [Ergolines Optical Slag Detector](https://www.ergolines.it/products/ladle-optical-slag-detector-osd/) · 제조사가 래들 광학 감시의 원격 영상 확인과 위험 구역 인원 접근 감소를 제품 장점으로 설명합니다.
- [Kiss KT2000 LadleSLAG](https://kisstechnologies.com/kt2000-ladleslag/) · 제조사가 래들→턴디시에서의 진동 감지, 감도 조정, 혼입과 수율 간 선택, 자동 게이트 폐쇄 옵션을 설명합니다.""")
        st.write("SteelTrust의 실제 측정값은 공개 SSFD의 모델 오류, 문헌의 진동 간섭 문제를 참고한 합성 교란 민감도, 판단 전달·보류의 대리지표입니다. 작업자 접근, 사고, 설비 정지, 슬래그 유출량에 대한 현장 효과는 측정하지 않았습니다.")
