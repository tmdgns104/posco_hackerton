"""Main workflow in the existing Streamlit app: reproduce one failure and inspect full regression."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .cnn_eval import paired
from .pair_runs import RUNS, ROCKET
from .pair_detector import file_hash
from .hi_ui import SIGNALS
from .pair_search import CHANGES, METHODS, ReplaySet



@st.cache_resource
def context(run,manifest_hash):
    return ReplaySet(run,manifest_hash)


def open_search(ctx,before,after):
    token=f"{ctx.run.name}_{before}_{after}"
    archive_name=f"search_{before}_{after}.json"
    key="live_"+ctx.manifest_hash+"_"+token
    a,b=st.columns(2)
    if a.button("실패 조건 자동 탐색",type="primary",key="cnn_search"):
        with st.spinner("같은 구간에 교란을 한 번씩 적용해 두 모델을 시험하고 있습니다."):
            st.session_state[key]=ctx.search(before,after)
    if b.button("저장된 탐색 결과 열기",key="cnn_saved"):
        st.session_state.pop(key,None)
    live=key in st.session_state
    run=st.session_state[key] if live else json.loads((ctx.run/archive_name).read_text(encoding="utf8"))
    if ctx.run.name==ROCKET.name and (run.get("manifest_sha256")!=ctx.manifest_hash or run.get("study")!=ctx.run.name):
        raise ValueError("탐색 결과의 평가 실행 ID 불일치")
    if (run["before"],run["after"])!=(before,after):
        raise ValueError("선택한 모델 쌍과 저장된 탐색 결과가 다릅니다.")
    hits=[r for r in run["records"] if r["before_induced"]]
    st.write(f"**{'이번 실행 결과' if live else '저장된 실제 탐색 결과'}** · {run['trials']}회 시험 · 원래 맞힌 판정이 뒤집힌 조건 {len(hits)}건")
    st.caption(f"{before}의 원본 오판 {run['excluded_original_errors']}개는 이 탐색에서 제외합니다. "
               f"원본 정답 {run['eligible']}개 중 {run['sample_size']}개를 골라 축·강도·위치·seed를 바꿉니다. "
               "전체 평가는 원본 오판을 포함한 480개를 사용합니다.")
    if ctx.run.name==ROCKET.name:
        st.caption(f"이번 결과 생성 시 모델 준비·워밍업 {run.get('warmup_seconds',0):.2f}초 · 실제 탐색 {run['elapsed_seconds']:.2f}초")
    pool=st.radio("탐색 사례 범위",["기존 판정이 뒤집힌 조건","전체 시험"],horizontal=True,key="pair_pool_"+token)
    records=hits if pool=="기존 판정이 뒤집힌 조건" else run["records"]
    if not records:
        st.info("해당 조건에서 발견되지 않음");return
    i=st.selectbox("동일 입력 재생",range(len(records)),key=f"pair_search_case_{token}_{pool}_{live}_{len(records)}",
        format_func=lambda i:f"{records[i]['file_id']} · 구간 {records[i]['window_idx']} · "
                             f"{records[i]['fault']['type']} ×{records[i]['fault']['scale']} · {CHANGES[records[i]['change']]}")
    return ctx.replay(records[i])


def open_regression(ctx,before,after):
    scope=f"{ctx.run.name}_{before}_{after}"
    a,b=st.columns(2)
    scenario=a.selectbox("전체 평가 입력",list(SIGNALS),format_func=SIGNALS.get,key="pair_condition_"+scope)
    category=b.selectbox("오류 변화",list(CHANGES),format_func=CHANGES.get,key="pair_change_"+scope)
    diff=paired(ctx.pred[ctx.pred.scenario==scenario],before,after)
    rows=diff[diff.change==category].sort_values(["group_id_before","file_id","window_idx"]).reset_index(drop=True)
    if rows.empty:
        st.info("해당 조건에서 발견되지 않음");return
    idx=st.selectbox(f"같은 구간 재생 · {len(rows)}개",range(len(rows)),
        format_func=lambda i:f"{rows.loc[i,'file_id']} · 구간 {rows.loc[i,'window_idx']}",key=f"pair_fixed_case_{scope}_{scenario}_{category}")
    return ctx.fixed_case(before,after,rows.iloc[idx])


def show_case(raw,test,r):
    before,after=r["before"],r["after"]
    filename=f"case_{before}_{after}_{r['execution_id'][:16]}.json"
    st.download_button(f"사례 JSON · {CHANGES[r['change']]} · {r['file_id']} #{r['window_idx']} · {before}→{after}",
                       json.dumps(r,ensure_ascii=False,indent=2).encode("utf8"),
                       filename,"application/json",key="case_json_"+r["execution_id"],on_click="ignore")
    st.subheader(f"{r['file_id']} · 구간 {r['window_idx']} · {CHANGES[r['change']]}")
    st.caption(f"파일 라벨 {r['phase']} · {r['window_idx']/2:.1f}~{(r['window_idx']+1)/2:.1f}초 · 그룹 {r['group_id']} 제외 fold {r['fold_id']} · 보류 없음")
    records=[]
    for name in (before,after):
        out=r["outputs"][name]
        for tag,col in (("실측 원본","clean"),("선택 입력","test")):
            result=out[col]
            records.append({"모델":name,"입력":tag,"원래 판정":"슬래그" if result["prediction"] else "비슬래그",
                            "모델 점수":f"{result['score']:.6f}","임계값":f"{out['threshold']:.6f}","보류 여부":"보류 없음"})
    st.dataframe(pd.DataFrame(records),hide_index=True,width="stretch")
    st.caption("각 점수는 같은 행의 임계값에 적용합니다. 모델 간 개선은 기록 정답 대비 판정으로 셉니다.")
    evidence=r["evidence"]; hypothesis=r["human_hypothesis"]
    st.write(f"**관찰**　{evidence}　 **개발자 가설**　{hypothesis}")
    st.write(f"**시험한 수정**　{r['modification']}")
    if r.get("inner_evidence"):
        v=r["inner_evidence"]
        st.caption(f"수정 근거는 outer 학습 그룹 내부의 별도 기록 {v['file_id']} · 구간 {v['window_idx']} · "
                   f"inner {v['inner_id']}입니다. 위 재생 파형은 outer 평가 구간입니다.")

    with st.expander("동일 입력 파형과 재현 식별자"):
        ax=st.radio("재생 축",list("xyz"),horizontal=True,key="pair_axis")
        j="xyz".index(ax)
        x=np.arange(len(raw))/6400+r["window_idx"]/2
        fig=go.Figure()
        fig.add_scatter(x=x,y=raw[:,j],name="실측 원본",line=dict(width=1,color="#007b78"))
        if r["source_sha256"]!=r["input_sha256"]:
            fig.add_scatter(x=x,y=test[:,j],name="합성 교란",line=dict(width=1,color="#d76b37"))
        fig.update_layout(height=240,xaxis_title="기록 시작 후 시간 (초)",yaxis_title="가속도 (m/s²)",
                          margin=dict(t=8,b=8),legend_orientation="h")
        st.plotly_chart(fig,width="stretch")
        st.json(dict(case_id=r["case_id"],input_sha256=r["input_sha256"],execution_id=r["execution_id"],
                     models={k:v["model_sha256"] for k,v in r["outputs"].items()},fault=r["fault"]),expanded=False)



def workbench():
    st.title("슬래그 검출 실패 재검증")
    st.caption("공개 SSFD · 0.5초 원시 진동 · 학습 제외 그룹 재생 · 합성 디지털 스트레스")
    study=st.selectbox("비교 실험",list(RUNS),format_func={"rocket_v1":"MiniRocket · 증강 비교","cnn_v1":"이전 CNN · 원시 파형 비교"}.get,key="pair_study")
    run=RUNS[study]
    manifest_hash=file_hash(run/"manifest.json")
    with st.spinner("실행 manifest, 데이터와 평가 결과를 확인하고 있습니다."):
        ctx=context(run,manifest_hash)
    a,b=st.columns(2)
    before=a.selectbox("수정 전 모델",list(dict.fromkeys(u for u,_ in ctx.pairs)),format_func=lambda k:k+" · "+METHODS[k],key="pair_before_"+study)
    after=b.selectbox("수정 후보",[v for u,v in ctx.pairs if u==before][::-1],format_func=lambda k:k+" · "+METHODS[k],key=f"pair_after_{study}_{before}")
    with st.spinner("선택 모델을 로드하고 첫 변환을 준비하고 있습니다. 이 시간은 탐색 제한에서 제외합니다."):
        ctx.warm(before,after)
    view=st.radio("사례 출처",["자동 실패 탐색","전체 회귀 사례"],horizontal=True,key=f"pair_view_{study}_{before}_{after}")
    case=open_search(ctx,before,after) if view=="자동 실패 탐색" else open_regression(ctx,before,after)
    if case:
        show_case(*case)
    st.subheader("같은 모델 쌍의 전체 회귀 결과")
    result=pd.read_csv(run/"pair_summary.csv")
    result=result[(result.before==before)&(result.after==after)]
    display=result.copy()
    display=display.sort_values("scenario",key=lambda s:s.map(list(SIGNALS).index))
    display["조건"]=display.scenario.map(SIGNALS)
    for side,name in (("before",before),("after",after)):
        display[name+" 검출 / 미검출 / 오경보 / 비슬래그 정답"]=display[[k+"_"+side for k in ("tp","fn","fp","tn")]].astype(str).agg(" / ".join,axis=1)
    display=display.rename(columns=CHANGES)
    st.dataframe(display[["조건",*[k for k in display if k.endswith("비슬래그 정답")],*[CHANGES[k] for k in ("rescued_FN","new_FN","removed_FP","new_FP")]]],
                 hide_index=True,width="stretch")
    verdict=json.loads((run/"verdict.json").read_text())[after]
    if run.name==ROCKET.name:
        st.info(" · ".join(["전체 검출 목표 "+("통과" if verdict["all_pass"] else "미달"),
                            "비교 기준 "+("충족" if verdict["comparison_eligible"] else "미충족"),
                            "채택" if verdict["adopt"] else "실험 후보 미채택"]))
    else:
        st.info("전체 검출 목표 통과 · 채택" if verdict["adopt"] else "검출 목표 또는 수정 효과 기준 미달 · 실험 후보 미채택")
    st.caption("조건별 S 160개, 비슬래그 320개. 목표는 원본 TP ≥112·FP ≤48, 각 교란 TP ≥96·FP ≤64입니다. "
               "이전 기본값 A0도 원본 목표 미달입니다. 같은 16개 그룹을 이용한 개발용 그룹 교차검증입니다.")
    st.download_button("선택 모델 쌍 전체 비교 CSV",result.to_csv(index=False).encode("utf-8-sig"),
                       f"regression_{before}_{after}.csv","text/csv",key=f"pair_csv_{before}_{after}",on_click="ignore")
