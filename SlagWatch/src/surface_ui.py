"""SteelTrust's image-classification workflow inside the existing Streamlit application."""
import io
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st

from .surface_data import CFG,RUN,sha
from .surface_replay import CHANGES,LABELS,SurfaceRun,png

POLICIES={"V0":"필터 없음","V1":"3×3 가우시안 필터","V2":"5×5 가우시안 필터"}


@st.cache_resource
def surface_context(manifest_hash):
    return SurfaceRun(manifest_hash)


def presentation(label):
    return f"{LABELS[label]} ({label})"


def case_controls(ctx,after):
    source=st.radio("사례 모음",["검증 실패 탐색","전체 회귀 비교"],horizontal=True,key="neu_source_"+ctx.id)
    search=source=="검증 실패 탐색"
    split="validation"
    if not search and ctx.status["evaluated"] and after==ctx.seal["selected"]:
        split=st.selectbox("분할",ctx.man["splits"],format_func={"validation":"검증","test":"동결 후 시험"}.get,key="neu_case_split_"+ctx.id)
    scope=f"neu_{ctx.id[:10]}_{source}_{after}_{split}"
    left,right=st.columns(2)
    options=CFG["sigmas"] if search else [0.]+CFG["sigmas"]
    sigma=left.selectbox("가우시안 잡음 표준편차",options,format_func=lambda x:f"σ = {x:.2f}" if x else "원본",key=scope+"_sigma")
    seed=right.selectbox("고정 난수 seed",CFG[split+"_seeds"] if sigma else [0],disabled=not sigma,key=scope+f"_{sigma}_seed")
    sig=scope+f"_{sigma}_{seed}"
    pairs=ctx.pair_rows(split,after,sigma,seed)
    n=len(pairs)
    if search:
        a,b=st.columns(2)
        if a.button("선택 조건에서 실제 실패 탐색",type="primary",key=sig+"_search"):
            bar=st.progress(0,text="검증 원본과 교란 이미지를 추론하는 중")
            live=ctx.live(after,sigma,int(seed),lambda i,total:bar.progress(i/total,text=f"검증 이미지 {i}/{total}장"))
            st.session_state[sig+"_hits"]=live.loc[live.induced,"sample_id"].tolist()
            bar.empty()
        if b.button("저장된 탐색 결과 열기",key=sig+"_saved"):
            st.session_state.pop(sig+"_hits",None)
        saved=pd.read_csv(ctx.checked("validation_discoveries.csv.gz"))
        saved=saved[(saved.sigma==sigma)&(saved.seed==seed)].set_index("sample_id")
        ids=st.session_state.get(sig+"_hits",saved.index.tolist())
        if set(ids)!=set(saved.index):raise ValueError("실시간 탐색과 저장된 실패 목록 불일치")
        pairs=pairs[pairs.sample_id.isin(ids)]
        mode="이번 실시간 탐색" if sig+"_hits" in st.session_state else "저장된 실제 탐색"
        st.write(f"**{mode}** · 검증 원본 {n}장 중 원본 정분류가 교란 후 오분류로 바뀐 이미지 **{len(pairs)}장**")
    else:
        category=st.selectbox("변화 유형",["fixed","new_error","both_wrong","both_correct"],format_func=CHANGES.get,key=sig+"_change")
        pairs=pairs[pairs.change==category]
        sig+=category
        st.caption(f"선택한 조건의 전체 {n}장 중 해당 사례 {len(pairs)}장")
    if pairs.empty:
        st.info("선택한 조건에서 해당 사례가 없습니다.");return
    rows=pairs.sort_values("sample_id").set_index("sample_id")
    sample=st.selectbox("재생할 이미지",rows.index.tolist(),format_func=lambda x:f"{x} · {LABELS[rows.loc[x,'label']]} · {CHANGES[rows.loc[x,'change']]}",
                        key=sig+"_sample")
    arr=None
    if search:
        with ZipFile(ctx.checked("validation_discovered_inputs.zip")) as archive:
            arr=np.load(io.BytesIO(archive.read(saved.loc[sample,"npy_member"])),allow_pickle=False)
    return ctx.case(sample,after,sigma,int(seed),provided=arr)


def show_case(ctx,case):
    r,original,raw,changed=case
    after=r["after"];old=r["outputs"]["V0"];new=r["outputs"][after]
    name=f"{r['sample_id']} · {CHANGES[r['change']]} · {after} · σ {r['sigma']:.2f} · seed {r['seed']}"
    st.download_button("이 사례 ZIP · "+name,ctx.export(case),f"case_{r['sample_id']}_{r['execution_id'][:16]}.zip",
                       "application/zip",key="neu_zip_"+r["execution_id"],on_click="ignore")
    st.subheader(f"{name}　|　정답 {LABELS[r['label']]}")
    cols=st.columns(3)
    for col,arr,title in zip(cols,(original,raw,changed),
                             (f"원본 · V0 {LABELS[old['original']['prediction']]}",
                              f"같은 시험 입력 · V0 {LABELS[old['trial']['prediction']]}",
                              f"필터 적용 · {after} {LABELS[new['trial']['prediction']]}")):
        col.image(png(arr),caption=title,width="stretch")
    st.write(f"**관찰**　원본에서 V0는 {LABELS[old['original']['prediction']]}, 선택 입력에서는 "
             f"{LABELS[old['trial']['prediction']]}로 분류했습니다. 같은 입력에서 수정 후보는 {LABELS[new['trial']['prediction']]}로 분류했습니다.")
    st.caption("개발자 가설은 잡음이 미세한 결함 질감을 가리고, 평활화가 잡음과 질감을 함께 줄일 수 있다는 것입니다. "
               "수정은 입력 필터와 그 처리에 맞춘 동일 분류기의 재학습입니다. 인코더 가중치는 같습니다.")
    with st.expander("클래스별 결정 점수와 입력 해시"):
        table=pd.DataFrame({"결함":[LABELS[n] for n in r["class_order"]]})
        for m,g in r["outputs"].items():
            for part,label in (("original","원본"),("trial","선택 입력")):
                table[f"{m} {label}"]=g[part]["scores"]
        st.dataframe(table,hide_index=True,width="stretch")
        st.json(dict(raw_input_sha256=r["raw_input_hash"],processed_before=old["trial"]["processed_input_hash"],
                     processed_after=new["trial"]["processed_input_hash"],execution_id=r["execution_id"]))
        st.caption("점수는 분류기 결정 점수입니다. ZIP의 float32 raw_input.npy를 재생하며 PNG는 미리보기입니다.")


def population(ctx,after):
    st.subheader("전체 평가 · 조건별 원본 이미지 360장")
    splits=["validation"]+(["test"] if ctx.status["evaluated"] and after==ctx.seal["selected"] else [])
    split=st.selectbox("전체 평가 분할",splits,index=len(splits)-1,format_func={"validation":"후보 선택용 검증","test":"동결 후 최종 시험"}.get,key="neu_summary_"+ctx.id+after)
    summary=ctx.table(split,"summary");pairs=ctx.table(split,"changes")
    sel=summary[summary.model_id.isin(["V0",after])]
    m=sel.groupby(["sigma","model_id"]).macro_f1.mean().unstack()
    counts=pairs[pairs.after==after].groupby("sigma")[["fixed","new_error"]].sum()
    tab=m[["V0",after]].join(counts)
    tab.columns=["수정 전 macro F1","수정 후 macro F1","고친 오분류","새 오분류"]
    tab.index=["원본" if not s else f"σ {s:.2f}" for s in tab.index]
    st.dataframe(tab,width="stretch")
    s=sel[sel.sigma>0].groupby("model_id").macro_f1.mean()
    st.write(f"**9개 교란 조건 평균 macro F1**　{s['V0']:.4f} → **{s[after]:.4f}**　({s[after]-s['V0']:+.4f})")
    st.caption("원본 행은 360장을 한 번 평가했습니다. 잡음 행의 F1은 강도별 seed 3개의 평균이며, 고친·새 오류는 1,080개 짝지은 판정의 합계입니다. 같은 360장이 반복됩니다.")
    with st.expander("클래스별 재현율과 혼동행렬"):
        a,b=st.columns(2)
        m=a.selectbox("결과 모델",["V0",after],key="neu_matrix_"+ctx.id+after)
        sig=b.selectbox("혼동행렬 잡음", [0.]+CFG["sigmas"],key="neu_cm_sig_"+ctx.id)
        cm=ctx.table(split,"confusion");cm=cm[(cm.model_id==m)&(cm.sigma==sig)].pivot_table(index="label",columns="predicted",values="count",aggfunc="sum")
        st.dataframe(cm.rename(index=LABELS,columns=LABELS),width="stretch")
        cl=ctx.table(split,"classes")
        cl=cl[(cl.model_id==m)&(cl.sigma==sig)].groupby("label").recall.mean()
        st.dataframe(cl.rename(index=LABELS).rename("재현율"),width="stretch")
    scope=f"{after} · {'검증' if split=='validation' else '시험'}"
    st.download_button(f"조건별 전체 비교 CSV · {scope}",sel.to_csv(index=False).encode("utf-8-sig"),f"neu_{split}_{after}_summary.csv","text/csv",
                       key=f"neu_summary_csv_{ctx.id}_{split}_{after}",on_click="ignore")
    st.download_button(f"원본별 수정 전후 예측 CSV · {scope}",ctx.table(split,"pairs").query("after==@after").to_csv(index=False).encode("utf-8-sig"),
                       f"neu_{split}_{after}_pairs.csv","text/csv",key=f"neu_pair_csv_{ctx.id}_{split}_{after}",on_click="ignore")


def workbench():
    st.title("철강 표면 결함 분류 AI의 신뢰성 검증")
    st.caption("NEU-CLS · 열연강판 결함 6종 · 이미지 1장당 하나의 분류 · 합성 가우시안 잡음")
    experiment=st.radio("표면 결함 실험",["surface_g1","surface_v1"],horizontal=True,
                        format_func={"surface_g1":"잡음 학습 전후 · G1","surface_v1":"기존 필터 실험 · V1/V2"}.get,
                        key="neu_experiment")
    if experiment=="surface_g1":
        from .surface_g1_ui import workbench as g1_workbench
        return g1_workbench()
    ctx=surface_context(sha((RUN/"manifest.json").read_bytes()))
    zero=ctx.table("validation","summary").query("model_id=='V0' and sigma==0").iloc[0]
    st.write(f"필터 없는 기본 모델은 검증 원본 **{int(zero.n_originals)}장 중 {int(zero.correct)}장**을 맞혔습니다. "
             f"정확도 {zero.accuracy:.2%} · macro F1 **{zero.macro_f1:.4f}**")
    after=st.selectbox("수정 전처리",["V1","V2"],index=1 if ctx.seal["selected"]=="V2" else 0,
                       format_func=lambda n:f"{n} · {POLICIES[n]}",key="neu_candidate_"+ctx.id)
    if not ctx.status["evaluated"]:
        st.info("검증에서 채택 기준을 모두 통과한 필터가 없습니다. 아래 후보는 비교용이며 최종 시험 360장의 모델 평가는 미실행입니다.")
    else:
        st.info(f"검증 선택 {ctx.seal['selected']} · 최종 시험 {'개선 확인' if ctx.status['adopt'] else '개선 미입증'}")
    chosen=case_controls(ctx,after)
    if chosen is not None:show_case(ctx,chosen)
    population(ctx,after)
