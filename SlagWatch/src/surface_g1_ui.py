"""G1 workbench embedded in the shared NEU-CLS Streamlit route."""
import pandas as pd
import streamlit as st

from .surface_g1 import DEFAULT_G1, file_sha, read_json
from .surface_g1_replay import G1Run
from .surface_replay import CHANGES, LABELS, png
from .surface_ui import case_controls, population


@st.cache_resource
def context(run_dir, manifest_hash):
    return G1Run(run_dir, manifest_hash)


def show_case(ctx, case):
    record, original, raw = case
    sig = record["sigma"]
    name = f"{record['sample_id']} · {CHANGES[record['change']]} · G1 · σ {sig:.2f} · seed {record['seed']}"
    st.subheader(f"{name}　|　정답 {LABELS[record['label']]}")
    left, right = st.columns(2)
    left.image(png(original), caption="원본 이미지", width="stretch")
    right.image(png(raw), caption="두 분류기에 입력한 동일 이미지", width="stretch")
    cols = st.columns(2)
    for col, key, title in zip(cols, ("V0","G1"), ("잡음 학습 전","잡음 학습 후")):
        result = record["outputs"][key]
        col.write(f"**{title} · {key}**　원본 {LABELS[result['original']['prediction']]} → 선택 입력 **{LABELS[result['trial']['prediction']]}**")
    with st.expander("클래스별 결정 점수와 동일 입력 확인"):
        df = pd.DataFrame({"결함":[LABELS[n] for n in record["class_order"]]})
        for model, scores in record["outputs"].items():
            df[f"{model} 원본"] = scores["original"]["scores"]
            df[f"{model} 선택 입력"] = scores["trial"]["scores"]
        st.dataframe(df, hide_index=True, width="stretch")
        st.json({"raw_input_hash":record["raw_input_hash"], "processed_V0":record["outputs"]["V0"]["trial"]["processed_input_hash"],
                 "processed_G1":record["outputs"]["G1"]["trial"]["processed_input_hash"], "execution_id":record["execution_id"]})
    st.download_button("이 사례 ZIP · "+name, ctx.export(case),
                       f"case_{record['sample_id']}_{record['execution_id'][:16]}.zip", "application/zip",
                       key="g1_zip_"+record["execution_id"], on_click="ignore")


def workbench():
    manifest = DEFAULT_G1 / "manifest.json"
    if not manifest.is_file():
        status_path = DEFAULT_G1 / "test_status.json"
        state = read_json(status_path)["phase"] if status_path.exists() else "미시작"
        st.info(f"G1 평가 묶음 준비 중 · 시험 상태 {state}")
        return
    ctx = context(str(DEFAULT_G1), file_sha(manifest))
    summary = ctx.table("validation", "summary")
    base = summary.query("model_id=='V0' and sigma==0").iloc[0]
    evidence = read_json(ctx.checked("failure_conditions.json"))
    st.write(f"검증 원본 360장에서 기본 분류기는 **{base.correct}장**을 맞혔습니다. "
             f"잡음을 더하자 원본 정분류가 **{evidence['affected_originals']}장**에서 한 번 이상 오분류로 바뀌었습니다 "
             f"(9조건 합계 {evidence['trials']}회).")
    with st.expander("실패 조건과 분류기 재학습 설정"):
        st.write("**관측**　입력에 합성 가우시안 잡음을 더하면 일부 결함의 분류가 바뀌었습니다.")
        st.write("**가설**　원본에서만 학습한 마지막 분류기가 잡음이 섞인 영상의 특징을 충분히 구분하지 못할 수 있습니다.")
        st.write("**변경**　학습 원본 1,080장마다 원본과 세 잡음 강도(0.02, 0.05, 0.10)를 사용했습니다. "
                 "가중치는 원본 1/2, 잡음 각각 1/6이며 증강 seed는 301입니다. 같은 인코더, scaler와 입력 처리에서 마지막 분류기만 다시 학습했습니다.")
    validation = "통과" if ctx.seal["assessment"]["eligible"] else "미달"
    test = "완료" if ctx.status["evaluated"] else "미실행"
    adoption = "채택" if ctx.status["adopt"] else "미채택"
    st.write(f"**학습 완료**　·　검증 {validation}　·　최종 시험 {test}　·　**{adoption}**")
    selected = case_controls(ctx, "G1")
    if selected is not None:
        show_case(ctx, selected)
    population(ctx, "G1")
