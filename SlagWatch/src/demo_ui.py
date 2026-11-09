"""SteelTrust's four-page demonstration of a frozen NEU-CLS comparison."""
import io
from html import escape
from zipfile import ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .demo_case import replay_upload
from .demo_design import CHANGES_KO, MESSAGES, MODELS, STRENGTH, heading, image_uri, outcome_title, prediction_card
from .demo_results import aggregate, counts, criteria
from .surface_data import RUN
from .surface_detector import read_raw
from .surface_g1 import DEFAULT_G1, file_sha, read_json
from .surface_g1_replay import G1Run
from .surface_replay import LABELS

SPLITS = {"validation":"개선 모델 선택용 검증", "test":"모델 확정 후 최종 평가"}
F1_HELP = "macro F1은 여섯 결함 종류의 분류 성능을 같은 비중으로 반영한 점수입니다."


@st.cache_resource(scope="session", show_spinner="실험 기록을 확인하고 있습니다…")
def context(manifest_hash):
    return G1Run(DEFAULT_G1, expected_manifest=manifest_hash)


def current():
    return context(file_sha(DEFAULT_G1 / "manifest.json"))


def seeds_for(run, split, sigma):
    return [k for s,k in run.protocol["conditions"][split] if s == sigma]


def condition(run, split, sigma, seed):
    if not sigma:
        return "추가 잡음 없는 원본"
    number = seeds_for(run,split,sigma).index(seed) + 1
    return f"{STRENGTH[sigma]} · 패턴 {number}"


def scope_line(run, split, sigma, seed):
    n = run.man["n_originals"][split]
    st.caption(f"{SPLITS[split]}　·　기존 모델 V0 / 개선 모델 G1　·　{condition(run,split,sigma,seed)}　·　원본 {n}장, 이 조건 {n}회 판정")


def denominators(run, split):
    conditions = [(s,k) for s,k in run.protocol["conditions"][split] if s]
    n = run.man["n_originals"][split]
    strengths = len({s for s,_ in conditions})
    patterns = len(conditions) // strengths
    st.caption(f"{'최종 평가' if split == 'test' else '검증'} 이미지 {n}장에 잡음 강도 {strengths}개와 패턴 {patterns}개를 적용한 "
               f"총 {n * len(conditions):,}회 판정입니다. 한 강도에서는 같은 {n}장을 {n * patterns:,}회 판정했습니다.")


@st.cache_data(show_spinner=False)
def specimen_strip(run_id, _run):
    training = _run.meta[_run.meta.split == "train"]
    items = []
    for label, title in LABELS.items():
        row = training[training.label == label].sort_index().iloc[0]
        items.append(f'<figure><img alt="{title} 학습 원본" src="{image_uri(read_raw(row))}">'
                     f'<figcaption>{title}</figcaption></figure>')
    return "".join(items)


def overview():
    run = current()
    st.html(f'''<header class="steel-hero"><div><span class="steel-scope">강판 표면 결함 6종 분류 · 공개 NEU-CLS 데이터 · 합성 가우시안 잡음</span>
<h1>철강 표면 결함 AI의 오류를 찾고, 개선 효과를 확인합니다</h1>
<p>SteelTrust는 강판 표면의 결함 6종을 구분하는 AI를 시험합니다. 이미지에 잡음을 더해 오분류가 발생하는 사례를 찾고,
같은 입력에서 개선 모델이 기존 오류를 얼마나 고쳤는지 확인합니다. 개선 과정에서 새로 생긴 오류도 함께 보여줍니다.</p></div>
<div class="steel-specimens">{specimen_strip(run.id,run)}</div></header>''')
    data = counts(run, "test")
    clean = aggregate(data[data.sigma == 0])
    noise = aggregate(data[data.sigma > 0])
    n = run.man["n_originals"]["test"]
    reduction = (noise["V0_wrong"] - noise["G1_wrong"]) / noise["V0_wrong"] if noise["V0_wrong"] else None
    clean_delta = clean["G1_correct"] - clean["V0_correct"]
    clean_note = f"정답 {abs(clean_delta)}장 {'감소' if clean_delta < 0 else '증가'}" if clean_delta else "같은 정답 수"
    cards = [
        ("good","잡음 조건의 오분류", f"{noise['V0_wrong']} → {noise['G1_wrong']}<sub>회</sub>",
         f"잡음 조건 오분류 {reduction:.1%} 감소" if reduction is not None else "기존 오류 0회"),
        ("good","바로잡은 오류",f"{noise['fixed']}<sub>회</sub>","기존 오답 → 개선 정답"),
        ("caution","새로 발생한 오류",f"{noise['new_error']}<sub>회</sub>","기존 정답 → 개선 오답"),
        ("caution" if clean_delta<0 else "","원본 이미지의 정답",
         f"{clean['V0_correct']} → {clean['G1_correct']}<sub> / {n}장</sub>",clean_note),
    ]
    st.subheader("모델 확정 후 최종 평가")
    st.html('<div class="steel-cards">' + "".join(
        f'<article class="steel-card {tone}"><span>{label}</span><strong>{value}</strong><small>{note}</small></article>'
        for tone,label,value,note in cards) + '</div>')
    denominators(run,"test")
    with st.container(horizontal=True):
        if st.button("오분류 사례 찾아보기", type="primary"):
            st.switch_page(st.Page(search, url_path="search"))
        if st.button("개선 전후 결과 보기"):
            st.switch_page(st.Page(regression, url_path="compare"))
    with st.expander("어떤 오류에서 개선을 시작했나요?"):
        evidence = read_json(run.checked("failure_conditions.json"))
        discovered = pd.read_csv(run.checked("validation_discoveries.csv.gz"))
        ids = ["sample_id","sigma","seed"]
        paired = discovered[ids].merge(run.table("validation","pairs")[ids+["change"]],on=ids,how="left",validate="one_to_one")
        if paired.change.isna().any() or len(paired)!=evidence["trials"] or paired.sample_id.nunique()!=evidence["affected_originals"]:
            raise ValueError("검증 발견 목록과 사례별 결과가 일치하지 않습니다.")
        all_val = counts(run,"validation")
        all_noise = aggregate(all_val[all_val.sigma>0])
        st.markdown(f"**검증에서 발견한 사례**　기존 모델이 원본에서는 맞혔다가 잡음에서 틀린 경우는 "
                    f"{evidence['affected_originals']}장, {len(paired)}회입니다. 이 중 개선 모델이 {paired.change.eq('fixed').sum()}회를 바로잡았습니다.")
        st.write("잡음을 추가한 학습 이미지로 마지막 분류기를 다시 학습했습니다. 이미지 특징을 추출하는 인코더와 이미 학습된 scaler는 고정했습니다.")
        st.write(f"검증의 모든 잡음 판정을 포함하면 오분류는 {all_noise['V0_wrong']}회에서 {all_noise['G1_wrong']}회로 줄었고, "
                 f"전체에서 {all_noise['fixed']}회를 바로잡았습니다.")
    with st.expander("개선 모델의 채택 기준 확인"):
        for split in run.man["splits"]:
            st.markdown(f"**{SPLITS[split]}**")
            st.table(criteria(run,split), hide_index=True, border="horizontal")
        st.caption(f"이 실험의 결론은 개선 모델 G1 {'채택' if run.status['adopt'] else '미채택'}입니다. 검증 범위는 공개 데이터의 결함 6종 분류와 지정한 합성 잡음입니다.")


def controls(run, split, purpose, include_clean):
    key = f"{run.id}_V0_G1_{split}_{purpose}"
    strengths = sorted({s for s,_ in run.protocol["conditions"][split] if include_clean or s})
    left,right = st.columns(2)
    sigma = left.selectbox("입력 조건" if include_clean else "잡음 강도",strengths,
                           index=strengths.index(.02),format_func=STRENGTH.get,key=key+"_sigma",
                           help="이번 실험의 가우시안 잡음 표준편차는 약한 0.02, 중간 0.05, 강한 0.10입니다.")
    seeds = seeds_for(run,split,sigma)
    seed = right.selectbox("잡음 패턴",seeds,disabled=not sigma,
                           format_func=lambda k:f"패턴 {seeds.index(k)+1}" if sigma else "해당 없음",
                           help="같은 강도에서 잡음의 픽셀 배치를 바꿉니다. 각 패턴의 실제 seed는 "+", ".join(map(str,seeds))+"입니다.",
                           key=key+f"_{sigma}_seed")
    return sigma,seed,key+f"_{sigma}_{seed}"


def display_case(run, case, download=True):
    record,original,raw = case
    label = LABELS[record["label"]]
    outputs = record["outputs"]
    before,after = [LABELS[outputs[name]["trial"]["prediction"]] for name in MODELS]
    kind, sample = record["change"], escape(record["sample_id"])
    split,sigma,seed = record["split"],record["sigma"],record["seed"]
    title = outcome_title(label,before,after,kind)
    trial_caption = "두 모델에 입력한 같은 잡음 이미지" if sigma else "두 모델에 입력한 원본 이미지"
    pictures = "".join(f'<figure><figcaption>{name}</figcaption><img src="{image_uri(image)}" alt="{name} · {sample}"></figure>'
                       for image,name in ((original,"잡음을 추가하기 전" if sigma else "기록된 원본 이미지"),(raw,trial_caption)))
    verdicts = "".join(prediction_card(name,LABELS[output["trial"]["prediction"]],output["trial"]["prediction"]==record["label"])
                       for name,output in outputs.items())
    tone = "warn" if kind in ("new_error","both_wrong") else ""
    st.html(f'''<article class="steel-case"><h2>{escape(title)}</h2>
<p class="case-meta"><b>실제 결함 · {label}</b>　|　{sample} · {SPLITS[split]} {run.man["n_originals"][split]}장 · {condition(run,split,sigma,seed)}</p>
<div class="steel-case-grid"><div class="steel-visuals">{pictures}</div><div class="steel-preds">{verdicts}</div></div>
<p class="steel-outcome {tone}">{MESSAGES[kind]}</p></article>''')
    if download:
        st.download_button("이 사례 저장하기",run.export(case),f"G1_{sample}_{record['execution_id'][:16]}.zip",
                           "application/zip",key="demo_zip_"+record["execution_id"],on_click="ignore",type="primary")
    with st.expander("모델 점수와 재현 정보"):
        for name,output in outputs.items():
            st.write(f"{MODELS[name]}의 원본 판정 · {LABELS[output['original']['prediction']]}")
        scores = {"결함":[LABELS[k] for k in record["class_order"]]}
        for name,output in outputs.items():
            for part,desc in (("original","원본"),("trial","선택 입력")):
                scores[f"{MODELS[name]} · {desc}"] = output[part]["scores"]
        st.dataframe(pd.DataFrame(scores),hide_index=True,width="stretch")
        st.caption(f"클래스별 결정 점수입니다. 재현 시 점수의 허용 오차는 절댓값 {run.protocol['score_atol']}입니다.")
        st.json({"sigma":sigma,"seed":seed,"raw_input_hash":record["raw_input_hash"],"execution_id":record["execution_id"],
                 "processed_input_hash":{k:v["trial"]["processed_input_hash"] for k,v in outputs.items()},
                 "model_run_id":{k:v["model_run_id"] for k,v in outputs.items()}})


def choose_case(run, rows, key, sigma, seed, discoveries=None):
    if rows.empty:
        st.info("이 조건에 해당하는 사례가 없습니다. 다른 잡음 패턴이나 입력 조건을 선택해 주세요.")
        return
    rows = rows.sort_values("sample_id").set_index("sample_id")
    sample = st.selectbox("확인할 사례", rows.index, key=key+"_image",
                          format_func=lambda s:f"{s} · {LABELS[rows.loc[s,'label']]} · {CHANGES_KO[rows.loc[s,'change']]}")
    supplied = None
    if discoveries is not None:
        with ZipFile(run.checked("validation_discovered_inputs.zip")) as archive:
            supplied = np.load(io.BytesIO(archive.read(discoveries.loc[sample,"npy_member"])),allow_pickle=False)
    display_case(run,run.case(sample,"G1",sigma,seed,provided=supplied))


def search():
    run = current()
    heading("오분류 사례 찾기","기존 모델이 원본에서는 맞혔지만, 잡음을 더한 뒤 틀린 사례를 찾습니다.")
    sigma,seed,key = controls(run,"validation","discovery",False)
    scope_line(run,"validation",sigma,seed)
    saved = pd.read_csv(run.checked("validation_discoveries.csv.gz"))
    saved = saved[(saved.sigma==sigma)&(saved.seed==seed)].set_index("sample_id",verify_integrity=True)
    with st.container(horizontal=True):
        execute = st.button("선택한 조건으로 사례 찾기",type="primary",key=key+"_live")
        if st.button("저장된 탐색 결과 보기",key=key+"_saved"):
            st.session_state.pop(key+"_hits",None)
    if execute:
        bar = st.progress(0,text="원본과 잡음 이미지를 두 모델로 확인하고 있습니다…")
        result = run.live("G1",sigma,seed,lambda done,total:bar.progress(done/total,text=f"원본 {total}장 중 {done}장 확인"))
        ids = result.loc[result.induced,"sample_id"].tolist()
        if set(ids) != set(saved.index):
            raise ValueError("이번 실행과 저장된 탐색 결과의 사례 목록이 다릅니다.")
        st.session_state[key+"_hits"] = ids
        bar.empty()
    ids = st.session_state.get(key+"_hits",list(saved.index))
    st.subheader(f"{'이번 실행 결과' if key+'_hits' in st.session_state else '저장된 탐색 결과'} · {len(ids)}장")
    rows = run.pair_rows("validation","G1",sigma,seed)
    choose_case(run,rows[rows.sample_id.isin(ids)],key,sigma,seed,saved)


def overall(run, split):
    data = counts(run,split)
    table = pd.DataFrame([dict(입력=STRENGTH[s],**aggregate(g)) for s,g in data.groupby("sigma")])
    clean = table.iloc[0]
    noise = aggregate(data[data.sigma>0])
    st.subheader("입력 조건별 분류 성능")
    st.write(f"잡음 9조건 평균 macro F1은 **{noise['V0_f1']:.4f} → {noise['G1_f1']:.4f}**, "
             f"원본은 **{clean.V0_f1:.4f} → {clean.G1_f1:.4f}**입니다.")
    chart = go.Figure()
    for model,color in (("V0","#60bfff"),("G1","#f5ae50")):
        vals = table[model+"_f1"].tolist()
        chart.add_bar(name=MODELS[model],x=table["입력"].tolist(),y=vals,marker_color=color,
                      hovertemplate="%{x} · %{y:.4f}<extra>%{fullData.name}</extra>")
    chart.update_layout(barmode="group",yaxis={"range":[0,1],"dtick":.25,"title":"macro F1","fixedrange":True},
                        xaxis={"fixedrange":True},legend={"orientation":"h","y":1.13},height=285,
                        margin={"l":35,"r":10,"t":40,"b":25},bargap=.35,plot_bgcolor="#111f38",
                        paper_bgcolor="#111f38",font={"color":"#e7eefb"})
    st.plotly_chart(chart,key=f"f1_{run.id}_{split}",width="stretch",config={"displayModeBar":False})
    st.caption(F1_HELP+" 원본은 한 조건의 점수이고, 잡음 강도별 값은 세 패턴 점수의 평균입니다.")
    denom = run.man["n_originals"][split]
    display = table[["입력","V0_f1","G1_f1","fixed","new_error","both_wrong"]].copy()
    display.columns = ["입력 조건","V0 macro F1","G1 macro F1","바로잡은 오류","새로 발생한 오류","둘 다 오답"]
    for c in ("V0 macro F1","G1 macro F1"):
        display[c] = display[c].map(lambda v:f"{v:.4f}")
    st.table(display,hide_index=True,border="horizontal")
    st.caption(f"조건 하나의 분모는 원본 {denom}장입니다. 두 모델이 모두 틀린 잡음 판정은 {noise['both_wrong']}회입니다.")
    with st.expander("결함별 성능과 채택 기준"):
        classes = run.table(split,"classes").copy()
        classes["입력"] = classes.sigma.map(STRENGTH)
        classes["결함"] = classes.label.map(LABELS)
        classes["모델"] = classes.model_id.map(MODELS)
        stats = classes.groupby(["입력","결함","모델"],sort=False)[["recall","f1"]].mean().reset_index()
        st.dataframe(stats.rename(columns={"recall":"재현율","f1":"F1"}),hide_index=True)
        st.table(criteria(run,split),hide_index=True,border="horizontal")
    with st.container(horizontal=True):
        for name,title in (("summary","조건별 전체 결과 CSV"),("pairs","원본별 전후 비교 CSV")):
            st.download_button(title,run.table(split,name).to_csv(index=False).encode("utf-8-sig"),
                               f"G1_{split}_{name}.csv","text/csv",key=f"all_{run.id}_{split}_{name}",on_click="ignore")


def regression():
    run = current()
    heading("개선 전후 비교","같은 이미지에서 두 모델이 내린 판정과, 전체 평가에서 바로잡거나 새로 만든 오류를 확인합니다.")
    split = st.radio("비교할 데이터",run.man["splits"],index=run.man["splits"].index("test"),horizontal=True,
                     format_func=SPLITS.get,key="compare_split_"+run.id)
    denominators(run,split)
    summary,examples = st.tabs(["전체 평가","사례별 비교"])
    with summary:
        overall(run,split)
    with examples:
        sigma,seed,key = controls(run,split,"regression",True)
        kind = st.selectbox("판정 변화",list(CHANGES_KO),format_func=CHANGES_KO.get,key=key+"_change")
        scope_line(run,split,sigma,seed)
        rows = run.pair_rows(split,"G1",sigma,seed)
        choose_case(run,rows[rows.change==kind],key+kind,sigma,seed)


def replay():
    run = current()
    heading("저장한 사례 다시 확인","저장한 사례를 불러와 당시와 같은 입력으로 두 모델을 다시 실행합니다. 판정과 입력 정보가 저장된 기록과 일치하는지 확인할 수 있습니다.")
    mode = st.radio("사례 불러오기",["ZIP 업로드","동봉 사례"],horizontal=True)
    if mode=="ZIP 업로드":
        uploaded = st.file_uploader("저장한 G1 사례 ZIP 선택",type=["zip"],max_upload_size=1,key=run.id+"_upload")
        if uploaded is None:
            st.caption("‘이 사례 저장하기’로 내려받은 ZIP을 올리거나 동봉 사례를 선택해 주세요.")
            return
        data = uploaded.getvalue()
    else:
        kind = st.selectbox("동봉 사례",["fixed","new_error"],format_func=CHANGES_KO.get)
        data = (run.out/"examples"/(kind+".zip")).read_bytes()
    try:
        with st.spinner("저장한 입력으로 두 모델을 다시 실행하고 있습니다…"):
            case = replay_upload(run,data)
    except (ValueError,RuntimeError,OSError) as exc:
        st.error("이 실행본에서 저장한 G1 사례 ZIP을 다시 선택해 주세요. 선택한 기록을 재현할 수 없습니다.")
        with st.expander("오류 상세"):
            st.code(str(exc))
        return
    st.success("저장 당시와 같은 판정이 재현되었습니다.")
    st.caption("입력과 모델 정보, 결정 점수도 확인했습니다.")
    display_case(run,case,download=False)


def filters():
    from .surface_ui import POLICIES, case_controls, population, show_case, surface_context
    st.title("이전 표면 필터 실험")
    ctx = surface_context(file_sha(RUN/"manifest.json"))
    after = st.selectbox("이전 필터",["V1","V2"],format_func=lambda k:f"{k} · {POLICIES[k]}")
    st.write(f"검증 선택 · {ctx.seal['selected'] or '미채택'} / 최종 시험 · {'완료' if ctx.status['evaluated'] else '미실행'}")
    case = case_controls(ctx,after)
    if case is not None:
        show_case(ctx,case)
    population(ctx,after)


def slag():
    st.title("이전 슬래그 실험")
    st.caption("공개 주조 진동 SSFD를 사용한 이전 개발 평가입니다. 다른 슬래그 화면은 ‘이전 통합 앱’에도 있습니다.")
    with st.spinner("이전 슬래그 모델을 여는 중… 첫 실행에는 수치 연산 준비 시간이 필요합니다."):
        from .pair_ui import workbench
        workbench()
