"""SteelTrust G8 - novice-first Streamlit UX.

Retains sealed G1, G7 and their original models, uses read-only evidence
adapters instead of rewriting historical experiments.
"""
from __future__ import annotations
import io
import json
from html import escape
from base64 import b64encode
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import g8_core as core

CAPTIONS = {
    'fixed': '고친 오류',
    'new_error': '새로 생긴 오류',
    'both_wrong': '두 모델 모두 오답',
    'both_correct': '두 모델 모두 정답',
    'unchanged': '판정 유지',
}


def style():
    # The exact dohyeon branch uses a light operational workspace.
    # Styling is isolated from model/evaluation code in an independent CSS file.
    css = Path(__file__).with_name('g8_dohyeon.css').read_text(encoding='utf-8')
    st.markdown('<style>' + css + '</style>', unsafe_allow_html=True)


def heading(title: str, sub: str):
    st.markdown(
        '<header class="g8-page-header">'
        '<span class="g8-eyebrow">SteelTrust · AI Reliability</span>'
        f'<h1>{escape(title)}</h1><p>{escape(sub)}</p></header>',
        unsafe_allow_html=True,
    )


def sidebar_brand():
    # Original POSCO wordmark used by the public dohyeon branch.
    # A local copy (if provided) takes priority so the logo can work offline.
    local_logo = Path(__file__).resolve().parent.parent / 'assets' / 'posco-logo.png'
    if local_logo.is_file():
        uri = 'data:image/png;base64,' + b64encode(local_logo.read_bytes()).decode('ascii')
    else:
        uri = ('https://raw.githubusercontent.com/tmdgns104/'
               'posco_hackerton/dohyeon/SlagWatch/assets/posco-logo.png')
    st.markdown(
        '<div class="g8-sidebar-brand">'
        f'<div class="g8-posco-logo"><img src="{uri}" alt="POSCO 로고"></div>'
        '<div class="g8-brand-line"><span class="g8-symbol" aria-hidden="true">'
        '<i></i><i></i><i></i></span><div>'
        '<div class="g8-brand-title">SteelTrust</div>'
        '<div class="g8-brand-subtitle">검사 AI 신뢰성 검증</div>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )


def note():
    st.caption('공개 데이터 기반 연구 도구 · 판정은 실시간 공정 제어나 안전 보증에 사용할 수 없습니다.')


def goto(page):
    # A callback runs before the widgets are created on the next rerun.
    st.session_state['g8_nav'] = page

def percent(v: float):
    return f'{v*100:.1f}%'


def gateway():
    heading('SteelTrust', '산업용 AI 모델의 잘못된 판단을 찾고, 개선 전후를 같은 자료로 비교합니다.')
    st.markdown('''<section class="g8-hero">
      <span class="g8-eyebrow">모델의 신뢰성을 숫자로 검증합니다</span>
      <h2>AI를 개선했을 때, 정말 더 믿을 수 있을까요?</h2>
      <p>기존에 틀리던 판단이 얼마나 개선되었는지, 새로운 오류가 생기지는 않았는지
      동일한 시험 자료로 확인합니다. 처음 사용하신다면 아래 검사 분야를 선택하세요.</p>
      <div class="g8-flow"><span class="g8-step">① 분야 선택</span>
      <span class="g8-step-arrow">→</span><span class="g8-step">② 모델 비교</span>
      <span class="g8-step-arrow">→</span><span class="g8-step">③ 오류 확인</span></div>
    </section>''', unsafe_allow_html=True)
    a,b,c=st.columns(3)
    with a:
        with st.container(border=True):
            st.subheader('① 금속 표면')
            st.write('결함 분류 V0·V1·V2·G1과 정상·불량 K0·K1을 모두 비교합니다.')
            st.button('표면 모델 확인',use_container_width=True,on_click=goto,args=('금속 표면',))
    with b:
        with st.container(border=True):
            st.subheader('② 슬래그 진동')
            st.write('A1부터 G6 Deep까지 8종 전체를 개별 확인·동일 조건에서 비교합니다.')
            st.button('슬래그 모델 확인',use_container_width=True,on_click=goto,args=('슬래그',))
    with c:
        with st.container(border=True):
            st.subheader('③ AI 취약점 검사')
            st.write('Adaptive Red Team으로 AI가 실패하는 조건을 능동적으로 탐색합니다.')
            st.button('취약점 탐색하기',use_container_width=True,on_click=goto,args=('취약점 검사',))
    st.divider()
    st.subheader('지금 확인된 핵심 결과')
    try:
        sf = core.surface_stats(core.surface_comparison().query('sigma > 0'))
        bi = core.binary_stats(core.binary_comparison())
        _, sg = core.slag_comparison()
        g6 = sg.set_index('model').loc['G6-B ExtraTrees']
        m1,m2,m3=st.columns(3)
        m1.metric('강판 잡음 · 새 오류',f"{sf['new_errors']}회")
        m2.metric('K1 · 불량 미검출',f"{bi['K1']['fn']}회",delta=f"{bi['K1']['fn']-bi['K0']['fn']:+d}회",delta_color='inverse')
        m3.metric('G6-B · 슬래그 검출률',percent(g6.recall))
        st.caption('강판 수치는 잡음 9개 조건의 반복 평가 횟수이며, 서로 다른 이미지 수가 아닙니다.')
    except (ValueError,KeyError,OSError,ImportError) as exc:
        st.warning(f'일부 저장 검증 결과를 읽을 수 없습니다: {exc}')
    st.info('모델 개선이란 단순 정확도 상승이 아닙니다. **기존 오류 복구와 새로운 오류, 미검출과 오경보를 함께** 확인해야 합니다.')
    note()


def surface_view():
    heading('강판 결함 모델 비교','기존 V0와 개선 G1이 같은 NEU-CLS 이미지에서 어떻게 판단했는지 확인합니다.')
    try:
        t = core.surface_comparison()
    except (ValueError,OSError) as exc:
        st.error(f'강판 검증 기록을 불러오지 못했습니다: {exc}');return
    mode=st.radio('시험 조건', ['잡음 조건 (추천)', '원본 이미지', '전체'],horizontal=True)
    if mode.startswith('잡음'): t=t[t.sigma>0].copy()
    elif mode.startswith('원본'): t=t[t.sigma==0].copy()
    m=core.surface_stats(t)
    a,b,c,d=st.columns(4)
    a.metric('기존 V0 오류',f"{m['before_errors']}회")
    b.metric('개선 G1 오류',f"{m['after_errors']}회",delta=f"{m['after_errors']-m['before_errors']:+d}회",delta_color='inverse')
    c.metric('바로잡은 오류',f"{m['fixed']}회")
    d.metric('새로 생긴 오류',f"{m['new_errors']}회")
    st.bar_chart(pd.DataFrame({'오분류 횟수':[m['before_errors'],m['after_errors']]},index=['기존 V0','개선 G1']),height=190)
    st.caption(f"동일 시험 입력 {m['n']:,}회 판정 · 같은 이미지에 여러 교란을 적용한 반복 측정입니다. 이미지 {m['n']:,}장이라는 뜻이 아닙니다.")
    if m['new_errors']:
        st.warning('G1도 새 오답이 발생했습니다. 개선된 모델이라고 해서 모든 이미지에서 더 정확한 것은 아닙니다.')
    st.subheader('실제 오분류 사례 확인')
    kind=st.selectbox('어떤 사례를 보시겠어요?', ['fixed','new_error','both_wrong','both_correct'],format_func=lambda v:CAPTIONS[v])
    cases=t[t.change==kind]
    if cases.empty:
        st.info('선택한 조건에는 해당 사례가 없습니다.')
    else:
        pos=st.number_input('사례 번호',1,len(cases),1,step=1)
        row=cases.iloc[int(pos)-1]
        a,b=st.columns(2)
        a.write(f"**기존 V0** · 예측 `{row.prediction_before}`")
        b.write(f"**개선 G1** · 예측 `{row.prediction_after}`")
        st.caption(f"정답 {row.label} · 이미지 {row.sample_id} · 잡음 σ={row.sigma:g} · seed {int(row.seed)}")
        if st.button('이 사례의 원본·교란 이미지 확인',type='primary'):
            try:
                from .surface_g1_replay import G1Run
                from .surface_g1 import DEFAULT_G1
                run=G1Run(DEFAULT_G1)
                case=run.case(row.sample_id,'G1',float(row.sigma),int(row.seed))
                _,original,transformed=case
                ca,cb=st.columns(2)
                ca.image(original,caption='원본',clamp=True,use_container_width=True)
                cb.image(transformed,caption='시험 입력 (동일 입력으로 V0·G1 비교)',clamp=True,use_container_width=True)
                st.download_button('사례 재검증 ZIP 저장',data=run.export(case),file_name=f"surface_case_{row.sample_id}.zip",mime='application/zip')
            except (ValueError,KeyError,OSError,RuntimeError,ImportError) as exc:
                st.error('사례 실행에 필요한 원본 파일 또는 모델 확인이 필요합니다.')
                with st.expander('오류 상세'):st.code(str(exc))
    with st.expander('이전에 저장한 G1 오류 사례 다시 검사'):
        uploaded=st.file_uploader('SteelTrust가 내보낸 G1 사례 ZIP',type=['zip'],key='g8_case_zip')
        if uploaded is not None and st.button('사례 재검증 실행'):
            try:
                from .surface_g1_replay import G1Run
                from .surface_g1 import DEFAULT_G1
                from .demo_case import replay_upload
                run=G1Run(DEFAULT_G1)
                record,original,raw=replay_upload(run,uploaded.getvalue())
                st.success('입력 해시와 저장된 모델의 실제 추론 결과가 일치합니다.')
                c1,c2=st.columns(2)
                c1.image(original,caption='원본',clamp=True,use_container_width=True)
                c2.image(raw,caption='재현된 시험 입력',clamp=True,use_container_width=True)
                st.caption(f"원본 이미지 {record['sample_id']} · 동일 입력 비교 재현 완료")
            except (ValueError,KeyError,TypeError,OSError,RuntimeError,ImportError) as exc:
                st.error('사례 ZIP의 파일·실험 정보가 일치하지 않습니다.')
                with st.expander('오류 상세'):st.code(str(exc))
    with st.expander('상세 데이터·연구자용 기록'):
        view=t[['sample_id','sigma','seed','label','prediction_before','prediction_after','change']].copy()
        view.change=view.change.map(CAPTIONS)
        st.dataframe(view,hide_index=True,use_container_width=True,height=280)
        st.download_button('상세 결과 CSV',view.to_csv(index=False).encode('utf-8-sig'),'surface_comparison.csv','text/csv')
        st.caption('현재 저장된 G1 검증 결과를 조회하며, 새로 학습하거나 공식 시험 결과를 변경하지 않습니다.')
    note()


def binary_view():
    heading('정상·불량 모델 교체 검사','K0를 K1으로 바꿀 때 놓치는 불량이 늘었는지 확인합니다.')
    try:
        c=core.binary_comparison()
    except (ValueError,OSError) as exc:
        st.error(f'정상·불량 평가 기록을 확인할 수 없습니다: {exc}');return
    labels={'전체':'전체','원본':'clean','약한 잡음':'noise_weak','강한 잡음':'noise_strong','어두운 영상':'dark','밝은 영상':'bright'}
    condition=st.selectbox('촬영 조건',list(labels))
    if condition!='전체':c=c[c.condition==labels[condition]].copy()
    s=core.binary_stats(c)
    a,b,c1,d=st.columns(4)
    a.metric('K0 불량 미검출',s['K0']['fn'])
    b.metric('K1 불량 미검출',s['K1']['fn'],delta=f"{s['K1']['fn']-s['K0']['fn']:+d}건",delta_color='inverse')
    c1.metric('K0 오경보',s['K0']['fp'])
    d.metric('K1 오경보',s['K1']['fp'],delta=f"{s['K1']['fp']-s['K0']['fp']:+d}건",delta_color='inverse')
    st.bar_chart(pd.DataFrame({'오류 횟수':[s['K0']['fn'],s['K1']['fn'],s['K0']['fp'],s['K1']['fp']]},
                             index=['K0 미검출','K1 미검출','K0 오경보','K1 오경보']),height=190)
    st.error('현재 K1은 기존보다 정상 오경보를 줄였지만 불량 미검출이 늘어 모델 교체를 권장하지 않습니다.')
    st.caption(f"검증 데이터의 {len(c):,}회 판정 · 공식 시험 데이터가 아닌 validation 기록입니다.")
    with st.expander('교체로 새로 발생한 오류 확인'):
        options={'새로 놓친 불량':'new_miss','새 오경보':'new_false_alarm',
                 '복구한 미검출':'fixed_miss','해소한 오경보':'fixed_false_alarm'}
        kind=st.selectbox('오류 유형',list(options))
        rows=c[c[options[kind]]]
        st.write(f'**{len(rows)}건**')
        st.dataframe(rows[['sample_id','condition','seed','true_label_before','predicted_label_before','predicted_label_after']],hide_index=True,use_container_width=True)
        st.download_button('선택 사례 CSV',rows.to_csv(index=False).encode('utf-8-sig'),'binary_cases.csv','text/csv')
        st.caption('KSDD2 원본 이미지가 이 G7 패키지에 동봉되지 않아 원본 미리보기는 표시하지 않습니다.')
    note()


def slag_view():
    heading('슬래그 진동 모델 비교','동일한 출강 조건과 240개 진동 구간에서 세 모델의 오류를 비교합니다.')
    try:
        frames,stat=core.slag_comparison()
    except (ValueError,KeyError,ImportError,OSError) as exc:
        st.error(f'슬래그 비교 데이터 검증에 실패했습니다: {exc}');return
    a,b,c=st.columns(3)
    for col,name in zip((a,b,c),('G5-B SVM','G6-B ExtraTrees','G6 Deep MLP')):
        r=stat.set_index('model').loc[name]
        with col:
            st.markdown(f'**{name.replace(" ExtraTrees", "").replace(" MLP", "").replace(" SVM", "")}**')
            st.metric('슬래그 검출률',percent(r.recall))
            st.caption(f'오경보율 {percent(r.fpr)} · 미검출 {int(r.FN)}개')
    st.info('이 비교에서는 G6-B가 가장 유리합니다. 하지만 슬래그를 약 46% 놓쳐 **현장 자동 제어 용도는 불가**합니다.')
    st.divider()
    names=list(frames)
    base=st.selectbox('기존 모델',names,index=0)
    candidate=st.selectbox('비교할 모델',names,index=1)
    if base==candidate:
        st.warning('서로 다른 모델을 선택해 주세요.')
    else:
        from .slag_compare import paired
        diff=paired(frames[base],frames[candidate]);s=diff.change.value_counts()
        a,b,c=st.columns(3)
        a.metric('고친 오류',int(s.get('fixed',0)))
        b.metric('새로 생긴 오류',int(s.get('new_error',0)))
        c.metric('오류 순감소',int(s.get('fixed',0)-s.get('new_error',0)))
        st.caption('같은 16개 출강 조건의 B/S 240구간에서, 두 모델 모두 동일한 정답을 기준으로 판정했습니다.')
        with st.expander('오분류 구간 상세 보기 및 다운로드'):
            view=diff[['group','phase','start_s','true_slag_before','pred_slag_before','pred_slag_after','change']].copy()
            view.change=view.change.map(CAPTIONS)
            st.dataframe(view,hide_index=True,use_container_width=True,height=320)
            st.download_button('전체 구간 비교 CSV',view.to_csv(index=False).encode('utf-8-sig'),'slag_paired_errors.csv','text/csv')
    with st.expander('출강 조건별 결과'):
        group=st.selectbox('확인할 모델',names,index=1,key='g8_slag_group_model')
        t=frames[group]
        data=t.groupby('group',sort=True).apply(lambda z: pd.Series({'슬래그 검출률':z[z.true_slag==1].pred_slag.mean(),
                                                                   '오경보율':z[z.true_slag==0].pred_slag.mean()}),include_groups=False)
        st.bar_chart(data*100)
        st.caption('출강 조건 16개는 서로 다른 그룹으로 나누어 평가했습니다. 같은 데이터에 반복적으로 모델을 개선했으므로 새로운 현장 성능은 아닙니다.')
    with st.expander('별도 진동 CSV로 연구용 모델 실행'):
        st.caption('같은 출강의 정상 B 상태를 확인한 기준 CSV와 검사 대상 CSV가 모두 있어야 합니다. G5-B 및 G6-B만 실측 입력 추론을 지원합니다.')
        ref=st.file_uploader('확인된 B 기준 CSV · 최소 2.5초',type=['csv'],key='g8_ref')
        target=st.file_uploader('검사할 진동 CSV · 최소 0.5초',type=['csv'],key='g8_target')
        check=st.checkbox('B가 정상 단계임을 외부 기록으로 확인했고 연구 목적으로만 사용합니다.')
        if st.button('연구용 진동 분석 실행',disabled=not (ref and target and check)):
            try:
                from .slag_compare import external_inference
                output=external_inference(ref.getvalue(),target.getvalue())
                st.dataframe(output,hide_index=True,use_container_width=True)
                st.download_button('분석 결과 CSV',output.to_csv(index=False).encode('utf-8-sig'),'slag_inference.csv','text/csv')
            except (ValueError,RuntimeError,OSError,TypeError,KeyError) as exc:
                st.error('CSV 형식·기준 신호·모델 호환성을 확인해 주세요.')
                st.code(str(exc))
    st.caption('G3R은 E 구간 기준으로 다른 범위를 평가하므로 여기서 직접 순위를 매기지 않습니다.')
    note()


def adaptive_view():
    heading('AI 취약점 자동 탐색','SteelTrust가 기존 AI를 시험해 오분류가 나타나는 조건을 찾습니다.')
    st.info('검사 모델이 자동으로 개선되는 기능은 아닙니다. 우선 실패 조건을 능동적으로 찾고, 무작위 탐색과 비교합니다.')
    model=st.radio('검사할 강판 모델',['V0','G1'],horizontal=True)
    run_mode=st.radio('탐색 깊이',['빠른 시험','정밀 시험'],horizontal=True)
    budget=12 if run_mode=='빠른 시험' else 32
    st.caption(f'{budget}회 시험/이미지 · 검증 데이터만 사용 · 모델은 변경되지 않습니다.')
    if st.button('취약 조건 검사 시작',type='primary'):
        try:
            from .adaptive_redteam import run_surface
            from .surface_data import ROOT, RUN, CFG
            meta=pd.read_csv(RUN/'split_manifest.csv')
            val=meta[meta.split=='validation']
            groups=[g.sample_id.tolist() for _,g in val.groupby('label',sort=True)]
            samples=[g[0] for g in groups[:4] if g]
            if not samples:raise ValueError('검증 이미지가 없습니다.')
            with st.spinner('실제 AI 모델로 교란 시험을 실행하고 있습니다.'):
                result=run_surface(ROOT,samples,model_id=model,budget=budget,warmup=6,
                                   seed=2026,compare_random=True)
            st.session_state['g8_adaptive_result']=result
        except (ValueError,RuntimeError,OSError,KeyError,ImportError) as exc:
            st.error('탐색을 실행하지 못했습니다. 설치 환경과 기존 데이터 파일을 확인해 주세요.')
            with st.expander('오류 상세'):st.code(str(exc))
    result=st.session_state.get('g8_adaptive_result')
    if result is None:
        st.write('검사 시작을 누르면 **실제 실패가 발생하는 조건**과 **무작위 탐색 대비 발견 효율**을 보여줍니다.')
        return
    try:
        from .adaptive_redteam import summary_table
        df=summary_table(result)
        st.caption(f"최근 검사 결과 · 모델 {result['model_id']} · 실험 seed {result['rng_seed']} · 독립 시험이 아닌 개발용 검사")
        if df.empty:
            st.info('이번 시험에서 유효한 탐색 사례를 찾지 못했습니다.')
        else:
            st.subheader('시험 결과')
            report=df.groupby('strategy').agg(시험_이미지=('sample_id','size'),오류_발견_이미지=('failures',lambda x:int((x>0).sum()))).reset_index()
            st.dataframe(report,hide_index=True,use_container_width=True)
        with st.expander('전체 시험 수치와 재현 기록'):
            st.dataframe(df,hide_index=True,use_container_width=True)
            st.download_button('재현 가능한 탐색 기록 JSON',json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'),'adaptive_redteam_result.json','application/json')
    except (ValueError,KeyError,TypeError) as exc:
        st.error(f'탐색 결과 표시 오류: {exc}')
    st.warning('합성 교란이 실제로 결함의 의미를 보존하는지는 별도로 확인해야 합니다.')
    note()


def help_view():
    heading('도움말 · 고급 검증','처음에는 다른 메뉴만 사용하셔도 됩니다.')
    st.markdown('''**사용 순서:** ① 금속 표면 또는 슬래그 선택 → ② 전체 성능 차트 → ③ 모델 하나씩 보기 → ④ 두 모델 비교 → ⑤ 필요하면 CSV 저장.

**자주 묻는 질문**
- **정확도가 높으면 무조건 채택해도 되나요?** 아닙니다. 새 미검출·오경보와 독립 평가가 필요합니다.
- **SteelTrust 자체가 모델인가요?** 분류기와 검증 엔진을 결합한 플랫폼입니다. 능동 탐색에는 머신러닝을 사용합니다.
- **G6 Deep이 더 정확한가요?** 아니요. 현재 동일 평가에서는 G6-B가 더 유리합니다.
- **외부 모델을 바로 올려 검사하나요?** 이 간편 화면에서는 보안을 위해 임의 joblib 업로드를 지원하지 않습니다.
- **현장 도입해도 되나요?** 아니요. 현재 슬래그 모델은 실사용 승인 대상이 아닙니다.''')
    with st.expander('모델별 상세 등록부'):
        st.dataframe(pd.DataFrame(core.model_registry(),columns=['검증 분야','모델','기술','검증 상태']),hide_index=True,use_container_width=True)
    with st.expander('기존 G7 전체 연구 화면 열기'):
        st.write('기존 G7 화면과 코드는 보존했습니다. 아래 버튼을 눌러 이전 상세 메뉴를 볼 수 있습니다.')
        st.code('START_G7_LEGACY_WINDOWS.cmd')
        st.caption('간편 화면은 유지하고, 이전 메뉴를 별도 Windows 실행 파일로 열 수 있습니다.')
    with st.expander('파일 무결성 직접 확인'):
        if st.button('기존 연구 파일 82개 검사 실행'):
            try:
                from .adaptive_audit import audit
                a=audit(core.ROOT)
                if a['n_failed']:st.error(f"검증 실패 {a['n_failed']}건")
                else:st.success(f"검증 통과 {a['n_checked']}개")
            except (OSError,ValueError,KeyError) as exc:st.error(str(exc))
    note()


def app():
    style()
    pages=['시작하기','금속 표면','슬래그','취약점 검사','도움말']
    if st.session_state.get('g8_nav') not in pages:
        st.session_state['g8_nav'] = '시작하기'
    with st.sidebar:
        sidebar_brand()
        page = st.radio('무엇을 확인할까요?', pages, key='g8_nav')
        st.divider()
        st.caption('처음 사용하신다면 **시작하기**부터 보세요. 복잡한 CSV·해시는 상세 보기 안에 있습니다.')
    if page=='시작하기':gateway()
    elif page=='금속 표면':
        from .g8_model_matrix_ui import surface_screen
        surface_screen()
    elif page=='슬래그':
        from .g8_model_matrix_ui import slag_screen
        slag_screen()
    elif page=='취약점 검사':adaptive_view()
    else:help_view()
