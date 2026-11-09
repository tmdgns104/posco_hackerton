"""G8.2 model-by-model visual comparison. Only saved, validated evidence is displayed."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from . import g8_catalog as c
from . import g8_core as core

CHANGE_NAMES={'fixed':'복구된 오류','new_error':'새 오류','both_wrong':'둘 다 오답','both_correct':'둘 다 정답'}


def _head(title, desc):
    import html
    st.markdown('<div class="g8-page-header"><span class="g8-eyebrow">MODEL LAB · 모델 검증</span>'
                f'<h1>{html.escape(title)}</h1><p>{html.escape(desc)}</p></div>',unsafe_allow_html=True)


def _pct(n):return f'{n:.1f}%'


def _metrics(cols):
    columns=st.columns(len(cols))
    for col,(label,val) in zip(columns,cols):col.metric(label,val)


def _download(df, filename, label='상세 CSV 내려받기'):
    st.download_button(label,df.to_csv(index=False).encode('utf-8-sig'),file_name=filename,mime='text/csv')


def _surface_group(widget_scope="overview"):
    t_all=c.surface_records('validation')
    t_test=c.surface_records('test')
    scope=st.radio('비교 데이터', ['4개 모델 개발 검증 (Validation)','V0·G1 최종 시험 (Test)'],key=f'g8_surface_dataset_{widget_scope}',horizontal=True,
                   help='V1·V2에는 대응하는 최종 Test 예측 기록이 없어, 4개 비교는 Validation으로 한정됩니다.')
    test=scope.startswith('V0')
    if test:
        st.caption('**최종 시험 · V0와 G1만 비교**합니다. V1·V2를 이 시험 결과에 끼워 넣지 않습니다.')
        return t_test,'test'
    st.caption('**개발 검증 · 4종 전체**를 동일한 원본 시험 입력으로 비교합니다. 모델 선택에 사용한 Validation이므로 최종 독립 성능으로 해석하지 마세요.')
    st.caption('V1·V2의 내부 전처리는 V0와 다를 수 있지만, 원본 입력 해시와 정답 라벨은 일치합니다.')
    return t_all,'validation'


def surface_screen():
    _head('금속 표면 AI 검증','모델 하나씩 검사하고, 두 모델의 오류를 비교하며, 모든 모델의 성능을 한눈에 확인합니다.')
    task=st.radio('검사 종류',['표면 결함 6종 분류 · V0/V1/V2/G1','정상·불량 검사 · K0/K1'],horizontal=True)
    if task.startswith('정상'):
        binary_screen();return
    tabs=st.tabs(['전체 모델 비교','모델 개별 확인','두 모델 비교'])
    with tabs[0]:
        frames,scope=_surface_group('overview')
        df=c.surface_summary(frames)
        _metrics([('비교 모델',str(len(df))+'종'),('동일 시험 입력',f'{len(next(iter(frames.values()))):,}회'),
                  ('원본 종류','6종')])
        st.subheader('모델별 원본·잡음 성능')
        st.bar_chart(df.set_index('모델')[['원본 정확도(%)','잡음 Macro F1(%)']],height=300)
        st.caption('수치가 클수록 좋습니다. Macro F1은 잡음 강도·시드별 값을 평균했으며, 정확도와 다른 지표입니다.')
        curves=c.surface_conditions(frames).pivot(index='잡음 σ',columns='모델',values='정확도(%)')
        st.subheader('잡음이 강해질 때 정확도는 어떻게 변할까?')
        st.line_chart(curves,height=270)
        with st.expander('모델별 정확한 수치와 CSV'):
            st.dataframe(df.round(2),hide_index=True,use_container_width=True)
            _download(df,f'surface_all_{scope}.csv')
    with tabs[1]:
        frames,scope=_surface_group('single')
        name=st.selectbox('확인할 모델',list(frames),key=f'g8_surface_single_{scope}')
        t=frames[name];cond=c.surface_conditions({name:t})
        clean=t[t.sigma==0]; noisy=t[t.sigma>0]
        _metrics([('원본 정확도',_pct(100*clean.correct.mean())),('잡음 정확도',_pct(100*noisy.correct.mean())),
                  ('잡음에서 오분류',f'{int((~noisy.correct).sum())}회')])
        st.subheader(f'{name} · 잡음 강도별 정확도')
        st.line_chart(cond.set_index('잡음 σ')[['정확도(%)']],height=250)
        mode=st.selectbox('확인할 영상 상태',['원본','약한 잡음 σ=0.02','중간 잡음 σ=0.05','강한 잡음 σ=0.10'],key='g8_surface_case_filter')
        sigma=[0,.02,.05,.1][['원본','약한 잡음 σ=0.02','중간 잡음 σ=0.05','강한 잡음 σ=0.10'].index(mode)]
        subset=t[t.sigma==sigma]
        cm=pd.crosstab(subset.label,subset.prediction).rename_axis(index='정답',columns='모델 예측')
        with st.expander('6개 결함별 혼동 행렬 · 어떤 결함과 헷갈릴까?'):
            st.dataframe(cm,use_container_width=True)
        errors=subset[~subset.correct][['sample_id','label','prediction','sigma','seed']]
        with st.expander(f'오분류 사례 {len(errors)}건 · 보기/저장'):
            if errors.empty:st.success('이 조건에 저장된 오분류 사례가 없습니다.')
            else:st.dataframe(errors.head(150),hide_index=True,use_container_width=True)
            _download(errors,f'surface_{scope}_{name}_errors.csv')
        with st.expander('실제 강판 원본 이미지와 예측 보기'):
            rows=errors if not errors.empty else subset
            ids=rows.sample_id.drop_duplicates().astype(str).tolist()
            if ids:
                sample=st.selectbox('확인할 이미지',ids,key=f'g8_surface_sample_{scope}_{name}_{sigma}')
                selected=rows[rows.sample_id==sample].iloc[0]
                image_path=c.ROOT/'data'/'neu_cls'/f'{sample}.bmp'
                if image_path.is_file():
                    st.image(str(image_path),caption=f'원본: {sample} · 정답 {selected.label}',width=240)
                else:st.warning('원본 이미지가 패키지에 없습니다.')
                st.write(f"**{name} 저장 예측:** {selected.prediction} · "+('정답' if selected.prediction==selected.label else '오답'))
                if sigma>0:st.caption('예측값은 잡음 변형 입력으로 계산됐지만 위 사진은 원본입니다. 원본을 변형 영상으로 표시하지 않습니다.')
    with tabs[2]:
        frames,scope=_surface_group('paired')
        names=list(frames)
        a,b=st.columns(2)
        with a:before=st.selectbox('기존 모델',names,index=0,key='g8_surface_before')
        with b:after=st.selectbox('비교 모델',names,index=len(names)-1,key='g8_surface_after')
        if before==after:st.info('서로 다른 모델을 선택하세요.')
        else:
            pair=c.surface_pair(before,after,frames)
            changes=pair[pair.sigma>0]
            vc=changes.change.value_counts()
            _metrics([('복구된 오류',f"{vc.get('fixed',0)}회"),('새로 발생한 오류',f"{vc.get('new_error',0)}회"),
                      ('오류 순감소',f"{vc.get('fixed',0)-vc.get('new_error',0):+d}회")])
            st.caption('위 숫자는 잡음 조건의 반복 판정 횟수입니다. 같은 이미지가 여러 차례 포함될 수 있습니다.')
            st.subheader('교란 강도별 성능')
            chart=c.surface_conditions({before:frames[before],after:frames[after]}).pivot(index='잡음 σ',columns='모델',values='정확도(%)')
            st.line_chart(chart,height=250)
            typ=st.selectbox('사례 유형',['fixed','new_error','both_wrong'],format_func=lambda k:CHANGE_NAMES[k],key='g8_surface_pair_kind')
            view=pair[pair.change==typ][['sample_id','sigma','seed','label_before','prediction_before','prediction_after','change']].copy()
            view=view.rename(columns={'label_before':'정답','prediction_before':before,'prediction_after':after,'change':'구분'})
            view['구분']=view['구분'].map(CHANGE_NAMES)
            st.write(f'**선택한 사례 {len(view)}회**')
            with st.expander('사례 데이터 보기 · CSV 저장'):
                st.dataframe(view.head(150),hide_index=True,use_container_width=True)
                _download(view,f'surface_pair_{before}_{after}_{scope}.csv')
            if before=='V0' and after=='G1':
                with st.expander('G1 공식 사례 재추론 · 이미지 · ZIP 재현'):
                    eligible=pair[(pair.sigma>0)&pair.change.isin(['fixed','new_error'])]
                    if eligible.empty:
                        st.info('이 조건에는 재현할 교란 오류 사례가 없습니다.')
                    else:
                        show=eligible.iloc[st.selectbox('재검증 사례 번호',range(min(len(eligible),100)),format_func=lambda i:f'{i+1}번 · {eligible.iloc[i].sample_id} · σ={eligible.iloc[i].sigma}',key='g8_g1_replay_idx')]
                        if st.button('원본과 동일 조건 재추론',key='g8_g1_replay_run'):
                            try:
                                from .surface_g1_replay import G1Run
                                from .surface_g1 import DEFAULT_G1
                                run=G1Run(DEFAULT_G1)
                                case=run.case(show.sample_id,'G1',float(show.sigma),int(show.seed))
                                _,original,disturbed=case
                                aa,bb=st.columns(2)
                                aa.image(original,caption='원본 이미지',clamp=True,use_container_width=True)
                                bb.image(disturbed,caption='교란 입력',clamp=True,use_container_width=True)
                                st.download_button('검증 사례 ZIP 저장',run.export(case),file_name='G1_case_replay.zip',mime='application/zip')
                            except (OSError,RuntimeError,KeyError,ValueError,ImportError) as exc:
                                st.error('기존 G1 사례 추론을 확인하지 못했습니다.')
                                st.code(str(exc))
    st.caption('사용한 결과는 프로젝트에 동봉된 저장 평가 기록입니다. 이 화면은 모델을 새로 학습하거나 수정하지 않습니다.')


def binary_screen():
    t=core.binary_comparison()
    scope=st.selectbox('촬영 환경',['전체','원본','약한 잡음','강한 잡음','어두운 화면','밝은 화면'])
    key={'전체':None,'원본':'clean','약한 잡음':'noise_weak','강한 잡음':'noise_strong','어두운 화면':'dark','밝은 화면':'bright'}[scope]
    if key:t=t[t.condition==key].copy()
    stats=core.binary_stats(t)
    tabs=st.tabs(['K0·K1 전체 비교','모델 개별 확인','교체 시 새 오류'])
    with tabs[0]:
        df=pd.DataFrame([{'모델':n,'불량 검출률(%)':100*v['recall'],'정상 오경보율(%)':100*v['fpr'],
                          '미검출':v['fn'],'오경보':v['fp']} for n,v in stats.items()])
        st.bar_chart(df.set_index('모델')[['불량 검출률(%)','정상 오경보율(%)']],height=280)
        st.caption('불량 검출률은 높을수록, 정상 오경보율은 낮을수록 좋습니다. K0/K1은 같은 Validation 기록을 사용합니다.')
        with st.expander('모든 지표 및 CSV'):
            st.dataframe(df.round(2),hide_index=True,use_container_width=True)
            _download(df,'binary_summary.csv')
    with tabs[1]:
        name=st.selectbox('확인할 모델',['K0','K1'],key='g8_binary_individual')
        m=stats[name]
        _metrics([('불량 검출률',_pct(100*m['recall'])),('미검출',f"{m['fn']}건"),('오경보',f"{m['fp']}건")])
        pred_field='predicted_label_before' if name=='K0' else 'predicted_label_after'
        cm=pd.crosstab(t.true_label_before,t[pred_field]).rename_axis(index='실제',columns='예측')
        st.subheader('정상·불량 혼동 행렬')
        st.dataframe(cm,use_container_width=True)
        errors=t[t[pred_field]!=t.true_label_before]
        with st.expander('오류 사례와 CSV 저장'):
            view=errors[['sample_id','condition','seed','true_label_before',pred_field]]
            st.dataframe(view.head(150),hide_index=True,use_container_width=True)
            _download(view,f'binary_{name}_errors.csv')
    with tabs[2]:
        fixed=int(t.change.eq('fixed').sum());new=int(t.change.eq('new_error').sum())
        _metrics([('복구된 오류',f'{fixed}건'),('새로운 오류',f'{new}건'),('순감소',f'{fixed-new:+d}건')])
        st.warning('K1은 오경보가 줄었지만 불량 미검출은 증가했습니다. 현재 자동 교체 후보가 아닙니다.')
        with st.expander('신규 미검출·오경보 유형 확인'):
            kinds=['new_miss','new_false_alarm','fixed_miss','fixed_false_alarm']
            label={'new_miss':'새로 놓친 불량','new_false_alarm':'새 오경보','fixed_miss':'복구한 미검출','fixed_false_alarm':'해소한 오경보'}
            typ=st.selectbox('분석 유형',kinds,format_func=lambda k:label[k])
            view=t[t[typ]][['sample_id','condition','seed','true_label_before','predicted_label_before','predicted_label_after']]
            st.write(f'{len(view)}건')
            st.dataframe(view.head(150),hide_index=True,use_container_width=True)
            _download(view,f'binary_{typ}.csv')
    st.caption('KSDD2 원본 사진은 배포본에 포함되지 않아 저장된 검증 기록만 표시합니다.')


def _slag_chart(frames):
    rows=[]
    for name,t in frames.items():
        m=c.slag_measures(t)
        rows.append({'모델':name,'슬래그 검출률(%)':m['슬래그 검출률(%)'],'오경보율(%)':m['오경보율(%)']})
    st.bar_chart(pd.DataFrame(rows).set_index('모델'),height=245)
    st.caption('검출률은 높을수록 좋고, 오경보율은 낮을수록 좋습니다. 두 지표를 함께 보세요.')


def slag_screen():
    _head('슬래그 진동 AI 검증','A1부터 G6 Deep까지 8종을 개별 확인하고, 같은 진동 구간을 사용한 모델끼리 비교합니다.')
    try:
        cohorts=c.slag_records()
    except (ValueError,OSError,KeyError,ImportError) as e:
        st.error('슬래그 저장 평가 자료가 누락되었거나 내용이 일치하지 않습니다.')
        with st.expander('검증 오류 상세'):st.code(str(e))
        return
    tabs=st.tabs(['전체 8종 한눈에','모델 개별 확인','두 모델 비교'])
    with tabs[0]:
        st.info('**8종을 모두 보여주되, 조건이 다른 평가를 한 순위로 합치지 않습니다.** J는 서로 다른 두 평가 그룹에 중복 표시됩니다.')
        df=c.slag_summary(cohorts)
        _metrics([('등록 모델','8종'),('비교 가능 그룹','3개'),('검증 출강 조건','16개')])
        for cohort,frames in cohorts.items():
            st.subheader(cohort)
            _slag_chart(frames)
        with st.expander('전체 8종 모델 목록 · 세부 평가표'):
            st.dataframe(df.round(3),hide_index=True,use_container_width=True)
            _download(df,'slag_all_models_by_protocol.csv')
        st.warning('현재 모든 슬래그 모델은 실제 제철소 자동 제어용으로 승인되지 않았습니다.')
    with tabs[1]:
        model=st.selectbox('검사할 모델',c.SLAG_MODELS)
        candidates=[k for k,values in cohorts.items() if model in values]
        cohort=st.selectbox('평가 기준',candidates) if len(candidates)>1 else candidates[0]
        t=cohorts[cohort][model];m=c.slag_measures(t)
        st.caption(f'{c.SLAG_TECH[model]} · {cohort} · 저장된 그룹 분리 예측')
        _metrics([('슬래그 검출률',_pct(m['슬래그 검출률(%)'])),('오경보율',_pct(m['오경보율(%)'])),
                  ('미검출',f"{m['FN']}/{m['TP']+m['FN']}구간"),('ROC-AUC',f"{m['ROC-AUC']:.3f}")])
        st.subheader('출강 그룹별 성능')
        group_rows=[]
        for group,z in t.groupby('group'):
            k=c.slag_measures(z)
            group_rows.append({'출강 조건':str(group),'검출률(%)':k['슬래그 검출률(%)'],'오경보율(%)':k['오경보율(%)']})
        st.bar_chart(pd.DataFrame(group_rows).set_index('출강 조건'),height=270)
        st.subheader('정답과 AI 판정')
        confusion=pd.DataFrame({'AI: 비슬래그':[m['TN'],m['FN']], 'AI: 슬래그':[m['FP'],m['TP']]},index=['실제 비슬래그','실제 슬래그'])
        st.dataframe(confusion,use_container_width=True)
        with st.expander('개별 진동 구간·오분류 기록 다운로드'):
            view=t[t.truth!=t.pred].sort_values(['group','phase','window'])
            st.dataframe(view.head(150),hide_index=True,use_container_width=True)
            _download(view,f'slag_{model.replace(" ","_")}_errors.csv')
    with tabs[2]:
        cohort=st.selectbox('동일 조건으로 비교할 평가 그룹',list(cohorts),key='g8_slag_pair_cohort')
        names=list(cohorts[cohort]);l,r=st.columns(2)
        with l:before=st.selectbox('기존 모델',names,key='g8_slag_pair_before')
        with r:after=st.selectbox('비교 모델',names,index=len(names)-1,key='g8_slag_pair_after')
        st.caption('이 그룹에 속한 모델만 선택할 수 있습니다. 다른 그룹의 모델은 입력 조건·분모가 달라 1:1 비교할 수 없습니다.')
        if before==after:st.info('서로 다른 두 모델을 선택해 주세요.')
        else:
            p=c.slag_pair(cohort,before,after,cohorts)
            vc=p.change.value_counts()
            _metrics([('복구된 오류',f"{vc.get('fixed',0)}구간"),('새 오류',f"{vc.get('new_error',0)}구간"),
                      ('오류 순감소',f"{vc.get('fixed',0)-vc.get('new_error',0):+d}구간")])
            _slag_chart({before:cohorts[cohort][before],after:cohorts[cohort][after]})
            typ=st.selectbox('변화 유형',['fixed','new_error','both_wrong'],format_func=lambda k:CHANGE_NAMES[k])
            view=p[p.change==typ][['group','phase','window','truth_before','pred_before','pred_after','change']].copy()
            view.change=view.change.map(CHANGE_NAMES)
            with st.expander(f'사례 확인 · {len(view)}구간'):
                st.dataframe(view.head(150),hide_index=True,use_container_width=True)
                _download(view,f'slag_pair_{before}_{after}.csv')
    with st.expander('연구용 진동 CSV 직접 추론 · G5-B / G6-B만 지원'):
        st.caption('두 CSV는 동일 출강의 진동이어야 하며, B 기준 단계가 외부 기록으로 확인되어야 합니다. G6 Deep의 교차검증 체크포인트는 이 입력에 사용하지 않습니다.')
        ref=st.file_uploader('확인된 B 단계 기준 CSV (2.5초 이상)',type=['csv'],key='g8_matrix_ref')
        target=st.file_uploader('검사 대상 CSV (0.5초 이상)',type=['csv'],key='g8_matrix_target')
        check=st.checkbox('기준 진동이 B 단계임을 확인했으며 연구 목적으로만 실행합니다.',key='g8_matrix_confirm')
        if st.button('G5-B·G6-B 추론 실행',disabled=not(ref and target and check),type='primary'):
            try:
                from .slag_compare import external_inference
                result=external_inference(ref.getvalue(),target.getvalue())
                st.dataframe(result,hide_index=True,use_container_width=True)
                _download(result,'slag_external_research.csv','추론 결과 다운로드')
            except (ValueError,RuntimeError,OSError,TypeError,KeyError) as exc:
                st.error('파일 길이·B 기준 데이터·모델 호환성을 확인하세요.')
                st.code(str(exc))
    st.caption('모든 평가는 공개 SSFD의 출강 16조건에 대한 기존 연구 결과이며, 별도 제철소의 현장 성능을 의미하지 않습니다.')
