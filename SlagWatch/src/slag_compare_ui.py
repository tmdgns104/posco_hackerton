"""Streamlit G7 comparison of G5-B, G6-B, and G6 deep with strict sample matching."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from . import slag_compare as eng


@st.cache_data(show_spinner=False)
def cached_tables():
    return eng.aligned(),eng.summary(),eng.group_table(eng.aligned()),eng.full_report()


def metric_table(t):
    cols=['model','TP','FN','FP','TN','recall','fpr','macro_f1','roc_auc','pr_auc']
    return t[cols].rename(columns={'model':'모델','TP':'S 검출','FN':'S 미검출','FP':'B 오경보',
               'TN':'B 정상 통과','recall':'S 검출률','fpr':'B 오경보율',
               'macro_f1':'Macro F1','roc_auc':'ROC-AUC','pr_auc':'PR-AUC'})


def workbench():
    st.title('SteelTrust G7 · 슬래그 AI 모델 신뢰성 비교')
    st.caption('G5-B SVM · G6-B ExtraTrees · G6 딥러닝(Residual MLP) / 그룹 홀드아웃·동일 시험 입력')
    st.warning('**연구용 검증** · 전 모델 현장 제어 승인 불가. B/S 모델은 같은 출강에서 독립적으로 확인된 B 상태 진동 2.5초가 필요합니다. 데이터는 16개 출강 조건만 포함합니다.')
    try:
        frames,stats,group,report=cached_tables()
    except (OSError,ValueError,KeyError) as ex:
        st.error(f'G6 비교 자료의 무결성/표본 확인 실패: {ex}')
        st.stop()
    sm=stats.set_index('model');g6=sm.loc['G6-B ExtraTrees'];deep=sm.loc['G6 Deep MLP']
    a,b,c,d=st.columns(4)
    a.metric('G6-B 슬래그 검출률',f"{g6.recall:.2%}")
    b.metric('G6-B B 오경보율',f"{g6.fpr:.2%}")
    c.metric('G6 Deep 슬래그 검출률',f"{deep.recall:.2%}",delta=f"{(deep.recall-g6.recall)*100:+.2f}%p",delta_color='inverse')
    d.metric('G6 Deep B 오경보율',f"{deep.fpr:.2%}",delta=f"{(deep.fpr-g6.fpr)*100:+.2f}%p",delta_color='inverse')
    t1,t2,t3,t4,t5,t6=st.tabs(['동일 구간 모델 비교','출강 그룹별 분석','오류 복구·새 오류','별도 진동 CSV 분석','모델 전체 등록부','무결성·실험 기록'])
    with t1:
        st.subheader('SSFD 동일 240개 B/S 구간')
        st.caption('16개 출강 그룹 × (B 평가 5창 + S 평가 10창) · 구간당 0.5초 · 처음 B 2.5초는 기준 신호이므로 평가에서 제외')
        d=metric_table(stats).copy()
        for k in ['S 검출률','B 오경보율','Macro F1','ROC-AUC','PR-AUC']:
            d[k]=d[k].map(lambda n:f'{n*100:.2f}%' if pd.notna(n) else '—')
        st.dataframe(d,hide_index=True,width='stretch')
        chart=stats.set_index('model')[['recall','fpr']].rename(columns={'recall':'슬래그 검출률','fpr':'오경보율'})*100
        st.bar_chart(chart,x_label='모델',y_label='비율 (%)')
        st.info('**현재 비교에서 G6-B가 G6 Deep보다 우세**합니다. 단, 같은 16개 출강 기록을 여러 개발 단계에서 반복 이용했기 때문에 외부 현장 성능이나 통계적 확정 성능으로 볼 수 없습니다.')
        st.caption('B/S 2분류 기준입니다. 기존 G3R은 E 기준 400개 평가 구간이므로 이 그래프와 직접 비교하지 않습니다.')
        if st.button('별도 G3R 평가 보기',key='g7_g3_info'):
            st.session_state['g7_g3_view']=True
        if st.session_state.get('g7_g3_view'):
            other=eng.ROOT/'experiments/slag_g3_aug/summary.csv'
            if other.exists():
                tab=pd.read_csv(other)
                st.dataframe(tab[tab.scenario=='clean'],hide_index=True)
                st.caption('G3R의 출강 초기 E 기준 보정은 G6B의 B 기준 보정과 서로 다릅니다.')
            else:st.caption('기존 G3R 저장 결과가 없습니다.')
        st.download_button('모델 비교표 CSV',stats.to_csv(index=False).encode('utf-8-sig'),'steeltrust_g7_model_comparison.csv','text/csv')
    with t2:
        st.subheader('모델별 출강 그룹 변동')
        model=st.selectbox('모델',list(frames),index=1,key='g7_group_model')
        part=group[group.model==model].sort_values('group').copy()
        st.line_chart(part.set_index('group')[['recall','fpr']]*100,x_label='출강 그룹',y_label='비율 (%)')
        dt=part[['group','TP','FN','FP','TN','recall','fpr','roc_auc']].copy()
        dt[['recall','fpr','roc_auc']]=dt[['recall','fpr','roc_auc']].round(4)
        st.dataframe(dt,hide_index=True,width='stretch')
        st.caption('각 그룹은 독립 출강 조건이지만 같은 공장·데이터셋의 기록입니다. 그룹별 점수는 표본 수가 작으므로 변동이 큽니다.')
    with t3:
        old=st.selectbox('비교 기준 모델',list(frames),index=1,key='g7_compare_old')
        new=st.selectbox('검사 모델',list(frames),index=2,key='g7_compare_new')
        changes=eng.paired(frames[old],frames[new]);n=changes.change.value_counts()
        fixed=int(n.get('fixed',0));regressed=int(n.get('new_error',0))
        l,r=st.columns(2)
        l.metric('기존 오류를 새로 수정',f'{fixed}건')
        r.metric('개선으로 새로 발생한 오류',f'{regressed}건')
        if old==new:st.info('같은 모델끼리 비교하고 있습니다. 새 오류 수는 0입니다.')
        else:
            st.json(eng.bootstrap_delta(frames[old],frames[new]))
            st.caption('95% 구간은 출강 그룹 단위 부트스트랩이며, 여러 차례 모델을 바꿔가며 분석한 개발 데이터이므로 확증적 신뢰구간이 아닙니다.')
        c=changes.copy()
        c['오류 변화']=c.change.map({'fixed':'복구','new_error':'신규 오분류','unchanged':'변화 없음'})
        choice=st.selectbox('표시할 사례',['복구','신규 오분류','변화 없음','전체'],key='g7_err_filter')
        if choice!='전체':c=c[c['오류 변화']==choice]
        fields=['group','phase','start_s','true_slag_before','score_before','pred_slag_before','score_after','pred_slag_after','오류 변화']
        st.dataframe(c[fields],hide_index=True,height=320,width='stretch')
        st.download_button('동일 구간 전체 비교 CSV',changes.to_csv(index=False).encode('utf-8-sig'),'g7_paired_error_changes.csv','text/csv')
        st.subheader('N개 연속 경보가 있을 때의 출강 기록 단위 결과')
        streak=st.slider('필요한 연속 S 구간 수',1,4,2,key='g7_streak')
        erows=[]
        for name,t in frames.items():
            ev=eng.events(t,streak)
            erows.append({'모델':name,'S 기록 중 경보 발생':f"{ev[(ev.phase=='S')& (ev.alarm==1)].shape[0]}/16",
                          'B 기록에서 잘못 경보':f"{ev[(ev.phase=='B')& (ev.alarm==1)].shape[0]}/16"})
        st.table(pd.DataFrame(erows))
        st.caption('B/S는 별도 5초 파일입니다. 이 숫자는 실제 출강 중 최초 감지 시간이나 연속 공정 전체 경보율이 아닙니다.')
    with t4:
        st.subheader('실측 CSV로 연구용 비교 추론')
        st.caption('이 기능은 SSFD 전체 데이터로 학습된 G5-B / G6-B 모델만 사용합니다. G6 딥러닝의 20개 파일은 홀드아웃 폴드 체크포인트이므로 새 현장 데이터용 모델로 자동 사용하지 않습니다.')
        st.warning('업로드하는 B 기준은 작업 기록으로 **별도로 확인된 슬래그 미유입 상태**여야 합니다. SteelTrust가 B임을 스스로 판정할 수는 없습니다.')
        ref=st.file_uploader('같은 출강의 확인된 B 구간 CSV (time,x,y,z · 2.5초 이상)',type='csv',key='g7_b_ref')
        target=st.file_uploader('검사할 진동 CSV (time,x,y,z · 0.5초 이상)',type='csv',key='g7_b_target')
        confirm=st.checkbox('B 기준이 작업 기록 등으로 확인됐고, 실험 목적에만 사용함을 이해했습니다.',value=False,key='g7_b_certified')
        if st.button('G5-B / G6-B 추론 실행',disabled=not(bool(ref and target and confirm)),type='primary'):
            try:
                out=eng.external_inference(ref.getvalue(),target.getvalue())
                st.dataframe(out,hide_index=True,width='stretch')
                st.line_chart(out.set_index('start_s')[['G5B_score','G6B_score']],y_label='슬래그 점수',x_label='기록 시작 후 초')
                st.download_button('CSV 내보내기',out.to_csv(index=False).encode('utf-8-sig'),'g7_b_reference_predictions.csv','text/csv')
                st.caption('정답 CSV가 없으므로 실측 업로드에 대해 정확도·검출률을 산출하지 않습니다.')
            except (ValueError,TypeError,OSError,RuntimeError,KeyError) as exc:
                st.error(f'추론 입력 또는 모델 호환성 확인 필요: {exc}')
    with t5:
        st.subheader('SteelTrust의 검사 대상 모델과 시험 분야')
        registry=[
            ('강판 표면 결함 6종','V0 / V1 / V2 / G1','ResNet18 특징 + 분류기','기존 연구 탭에서 검증','독립 공장 검증 없음'),
            ('정상·불량','K0 / K1','이진 표면 검사','기존 기록 존재','K1 불량 미검출 증가로 채택 보류'),
            ('슬래그 3축 진동','A1 / J / C0 / C1','기존 특징 모델 및 CNN','기존 슬래그 탭에서 검증','채택 기준 미달'),
            ('슬래그 E기준 진동','G3R','ExtraTrees, E기준','별도 400구간 평가','B2와 직접 비교 불가'),
            ('슬래그 B기준 진동','G5-B / G6-B','SVM / ExtraTrees','동일 240구간 비교','둘 다 연구 후보'),
            ('슬래그 B기준 딥러닝','G6 Deep','Residual MLP','동일 240구간 비교','G6-B보다 열세·채택 보류'),
        ]
        st.dataframe(pd.DataFrame(registry,columns=['분야','모델','기술','평가 방식','현재 판단']),hide_index=True,width='stretch')
        st.subheader('SteelTrust 검증 엔진 자체는 어떤 기술인가?')
        st.markdown('''- **기존 오류 탐색:** 사전 정의 교란에 의한 통계·규칙 시험\n- **Adaptive Red Team:** 과거 시험 결과를 학습하는 **Random Forest 기반 대리모델**로 시험 조건 선택\n- **모델 개선:** 이미지 분야에서 고정 CNN 특징과 재학습된 분류기 활용\n- **검증·채택 판단:** 독립 기준 및 미탐·오탐 변화로 판단하는 통계·규칙 기반 검사''')
        st.info('SteelTrust는 **신뢰성 검증 프레임워크**이며, 별도의 단일 범용 딥러닝 모델 하나로 작동하는 것은 아닙니다. 현재 Adaptive Red Team은 딥러닝이 아니라 머신러닝(Random Forest) 기반입니다.')
        st.caption('범용성을 입증하려면 모델별 입력·정답·평가 방식을 표준화하고, 무작위 탐색 대비 오류 발견 효율과 새로운 환경에서의 개선 효과를 별도로 검증해야 합니다.')
    with t6:
        st.subheader('학습 자료·평가 결과 무결성')
        st.dataframe(pd.DataFrame(report['integrity']),hide_index=True)
        st.json(report['adoption'])
        st.download_button('검증 보고서 JSON',json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8'),'g7_reliability_report.json','application/json')
        st.caption('G6 Deep 체크포인트는 4개 폴드 × 5개 시드 총 20개이며, **완전 학습 후 실측 추론용 최종 모델은 아닙니다.**')
        st.caption('학습 데이터 원본은 기존 프로젝트의 data/raw/ 아래에, 재학습 코드는 training/slag_g6/ 에 보관되어 있습니다.')
