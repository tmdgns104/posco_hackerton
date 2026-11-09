"""Read-only G3 model audit and opt-in experimental inference dashboard."""
from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from .slag_g3 import infer_with_reference
from .slag_g3_aug import OUT, RUN_NAME

ROOT=Path(__file__).resolve().parents[1]


def uploaded_windows(f):
    csv=pd.read_csv(f)
    if list(csv.columns)!=['time','x','y','z']:
        raise ValueError('CSV columns must be time,x,y,z')
    if len(csv)>3200*240 or len(csv)<3200 or len(csv)%3200:
        raise ValueError('Must contain 1..240 complete windows of 3200 samples each')
    v=csv[['x','y','z']].to_numpy(dtype=float)
    if not np.isfinite(v).all():raise ValueError('NaN/inf values are not permitted')
    return v.reshape(-1,3200,3)


@st.cache_resource
def full_model():
    return joblib.load(OUT/'models'/f'full_{RUN_NAME}.joblib')


@st.cache_data
def output_tables():
    return pd.read_csv(OUT/'summary.csv'),pd.read_csv(OUT/'predictions.csv'),pd.read_csv(OUT/'paired_vs_J.csv')


def workbench():
    st.title('SteelTrust G3 · 슬래그 진동 모델 개선 실험')
    st.caption('SSFD 16회 출강 · E(초기 무슬래그) 구간으로 센서 기준 보정 · G3R 노이즈 증강 모델')
    st.warning('연구 시제품: 독립 현장 검증 미완료. 슬래그 미검출이 많으므로 현장 차단·안전 판단에 사용 금지.')
    try:
        summary,pred,paired=output_tables()
    except (FileNotFoundError,ValueError) as exc:
        st.error('새 모델 평가 파일이 없습니다: '+str(exc))
        st.code('python -m src.slag_g3_aug train')
        return
    pane,review,deploy=st.tabs(['개발 평가','홀드아웃 구간 검토','외부 CSV 분석'])
    with pane:
        st.subheader('16개 출강 그룹 분리 · 400개 평가 구간')
        st.caption('E 첫 2.5초 80개 구간은 기준 보정 전용으로 제외. 나머지 400개는 그룹 단위 4-fold 평가. 모델군을 같은 데이터에서 탐색했으므로 독립 외부시험 결과는 아닙니다.')
        view=summary.assign(검출률=(summary.recall*100).round(1),오경보율=(summary.fpr*100).round(1))
        st.dataframe(view[['scenario','tp','fn','fp','tn','검출률','오경보율','roc_auc','average_precision']].rename(columns={'scenario':'조건','tp':'슬래그 검출','fn':'슬래그 미검출','fp':'오경보','tn':'정상 통과','roc_auc':'ROC-AUC','average_precision':'AP'}),hide_index=True)
        clean=paired
        y=clean.y_true_G3R
        st.metric('G3R 원본 검출률',f"{((clean.prediction_G3R.eq(1)&y.eq(1)).sum()/y.eq(1).sum())*100:.1f}%")
        st.metric('기존 J 원본 검출률 (같은 400구간)',f"{((clean.prediction_J.eq(1)&y.eq(1)).sum()/y.eq(1).sum())*100:.1f}%")
        st.info('0.05 잡음은 학습에 사용한 증강 계열이며, 0.08은 학습 강도 밖의 합성 잡음입니다. 오경보 및 미검출을 동시에 확인하세요.')
        st.download_button('G3R 예측 기록 CSV',pred.to_csv(index=False).encode('utf-8-sig'),'steeltrust_g3r_predictions.csv','text/csv')
    with review:
        phase=st.selectbox('상태',options=['S','B','E'],format_func={'S':'S · 슬래그 유동 중','B':'B · 슬래그 직전','E':'E · 초기 무슬래그'}.get)
        group=st.selectbox('출강 그룹(시험 제외 그룹의 모델 결과)',options=list(range(1,17)))
        condition=st.selectbox('시험 조건',options=list(summary.scenario))
        rows=pred[(pred.phase==phase)&(pred.group_id==group)&(pred.scenario==condition)].sort_values('window_idx')
        st.caption('선택한 그룹의 데이터로 학습하지 않은 fold 모델의 저장된 예측입니다.')
        if phase=='E':st.caption('E 전반 2.5초는 정상 기준 보정에 사용돼 평가에서 제외됩니다.')
        if len(rows):
            st.dataframe(rows[['window_idx','y_true','score','threshold','prediction','fold_id']],hide_index=True)
            st.line_chart(rows.set_index('window_idx')[['score','threshold']])
        else:st.info('해당 조건으로 점수화한 구간이 없습니다.')
    with deploy:
        st.subheader('사용자가 확보한 별도 진동 CSV 분석')
        st.warning('먼저 같은 출강 작업의 초기 무슬래그 2.5초를 확보해야 합니다. 이 모델은 초기 기준 신호 없이 사용할 수 없습니다.')
        a=st.file_uploader('초기 무슬래그 기준 CSV (time,x,y,z, 최소 16,000행)',type='csv',key='g3_ref')
        b=st.file_uploader('검사 CSV (time,x,y,z, 0.5초의 정수배)',type='csv',key='g3_target')
        certified=st.checkbox('기준 구간이 실제로 무슬래그 상태였음을 작업 기록으로 확인했습니다.',value=False)
        if st.button('실험용 분석 실행',disabled=not(bool(a) and bool(b) and certified)):
            try:
                ref=uploaded_windows(a); target=uploaded_windows(b)
                if len(ref)<5:raise ValueError('E reference needs at least five 0.5s windows')
                bundle=full_model();output=[];consecutive=0
                for i,v in enumerate(target):
                    r=infer_with_reference(ref,v,bundle)
                    consecutive=consecutive+1 if r.get('prediction')==1 else 0
                    output.append(dict(end_s=(i+1)*.5,**r,confirmed_2_windows=int(consecutive>=2)))
                t=pd.DataFrame(output)
                st.caption('이 모드는 전체 SSFD로 학습한 full 모델을 사용합니다. 입력이 학습 원본이면 새로운 정확도 평가로 간주할 수 없습니다.')
                st.dataframe(t,hide_index=True)
                st.download_button('출력 CSV',t.to_csv(index=False).encode('utf-8-sig'),'slag_g3r_prediction.csv')
                st.caption('2회 연속 검출은 별도 경보 확인용 후처리입니다. 이 옵션의 제철소 안전성은 검증되지 않았습니다.')
            except (ValueError,KeyError,RuntimeError,OSError) as exc:
                st.error(f'입력 확인 필요: {exc}')
