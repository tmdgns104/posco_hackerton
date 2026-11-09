"""Unified home dashboard with separate model-level and platform-level health."""
from __future__ import annotations
import streamlit as st
from .slag_compare import summary, adoption_report
from .adaptive_audit import audit


def overview():
    st.title('SteelTrust G7 · 산업용 AI 신뢰성 검증 플랫폼')
    st.caption('표면 결함 분류 · 정상/불량 판정 · 슬래그 진동 · Adaptive Red Team')
    st.warning('공개 데이터 기반 연구용 프로그램입니다. 설비 자동 차단·인명 안전 판단에 직접 사용하지 마세요.')
    st.markdown('### 산업용 AI의 실패를 찾아내고, 개선 전후의 새로운 오류까지 검증합니다')
    st.write('SteelTrust는 개별 결함 탐지 모델을 대체하지 않습니다. **이미 학습된 AI를 시험하는 검증·개선 플랫폼**입니다. 모델의 점수와 SteelTrust 검사 엔진 자체의 효과는 구분해서 평가해야 합니다.')
    try:
        a=audit();m=summary().set_index('model');g=adoption_report()
    except (OSError,ValueError,KeyError) as exc:
        st.error('통합 결과 검증에 필요한 파일이 누락됐습니다: '+str(exc));return
    if a['n_failed']:
        st.error(f"기존 연구 파일 무결성 실패: {a['n_failed']} / {a['n_checked']}")
    else:
        st.success(f"기존 연구 파일 무결성 검사 통과: {a['n_checked']}개")
    col1,col2,col3=st.columns(3)
    col1.metric('표면 결함 G1 · 기존 채택 기준', '통과' if a['model_adoption']['G1'] else '보류')
    col2.metric('G6-B 슬래그 검출률', f"{m.loc['G6-B ExtraTrees','recall']:.2%}")
    col3.metric('G6 딥러닝 슬래그 검출률',f"{m.loc['G6 Deep MLP','recall']:.2%}")
    st.markdown('### 수행할 검증 분야')
    x,y=st.columns(2)
    with x:
        st.subheader('강판 표면 결함 · ResNet18 및 G1')
        st.write('6개 결함의 분류 오류, 잡음 강건성, 개선 결과에서 새로 생긴 오류를 분석합니다.')
        st.caption('왼쪽 메뉴: 표면 결함 소개 / 오분류 사례 찾기 / 개선 전후 비교')
    with y:
        st.subheader('슬래그 진동 · G5-B / G6-B / G6 Deep')
        st.write('동일 출강 그룹의 동일 240개 0.5초 구간을 맞춰 검출률·오경보·신규 오류를 비교합니다.')
        st.caption('왼쪽 메뉴: 슬래그 G5·G6 모델 검증')
    st.markdown('### 모델 채택 정책')
    st.table({
        '항목':['표면 결함 G1','슬래그 G6-B','슬래그 G6 Deep','산업 현장 자동 제어'],
        '현재 판정':['기존 NEU 연구 기준 통과','연구용 후보·현장 기준 미달','기존 G6-B보다 성능 열세','전체 모델 승인 불가']
    })
    st.markdown('### SteelTrust 자체의 개선 과제')
    st.write('Adaptive Red Team은 Random Forest 기반 능동 탐색을 수행하지만, 다른 데이터·다른 종류의 AI에서도 무작위 탐색보다 효율적인지는 별도로 입증해야 합니다. 입력 교란의 물리적 타당성 검증, 독립 환경 평가, 경보 보류 기능도 추가 연구가 필요합니다.')
    if not g['industrial_approved']:
        st.info('슬래그 모델은 독립 현장 데이터와 실무적인 B 기준 확인 절차가 부족하므로 모두 연구용으로 유지합니다.')
