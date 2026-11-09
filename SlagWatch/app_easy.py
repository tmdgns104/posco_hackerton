"""SteelTrust G8 simple UI. Original sealed research code is unchanged."""
import streamlit as st
st.set_page_config(page_title='SteelTrust · 모델 신뢰성 확인',page_icon='🔍',layout='wide',initial_sidebar_state='expanded')
try:
    from src.g8_easy import app
    app()
except Exception as exc:
    st.error('SteelTrust 간편 화면을 열지 못했습니다. 기존 파일 또는 설치 환경을 확인해 주세요.')
    with st.expander('오류 상세 (진단용)'):
        st.exception(exc)
