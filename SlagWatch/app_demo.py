from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SteelTrust · 강판 표면 결함", layout="wide")
try:
    from src import demo_ui as ui
    from src import adaptive_ui
    from src import slag_compare_ui
    from src import g7_home_ui
    from src.demo_design import apply_style, brand

    primary = [
        st.Page(g7_home_ui.overview, title="통합 관제", default=True, url_path="home"),
        st.Page(ui.overview, title="표면 결함 소개", url_path="surface"),
        st.Page(ui.search, title="오분류 사례 찾기", url_path="search"),
        st.Page(adaptive_ui.workbench, title="Adaptive Red Team", url_path="redteam"),
        st.Page(slag_compare_ui.workbench, title="슬래그 G5·G6 모델 검증", url_path="slag-g6"),
        st.Page(ui.regression, title="개선 전후 비교", url_path="compare"),
        st.Page(ui.replay, title="저장한 사례 다시 확인", url_path="replay"),
    ]
    previous = [
        st.Page(ui.filters, title="이전 표면 필터 실험", url_path="filters"),
        st.Page(ui.slag, title="이전 슬래그 실험", url_path="slag"),
        st.Page(Path(__file__).with_name("app.py"), title="이전 통합 앱", url_path="archive"),
    ]
    active = st.navigation(primary + previous, position="hidden")
    apply_style()
    with st.sidebar:
        brand()
        for page in primary:
            st.page_link(page)
        with st.expander("이전 실험 기록", expanded=active.url_path in {p.url_path for p in previous}):
            for page in previous:
                st.page_link(page)
    active.run()
except FileNotFoundError as exc:
    st.error("실행본 ZIP 전체를 한 폴더에 다시 풀고 실행해 주세요.")
    with st.expander("누락된 파일 확인"):
        st.code(str(exc))
except (ValueError, OSError, RuntimeError, KeyError) as exc:
    st.error("실행본의 모델과 평가 파일이 같은 버전인지 확인해 주세요.")
    with st.expander("오류 상세"):
        st.code(str(exc))
