from base64 import b64encode
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SteelTrust · 검사 AI 교체 검증", layout="wide")
try:
    from src import demo_ui as ui
    from src.demo_design import apply_style, brand
    from src.inspection_ui import workbench
    from src.p1_tabular_ui import page as tabular_lab
    from src import security

    home = st.Page(workbench, title="검사 AI 교체 검증", default=True)
    primary = [
        st.Page(ui.overview, title="프로젝트 소개", url_path="overview"),
        st.Page(ui.search, title="오분류 사례 찾기", url_path="search"),
        st.Page(ui.regression, title="개선 전후 비교", url_path="compare"),
        st.Page(ui.replay, title="저장한 사례 다시 확인", url_path="replay"),
    ]
    previous = [
        st.Page(ui.filters, title="이전 표면 필터 실험", url_path="filters"),
    ]
    lab = st.Page(tabular_lab, title="표형 모델 검증", url_path="tabular-lab")
    active = st.navigation([home] + primary + previous + [lab], position="hidden")
    apply_style()
    with st.sidebar:
        logo = b64encode((Path(__file__).parent / "assets" / "posco-logo.png").read_bytes()).decode("ascii")
        st.html(f'<div class="posco-logo"><img src="data:image/png;base64,{logo}" alt="POSCO"></div>')
        if active.url_path == home.url_path:
            st.html('<div class="steel-brand"><span class="steel-symbol" aria-hidden="true"><i></i><i></i><i></i></span>'
                    '<div><b>SteelTrust</b><small>검사 모델 검증실</small></div></div>')
        else:
            brand()
        if st.button("모델 등록·교체", key="model-register-shortcut", type="primary", use_container_width=True):
            st.switch_page(lab)
        with st.expander("사용자 인증", expanded=False):
            auth_token = st.session_state.get("security-token")
            try:
                principal = security.authorize(auth_token, roles=tuple(security.ROLES)) if auth_token else None
            except security.SecurityError:
                st.session_state.pop("security-token", None)
                principal = None
                st.warning("로그인이 만료되었거나 계정을 사용할 수 없습니다.")
            if principal:
                st.write(f"로그인: {principal['username']} ({principal['role']})")
                if st.button("로그아웃", key="security-logout"):
                    security.logout(auth_token)
                    st.session_state.pop("security-token", None)
                    st.rerun()
            else:
                with st.form("security-login", clear_on_submit=True):
                    username = st.text_input("계정")
                    password = st.text_input("비밀번호", type="password")
                    submitted = st.form_submit_button("로그인")
                if submitted:
                    try:
                        st.session_state["security-token"] = security.login(username, password)
                    except (security.SecurityError, OSError) as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()
                st.caption("초기 관리자는 서버 콘솔에서 생성합니다. 모델 등록·교체는 로그인과 감사 기록이 필요합니다.")
            st.caption("임시 로컬 감사 저장소를 사용합니다. 외부 감사 서버를 설정하면 외부 서버 기록이 필수입니다.")
    active.run()
    with st.sidebar:
        st.divider()
        if active.url_path != home.url_path:
            st.page_link(home)
        with st.expander("NEU · 기존 G1 검증", expanded=active.url_path in {p.url_path for p in primary}):
            for page in primary:
                st.page_link(page)
        with st.expander("이전 실험 기록", expanded=active.url_path in {p.url_path for p in previous}):
            for page in previous:
                st.page_link(page)
        with st.expander("추가 검증 실험", expanded=active.url_path == lab.url_path):
            st.page_link(lab)
except FileNotFoundError as exc:
    st.error("실행본 ZIP 전체를 한 폴더에 다시 풀고 실행해 주세요.")
    with st.expander("누락된 파일 확인"):
        st.code(str(exc))
except (ValueError, OSError, RuntimeError, KeyError) as exc:
    st.error("실행본의 모델과 평가 파일이 같은 버전인지 확인해 주세요.")
    with st.expander("오류 상세"):
        st.code(str(exc))
