# SteelTrust G8.1 — dohyeon 스타일 디자인 패치

## 변경 사항
- 공개 GitHub `dohyeon` 브랜치의 밝은 운영화면 스타일(라벤더/보라 포인트, 흰색 카드, 절제된 내비게이션)을 참고한 UI만 수정했습니다.
- 왼쪽 사이드바 **SteelTrust 제목 위에 POSCO 공식 워드마크 이미지**를 배치했습니다.
- 로고 출처: `https://raw.githubusercontent.com/tmdgns104/posco_hackerton/dohyeon/SlagWatch/assets/posco-logo.png`
- 첫 화면을 짧은 설명/3단계 안내/3개 검사 분야 카드로 재구성했습니다.
- 모델 가중치, 데이터, 예측값, 임계값, 평가 통계 및 백엔드 알고리즘은 변경하지 않았습니다.

## 기존 G8에 적용
1. 패치 ZIP의 `SlagWatch` 폴더를 기존 G8 설치 폴더 상위에 병합하세요.
2. 기존 `START_WINDOWS.cmd`를 실행합니다.
3. 브라우저에서 `http://127.0.0.1:8501`을 새로고침합니다(Ctrl+F5).
4. POSCO 로고가 인터넷에서 표시되지 않으면 `GET_POSCO_LOGO_WINDOWS.cmd`를 한 번 실행해 로컬 저장하세요. 이후 오프라인에서도 보입니다.

## 변경된 소스
- `src/g8_easy.py` : 헤더/사이드바/첫 화면 표시만 변경
- `src/g8_dohyeon.css` : CSS 스타일 파일 신규
- `GET_POSCO_LOGO_WINDOWS.cmd` : 원본 로고 이미지 캐시용 선택 파일

## 기타
- 브랜드 표시는 대회·연구 시연용입니다. POSCO가 SteelTrust를 공식 개발·승인·후원했다는 의미는 아닙니다.
- 현재 로고 파일은 저희가 재가공하지 않고 GitHub dohyeon 브랜치 공개 원본을 그대로 사용합니다.
