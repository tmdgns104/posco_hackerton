# SteelTrust · 검사 AI 교체 검증

SteelTrust는 검사 AI 개발자와 QA 담당자가 모델 교체 전에 새로 놓치는 불량과 새 오경보를 확인하는 Streamlit 도구입니다. 같은 이미지와 시험 조건으로 기존 모델과 후보를 실행하고, 판정이 달라진 입력을 저장해 다음 버전의 회귀시험에 사용합니다.

## 프로젝트 구성

| 위치 | 내용 |
| --- | --- |
| [SlagWatch/src](SlagWatch/src) | 데이터 등록, 모델 추론, 조건별 비교, 사례 저장·재시험, 화면 |
| [SlagWatch/app_demo.py](SlagWatch/app_demo.py) | Streamlit 기본 진입점 |
| [SlagWatch/experiments](SlagWatch/experiments) | 학습·재현 코드, 실험 규약, 모델 선택 근거 |
| [SlagWatch/inspection_runs](SlagWatch/inspection_runs) | 정상·불량 검사와 결함 6종 분류의 조건별 검증 결과 |
| [SlagWatch/tests](SlagWatch/tests) | 데이터, 추론, 집계, 사례 재생 및 회귀시험 |
| [SlagWatch/README.md](SlagWatch/README.md) | 실행 방법과 데이터 복원 명령 |

## Windows 실행

전체 소스와 설정은 이 저장소에 있습니다. 모델 가중치·검증 이미지·사례 ZIP을 포함한 배포 파일은 `SteelTrust_Inspection_Regression.zip`(348 MB)입니다. GitHub의 **Code → Download ZIP**은 소스 배포입니다.

1. Python 3.11 64비트와 Windows Python 런처(`py`)를 설치합니다.
2. 모델과 검증 이미지가 들어 있는 실행본 ZIP을 모두 풉니다. Git 작업 폴더에서 실행하려면 압축 안의 `SlagWatch` 폴더와 저장소의 같은 이름 폴더를 병합합니다.
3. [START_WINDOWS.cmd](SlagWatch/START_WINDOWS.cmd)를 실행합니다. 첫 의존성 설치에는 인터넷이 필요합니다. 앱 실행과 사례 재시험에는 동봉한 학습 모델을 사용합니다.

## 정상·불량 검사 결과

KolektorSDD2 검증 원본 466장에 원본·잡음·밝기 변화 7조건을 적용했습니다. 같은 3,262쌍에서 후보 K1은 K0 대비 정상 오경보를 601회에서 263회로 줄였고, 불량 미검출은 48회에서 76회로 늘었습니다. 후보는 사전 교체 기준에 미달하여 공식 시험 분할을 열지 않았습니다.

데이터 출처와 이용 조건은 [산업적 근거](SlagWatch/INSPECTION_SOURCES.md)와 [KolektorSDD2 출처](SlagWatch/data/ksdd2/SOURCES.md)에 있습니다.
