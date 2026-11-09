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

이 저장소의 `main`에는 소스·설정뿐 아니라 학습된 모델, 전체 원자료, 실험 결과와 사례 ZIP이 들어 있습니다. **Code → Download ZIP** 또는 `git clone`으로 받아 수정·실행·재학습할 수 있습니다. 별도의 실행본 ZIP을 병합할 필요가 없습니다. 전체 데이터가 포함되어 다운로드 용량이 큽니다.

1. Python 3.11 64비트와 Windows Python 런처(`py`)를 설치합니다.
2. GitHub의 전체 ZIP을 모두 풀거나 저장소를 복제합니다.
3. [START_WINDOWS.cmd](SlagWatch/START_WINDOWS.cmd)를 실행합니다. 첫 의존성 설치에는 인터넷이 필요합니다. 앱 실행과 사례 재시험에는 동봉한 학습 모델을 사용합니다.

## 포함 범위와 확인

| 항목 | 저장소에 포함된 내용 |
| --- | --- |
| 코드·환경 | 앱, 학습·재현 코드, 테스트, Windows 시작 파일, 고정 의존성 목록 |
| 모델 | K0·K1, 기존 NEU/G1 및 슬래그 실험 모델, ResNet18 인코더 등 74개 모델 파일 |
| KSDD2 | 공식 원본 3,336쌍 전부와 분할 메타데이터. 정식 항목 3,335쌍 및 공식 목록 밖 중복 1쌍을 원형 보존 |
| NEU-CLS | 원본 BMP 1,800장, 분할 목록·해시·출처 |
| SSFD V3 | 원본 CSV 48개·JPG 3개, 원본별 해시·출처 |
| 실험 | 기존 결과·규약·모델 선택 근거·검증 실행·재현 사례·보고서 |

파일별 바이트 목록은 [PROJECT_FILES.sha256](PROJECT_FILES.sha256), 집계·원본 출처는 [PROJECT_INVENTORY.json](PROJECT_INVENTORY.json)에 있습니다. 압축 해제 후 저장소 루트에서 Python 3.11 이상으로 `python tools/verify_project.py`를 실행하면 누락·변경을 확인합니다. 이 검사는 모델을 로드하거나 시험 데이터를 평가하지 않습니다. 수정 후 달라진 파일이 표시되는 것은 정상입니다.

KSDD2 학습 1,865장·검증 466장·공식 시험 1,004장은 그대로 분리했습니다. 공식 시험 데이터가 있어도 기존 평가 잠금과 채택 기준은 유지됩니다. 재학습은 `SlagWatch`에서 `python -m experiments.inspection_binary.reproduce --out ../ksdd2_retrained --cache ../ksdd2_features`를 실행합니다. 원본 데이터 복원 명령을 먼저 실행할 필요가 없습니다.

이 공개본은 확보된 최신 프로젝트 ZIP의 3,208개 파일을 그대로 보존하고 KSDD2 전체 원본을 추가한 것입니다. 가상환경·설치 패키지·재생성 가능한 중간 특징 캐시는 배포 대상에 넣지 않았습니다. 원래 클라우드 작업 폴더의 임시 파일 전체를 보관한 백업은 아닙니다. 실제 Windows 앱 기동과 전체 재학습은 별도 검증 대상입니다.

## 데이터 이용 조건

KSDD2 원본과 해당 이미지 파생물은 **CC BY-NC-SA 4.0**, SSFD는 **CC BY 4.0**입니다. 각 데이터 폴더의 저자·출처·라이선스 표시를 유지해야 합니다. NEU는 저자 배포본과 바이트 일치 및 출처를 기록했으나 명시적인 재배포 라이선스는 확인되지 않았습니다. 공개 저장소라는 이유로 모든 데이터에 상업적 이용 허가가 생기지는 않습니다. 데이터별 출처 문서를 확인하세요.

## 정상·불량 검사 결과

KolektorSDD2 검증 원본 466장에 원본·잡음·밝기 변화 7조건을 적용했습니다. 같은 3,262쌍에서 후보 K1은 K0 대비 정상 오경보를 601회에서 263회로 줄였고, 불량 미검출은 48회에서 76회로 늘었습니다. 후보는 사전 교체 기준에 미달하여 공식 시험 분할을 열지 않았습니다.

데이터 출처와 이용 조건은 [산업적 근거](SlagWatch/INSPECTION_SOURCES.md)와 [KolektorSDD2 출처](SlagWatch/data/ksdd2/SOURCES.md)에 있습니다.
