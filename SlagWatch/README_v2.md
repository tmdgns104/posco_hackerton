# SlagWatch | 공개 주조 진동의 슬래그 유동 검사

2026년 11월 7일 해커톤 개발 v2. 데이터, 그룹별 학습 모델, 평가 결과, 실행 코드가 함께 들어 있다. 앱에서 기록을 선택하면 학습에서 그 기록의 그룹을 제외한 모델로 다시 계산한다.

## Windows에서 열기

1. ZIP 전체를 한 폴더에 풀고 Windows용 Python 3.12 64비트와 Python Launcher가 설치되어 있는지 확인한다. 터미널에서는 `py -3.12 -V`로 확인한다.
2. **START_WINDOWS.cmd**를 더블클릭한다. 첫 실행은 인터넷에서 패키지를 설치한다. 설치 후에는 학습 없이 바로 실행할 수 있다. 실행 창을 열어 둔다.
3. 브라우저가 자동으로 열리지 않으면 실행 창에 표시된 Local URL을 연다. 기본 주소는 http://127.0.0.1:8501 이다. 실행 창에서 Ctrl+C로 종료한다.

Windows 실행기를 쓰지 않을 때는 프로젝트 폴더에서 아래 명령을 사용한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.lock.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Linux에서는 Python 3.12 가상환경에서 `python -m pip install -r requirements.lock.txt` 후 `python -m streamlit run app.py`로 연다. 실행 검증 환경은 Linux / Python 3.12이며, Windows 64비트용 의존성 파일의 배포 여부도 확인했다.

## 처음 확인할 구간

앱 상단 **미검출 개선 사례**를 누르면 S-1의 9번 구간으로 이동한다. 기존 RF는 0.406, 주파수를 추가한 RF는 0.580으로 실제 슬래그를 검출한다. **스파이크 오경보 사례**는 주파수 RF가 실제 음성을 양성으로 잘못 판정하는 예시다. 두 사례 모두 같은 화면에서 원본·시험 입력을 나란히 비교한다.

- **B-12.csv / 구간 1 / 원본 → 스파이크**. 원본의 진폭 RF 유동 점수 0.355가 0.531로 변한다. 실제 라벨은 B이며 스파이크를 넣은 뒤 오경보가 발생한다.
- **B-12.csv / 구간 0 / 값 누락 → 값 고정 → 원본**. 누락과 고정은 각각 다른 사유로 보류된다. 원본을 선택하면 원래 점수와 판정이 복원된다. 홀수 구간은 이 두 시험에서 원본을 유지한다.
- **S-1.csv / 구간 0 / 원본**. 실제 라벨은 S이지만 세 모델이 모두 놓친다. 전체 평가 탭에서 다른 오판도 찾을 수 있다.
- **전체 평가**에서 세 모델의 검출·오경보·보류와 실패 목록을 확인한다. 선택 기록의 10개 구간, 전체 예측 2,400행 및 평가 JSON을 내려받을 수 있다.

## 학습·검증 재실행

프로젝트 폴더에서 설치한 가상환경의 Python을 사용한다.

```text
python -m src.train_eval --data-dir data/raw --out artifacts --reports reports
python -m unittest discover -s tests -v
```

재학습 후 앱을 다시 시작하면 생성된 결과를 읽는다. 데이터 파일의 해시는 배포 묶음에 포함된 원본 명세와 비교한다. 파형은 공식 샘플링 속도 6,400Hz로 계산한 파일 시작 기준 상대 시간과 센서 원본 단위를 쓴다.

## 코드에서 따라갈 순서

| 경로 | 확인할 내용 |
|---|---|
| src/data.py | CSV 이름·해시·열·행 수 검증 → 3,200행씩 10구간으로 분할 |
| src/features.py | 진폭 특징 12개와 주파수 대역별 에너지 비율 24개 계산 |
| src/train_eval.py | 그룹을 나눈 뒤 원본 학습, 5개 조건 평가, 모델·결과 저장 |
| src/faults.py, src/quality.py | 학습 fold의 축별 기준과 고정 seed로 교란 생성, 배열 자체로 누락·고정 검사 |
| src/inference.py | 입력 검사 후 기준선과 두 RF 판정. 앱과 일괄 평가가 함께 호출 |
| app.py | 파일·구간 선택, 원본과 시험 파형 비교, 실패 목록·다운로드 |
| reports/ | predictions.csv는 구간 단위 결과, by_group.csv는 그룹별 지표, manifest.csv는 원본별 SHA256, metrics.json은 전체 지표·설정·버전 |
| artifacts/ | fold 4개의 모델과 그룹별 모델 연결표 |
| tests/test_contracts.py | 그룹 분리, 구간별 재현, 집계 분모, 화면 조작 검증 |

## 데이터와 사용 범위

Sehri, Mert; Ayres, Lucas Mantuan; Boldt, Francisco de Assis; Dumond, Patrick; Cuadros, Marco Antonio de Souza Leite (2026). Steel Slag Flow Dataset (SSFD), Mendeley Data, V3. [원본](https://data.mendeley.com/datasets/mjnpsbs6fs/3) · [DOI](https://doi.org/10.17632/mjnpsbs6fs.3) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

원본 CSV와 데이터 설명 이미지는 제공본 그대로 들어 있다. 0.5초 구간화, 특징, 교란 신호, 예측과 통계는 이 프로젝트가 생성했다. data/raw/README_KO.md 및 manifest.json에 원출처와 파일별 해시가 있다. 검증 범위는 공개 주조 기록의 상태 라벨과 합성 입력 이상이다. 포스코 전로의 출강 종료 판단 및 현장 센서 고장 검출 성능은 별도 현장 데이터가 필요하다.

## 개발 기록과 평가 파일

11월 7일 사전 개발·학습을 허용한다는 작업 전제로 구현했다. v1 평가에서 미검출이 많아 v2에서 주파수 특징을 추가했다. 세 모델 모두 같은 16개 그룹의 같은 구간으로 비교한다. RF 설정, 임계값 0.5, 합성 교란 생성 규칙은 두 RF에 공통이다.

보고서 RESULT_KO.md에서 원본·교란 결과와 그룹별 편차를 확인할 수 있다. 예측 CSV에서 baseline_pred는 단일 RMS, rf_pred/rf_score는 기존 진폭 RF, spectral_pred/spectral_score는 주파수를 추가한 RF를 나타낸다. input_status는 v1 호환용 기존 진폭 RF 상태다. 보류일 때는 세 모델의 예측과 점수가 모두 비어 있고 reason_codes에 사유가 기록된다.
