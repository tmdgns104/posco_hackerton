# SteelTrust · 검사 AI 교체 검증

검사 AI를 바꿀 때 **기존에 잡던 불량을 새로 놓치는지, 정상 제품에 새 경보가 생기는지** 같은 입력에서 확인합니다. 검사 AI 개발자·품질 검증 담당자가 모델 교체를 검토하고 실패를 다음 버전의 개발 회귀시험으로 보관할 수 있습니다.

## 시작

**전체 GitHub 배포본 안내:** 현재 `main`에는 아래 실행본 설명보다 넓은 범위인 KSDD2 전체 학습·시험 원본과 모든 동봉 모델·NEU·SSFD 데이터가 포함됩니다. GitHub ZIP만 풀면 되며 별도 실행본 병합이나 원자료 복원은 필요하지 않습니다. 아래 과거 실행본의 제외 항목 설명은 348 MB ZIP에 대한 기록입니다. 전체 배포 범위·무결성 확인은 저장소 루트 `README.md`와 `PROJECT_FILES.sha256`을 확인하세요.

Windows는 ZIP 전체를 풀고 `SlagWatch/START_WINDOWS.cmd`를 실행합니다. Python 3.11 64비트, Windows py 런처, 첫 패키지 설치 시 인터넷이 필요합니다. Linux에서는 `SlagWatch`에서 Python 3.11 가상환경을 활성화한 후 `python start_demo.py --bootstrap`을 실행합니다. 설치된 환경은 `python start_demo.py` 또는 `python -m streamlit run app_demo.py`로 시작합니다. 첫 화면은 **검사 AI 교체 검증**입니다. 기존 강판 6종 G1 시연은 사이드바의 **NEU · 기존 G1 검증**에서 열 수 있습니다. Windows 실기 검증은 대기 상태입니다.

1. **교체 판단**에서 KolektorSDD2 프로젝트의 전체 검증 기록을 엽니다. 저장 기록을 볼 때는 모델을 학습하거나 새 실행을 만들지 않습니다.
2. **새 검증 실행**에서 전체 466장 또는 명시된 시연 부분집합을 선택한 뒤 **새 추론 실행**을 누릅니다. 두 모델에 같은 입력을 주고 별도 실행 ID를 발급합니다. 부분집합의 결과는 전체 채택 근거에서 제외합니다.
3. **실패 사례**에서 고친 미검출, 새 미검출, 해소한 오경보, 새 오경보, 두 모델 모두 틀린 입력을 열어 봅니다. **사례 모음에 저장**한 뒤 **선택한 사례 ZIP**을 내려받을 수 있습니다.
4. **사례 모음·재시험**에서 **과거 실행 재현**은 당시 두 모델의 판정과 점수를 확인합니다. **모델 버전 재시험**은 저장한 개발 사례와 등록 모델을 선택해 동일 배열을 다시 추론하고 정답과 비교합니다. 새 판정과 실패를 별도 결과 파일에 기록합니다. 과거 ZIP 재현에는 해당 기록을 만든 실행 폴더와 당시 모델이 필요합니다.
5. **프로젝트 설정**에서는 로컬 모델 조합을 골라 정답이 있는 새 이미지 ZIP을 등록할 수 있습니다. `inspection_examples/registration_example.zip`은 새 ID로 등록해 볼 수 있는 KSDD2 검증 원본 4장(정상 2, 불량 2)입니다. ZIP 루트에 `manifest.csv`를 두고 `sample_id,image_path,true_label`을 제공합니다. 정답 없는 자료는 성능 평가에서 등록할 수 없습니다. 모델 파일 업로드를 받지 않습니다.

## 측정한 결과와 판단

KolektorSDD2 공식 학습/시험 경계를 지켜 개발 학습 1,865장, 검증 466장, 공식 시험 1,004장으로 구성했습니다. 정답은 원 제작자가 제공한 이진 마스크로 확인했습니다. 검증에서는 불량 49장·정상 417장마다 원본, 약한/강한 가우시안 잡음 각 2패턴, 밝기 0.8/1.2배를 시험했습니다. 같은 원본을 반복한 3,262회의 결과입니다.

| 검증 범위 | 모델 | TP | FN | FP | TN | 불량 재현율 | 정상 오경보율 |
|---|---|---:|---:|---:|---:|---:|---:|
| 원본 466장 | K0 | 42 | 7 | 33 | 384 | 85.71% | 7.91% |
| 원본 466장 | K1 | 39 | 10 | 30 | 387 | 79.59% | 7.19% |
| 7조건 3,262회 | K0 | 295 | 48 | 601 | 2,318 | 86.01% | 20.59% |
| 7조건 3,262회 | K1 | 267 | 76 | 263 | 2,656 | 77.84% | 9.01% |

기준 모델 K0는 강한 잡음에서 정상에 많은 경보를 냈습니다. K1은 인코더·특징 구성과 분류 알고리즘을 유지했습니다. 학습 원본 1,865장에 같은 수의 약한·강한 잡음과 어두운·밝은 변형을 더해 총 9,325개 입력으로 scaler와 로지스틱 분류기를 다시 학습했습니다. 결과는 학습 분포와 입력 수가 함께 바뀐 파이프라인에서 측정했습니다. 두 임계값은 각 모델의 검증 원본 점수에서 정했습니다. 고친 미검출 6회, **새 미검출 34회**, 해소한 오경보 409회, **새 오경보 71회**입니다. K1의 원본 재현율은 39/49이고 강한 잡음 패턴 102의 오경보율은 54/417입니다. 사전 기준(각 조건 재현율 ≥80%, 오경보율 ≤10%, 상대 악화 제한, 대표 범위에서 한 오류 감소·다른 오류 비증가)에 미달하여 교체 후보를 채택하지 않았습니다. 공식 시험 분할의 추론은 잠겨 있습니다. 실제 공장의 재검사 시간·비용·생산 중단·안전 효과는 측정하지 않았습니다.

- `experiments/inspection_binary/` — 사전 규약과 해시, K0 실패 근거, 수정 가설, 모델 파일, 임계값 및 최종 검증 판정. `test_access.json`은 공통 실행기의 검증 결과와 모델·데이터·규약을 묶어 공식 시험 접근 여부를 기록합니다.
- `inspection_runs/20261108T193331597782Z-ca753bd910bc42fcb3b785eda802f8f9/` — KSDD2 검증 전체. `inspection_runs/20261108T192809060181Z-cd886416dfc249b0ab65a0a9493e416e/` — 같은 실행기로 새로 실행한 NEU 검증 전체. 각 폴더에는 원본 목록, 설정, 개별 예측, 짝지은 비교, 전체·조건·클래스별 결과와 실패 기록이 있습니다. 정상/불량 프로젝트는 모든 이미지를 정상 또는 불량이라고 할 때의 기준선도 `baselines.csv`에 기록합니다.
- `inspection_cases/`, `inspection_examples/index.csv` — validation에서 가져온 실제 사례, 대응하는 실행 ID, 원래 예측과 입력 해시, 재시험 결과. 6종 프로젝트에서는 일반 오분류를 측정합니다.
- [산업적 근거와 범위](INSPECTION_SOURCES.md) · [KSDD2 출처·마스크 규칙·라이선스](data/ksdd2/SOURCES.md) · [조건별 모델 측정](experiments/inspection_binary/VALIDATION_FINDINGS_KO.md)

## 실행 및 재현 명령

아래 명령은 `SlagWatch`에서 실행합니다. `--demo`를 빼면 해당 분할의 전체 원본을 실제로 평가합니다. 새 실행은 기존 기록을 덮어쓰지 않습니다.

```bash
python -m src.inspection_cli list
python -m src.inspection_cli run --project ksdd2 --split validation --demo
python -m src.inspection_cli run --project neu_cls6 --split validation --demo
python -m src.inspection_cli register --template ksdd2 --title "개발용 등록 예시" --zip inspection_examples/registration_example.zip
python -m src.inspection_cases replay --project ksdd2 --run inspection_runs/20261108T193331597782Z-ca753bd910bc42fcb3b785eda802f8f9 --case inspection_cases/434867a2c16754fb9e362271a7e9b4658ad534511c18af70f9201dfbae9e69a6_54ffeb8f78ee3685082e40a325793159b5ffe5a3e7930da7a8be46179299a631.zip
python -m src.inspection_cases retest --project ksdd2 --run inspection_runs/20261108T193331597782Z-ca753bd910bc42fcb3b785eda802f8f9 --case inspection_cases/434867a2c16754fb9e362271a7e9b4658ad534511c18af70f9201dfbae9e69a6_54ffeb8f78ee3685082e40a325793159b5ffe5a3e7930da7a8be46179299a631.zip --model K1 --out inspection_cases/retests
```

모델과 공통 실행기는 `src/inspection_model.py`, `src/inspection_projects.py`, `src/inspection_runner.py`로 연결합니다. 새 승인 모델을 붙일 때에는 기존 클래스 목록, 고유 모델 ID와 전처리·가중치·임계값을 담은 identity, 픽셀 배치만 받는 `predict(raws)` 어댑터가 필요합니다. 학습 데이터와 타일은 독립 평가 원본 수에 더하지 않습니다. 검증에서 반복 사용한 사례는 개발용 회귀 세트입니다.

**배포 범위**　KSDD2 검증 전체 466장의 원본·마스크와 모든 공식 항목의 인덱스·해시를 담았습니다. 자료 이용 조건은 CC BY-NC-SA 4.0입니다. 동봉 사례의 float32 잡음 입력은 변경 자료입니다. 원본 압축 853 MB, 학습/시험 이미지 전체, 중간 특징 캐시와 가상환경은 실행본에서 제외했습니다. 원자료 전체와 재학습을 준비할 때만 다음 복원 절차가 필요합니다.

```bash
python -m src.inspection_data verify --split validation
python -m src.inspection_data fetch --archive ../KolektorSDD2.zip
python -m src.inspection_data index --archive ../KolektorSDD2.zip
python -m src.inspection_data verify --split all
```

전체 원본 복원 후 재학습하려면 아래 명령을 실행합니다. 결과와 특징 캐시는 새 폴더에 저장됩니다. 시연과 사례 재시험에는 동봉한 모델을 사용합니다.

```bash
python -m experiments.inspection_binary.reproduce --out ../ksdd2_retrained --cache ../ksdd2_features
```

## 이전 G1 및 슬래그 실험 사용 기록

## 실행

- **Windows**　ZIP 전체를 풀고 `SlagWatch/START_WINDOWS.cmd`를 실행합니다. 최초 설치에는 Python 3.11 64비트, Windows Python Launcher와 인터넷이 필요합니다. Windows 실기 검증 대기.
- **일반 실행**　`SlagWatch` 폴더에서 Python 3.11 64비트 가상환경을 활성화한 뒤 `python start_demo.py --bootstrap`을 실행합니다. 이미 설치한 환경에서는 `python start_demo.py`로 시작합니다. Streamlit 직접 실행 명령은 `python -m streamlit run app_demo.py`입니다.
- **시작 점검**　`python start_demo.py --check`로 패키지·필수 파일을 확인합니다. 시작 오류는 `logs/startup.log`, Windows 가상환경 생성 오류는 `logs/bootstrap.log`에 기록합니다. 시연에는 동봉한 데이터와 학습 완료 모델을 사용합니다.

## 네 단계로 시연하기

1. **프로젝트 소개**　모델 확정 후 최종 평가의 잡음 오분류와 원본 정답 감소를 함께 봅니다. 아래에서 오류를 처음 발견한 검증 자료와 여섯 채택 기준도 확인할 수 있습니다.
2. **오분류 사례 찾기**　검증 이미지의 잡음 강도와 패턴을 선택합니다. 기본 목록은 저장된 탐색 결과입니다. ‘선택한 조건으로 사례 찾기’를 누르면 두 모델을 실행합니다. 선택한 사례의 실제 결함과 동일 입력에 대한 두 모델의 판정을 비교합니다.
3. **개선 전후 비교**　검증 또는 최종 평가 중 하나를 선택합니다. ‘전체 평가’의 그래프·표·CSV와 ‘사례별 비교’가 이 선택을 함께 사용합니다. ‘사례별 비교’에서 바로잡은 오류와 새로 발생한 오류를 각각 열고 ‘이 사례 저장하기’를 누릅니다.
4. **저장한 사례 다시 확인**　받은 ZIP을 올리거나 동봉 사례를 선택합니다. ZIP에 저장한 float32 배열로 두 모델을 실행해 입력·모델·판정·결정 점수가 일치하는지 확인합니다. PNG는 화면용 미리보기입니다.

기존 V1/V2, 슬래그 및 통합 앱은 사이드바의 **이전 실험 기록**에서 엽니다. 아래의 슬래그 관련 설명과 `RESULT_KO.md`는 프로젝트의 이전 실험 기록입니다. 과거 통합 화면은 `python -m streamlit run app.py`로도 실행할 수 있습니다.

재현용 CLI 명령입니다. 마지막 인수에 앱에서 받은 ZIP의 경로를 넣을 수 있습니다.

```bash
python -m src.surface_g1_replay --run-dir experiments/surface_g1 --replay experiments/surface_g1/examples/fixed.zip
```

## 잡음 학습 효과

V0는 검증 원본 359/360장을 맞혔다. 같은 원본에 합성 잡음을 더하자 132장에서 한 번 이상 정답이 오답으로 바뀌었다(총 446회). G1은 기존 동결 ResNet18과 원본에서 이미 학습한 scaler를 사용한다. 원본 학습 이미지 1,080장(고유 parent 1,079개)에서 각 원본에 가중치 1/2, 잡음 σ 0.02·0.05·0.10에 각 1/6을 주어 마지막 로지스틱 분류기만 다시 학습했다. 학습 seed 301은 검증·시험 seed와 분리했다. 입력 필터는 두 모델 모두 없다.

| 최종 시험 조건 | V0 macro F1 | G1 macro F1 | 고친 오분류 | 새 오분류 |
|---|---:|---:|---:|---:|
| 원본 | 0.9972 | 0.9917 | 0 | 2 |
| σ 0.02 | 0.9794 | 0.9991 | 21 | 0 |
| σ 0.05 | 0.8997 | 0.9862 | 95 | 4 |
| σ 0.10 | 0.6521 | 0.9688 | 302 | 4 |
| 잡음 9조건 | 0.8437 | 0.9847 | 418 | 8 |

원본은 시험 이미지 360장, 각 강도는 같은 360장에 세 seed를 적용한 1,080회 판정이다. F1은 조건별 macro F1의 동일 가중 평균, 오류 수는 짝지은 판정의 합계다. 시험 원본 정답은 359장에서 357장으로 감소했다. 원본에서 개재물과 표면 패임을 각 1장 새로 틀렸다. 검증의 잡음 평균은 0.8480에서 0.9889로 상승했다. 검증 통과 후 두 모델을 동결했고, 1회 개방한 최종 시험에서도 여섯 기준을 모두 통과해 이 실험의 G1을 채택했다. 평가 대상은 공개 NEU-CLS의 결함 6종과 합성 잡음이다. 정상 제품 검출, 새 공장·카메라 환경, 실제 작업 안전은 시험 범위 밖이다.

이번 결과는 `experiments/surface_g1`에 있다. `protocol.json`과 `train_views.csv`는 조건·seed·가중치·원본/파생 해시를, `failure_conditions.json`은 이전 V0 탐색 근거를 담는다. `selection.json`과 `test_status.json`은 선택 및 시험 개방 이력을 기록한다. predictions/pairs/summary/classes/confusion/changes/by_parent 파일에는 동일 입력별 결정 점수와 분류, 조건·클래스·parent별 집계가 있다. manifest와 재생기는 기존 이미지 처리·평가 코어 3개, G1 학습·평가, 두 재생기와 두 화면 모듈, 모델·실험 결과의 해시를 검증한다. 재생 시 판정·입력·모델 해시는 정확히 일치해야 하며 결정 점수 허용 오차는 절댓값 0.0003이다.

직접 학습을 재현할 때는 새 빈 경로를 사용한다. 앱은 동봉한 모델로 바로 실행된다.

```bash
python -m src.surface_g1_study --output experiments/g1_repeat
python -m src.surface_g1_replay --run-dir experiments/g1_repeat --finalize
python -m src.surface_g1_replay --run-dir experiments/g1_repeat --replay experiments/g1_repeat/examples/fixed.zip
```

검증을 통과했으나 시험 평가가 중단된 경우 같은 경로에서 `--resume-test`로 동결한 모델을 재사용할 수 있다. 완료한 결과에 재학습하면 프로그램이 중단된다. 이전 필터·슬래그 실험은 사이드바의 **이전 실험 기록**에서 열 수 있다. 기존 사례 명령은 `python -m src.surface_replay --replay experiments/surface_v1/examples/fixed.zip`이다.

## 이전 슬래그 및 표면 필터 실험 기록

- SSFD는 완결된 5초 파일 48개를 한 번씩 판정했다. T1(시간 특징)은 S 검출 8/16, 오경보 5/32, T2(주파수 추가)는 9/16, 4/32였다. 고친 미검출 2, 새 미검출 1, 해소 오경보 1, 새 오경보 0. 원본 기준 TP≥12를 통과하지 못해 교란 검사를 열지 않고 경로를 전환했다. J의 저장 점수를 파일 평균으로 바꾸고 내부 파일 임계값을 맞춘 참고값은 2/16 검출, 2/32 오경보다. T1은 첨부 진단 F1과 48개 판정 모두 일치했다. `experiments/temporal5s`에 모델·분할별 경계·개별 예측·그룹 결과가 있다. 0.5초 검출과 평가 단위가 다르다.
- NEU-CLS는 정상 클래스가 없는 철강 결함 6종 분류다. 대학 공식 배포 원본 1,800장, 200×200 회색조·클래스당 300장을 확인했다. 동일 픽셀 Pa_101/Pa_105를 같은 parent 그룹으로 묶었고 원본은 모두 유지했다. 회전·반사 동등성과 밝기 정규화 후 거의 같은 영상도 검사했다. 원본 1,080/360/360장을 학습/검증/시험에 고정했다. 공장·코일 단위 독립성은 검증하지 않았다. 출처 및 원본 해시는 [데이터 출처](data/neu_cls/SOURCE.md)에 있다.
- ImageNet 고정 ResNet18에서 512개 특징을 배치 추출했다. 세 후보는 같은 train 원본을 썼고 필터별로 scaler와 로지스틱 분류기를 다시 적합했다. 필터 적용 순서는 원본 해상도에서 잡음 추가 → 필터 → 전체 이미지 resize → RGB 복제·ImageNet 정규화다. crop 또는 잡음 학습은 없다. 특징 캐시는 사례·실행·정책·강도·seed·실제 입력 해시를 구분한다.
- 검증 원본 360장에서 V0는 359장을 맞혔다(macro F1 0.9972). 잡음 3강도×3seed 평균 macro F1은 V0 0.8480, V1 0.8368, V2 0.8190이다. V1은 약한 잡음에서 개선됐지만 교란 전체에서 고친 오류 133회, 새 오류 178회다. V2는 각각 137회, 220회다. 평균 F1 +0.03 및 강도별 하락 제한에 두 후보 모두 미달했다. 후보 미채택, 시험 360장 모델 평가 미실행. 새 카메라 환경 또는 현장 효과는 측정하지 않았다.
- `experiments/surface_v1`의 protocol·split_manifest·selection·test_status로 평가 경계를 확인한다. validation_summary/classes/confusion/changes/by_parent는 조건별 전체 지표·클래스 재현율·혼동행렬·변화 수·원본 그룹 결과다. validation_predictions와 validation_pairs는 개별 예측 및 동일 입력 쌍이다. validation_discoveries 및 validation_discovered_inputs.zip은 V0 원본 정분류가 잡음에서 오분류가 된 446조건의 기록과 실제 .npy다. 전체 검증은 같은 360장에 대한 9조건을 동등하게 평균하며 반복 판정을 서로 다른 이미지로 세지 않는다. 이전 슬래그 실험은 좌측에서 계속 실행할 수 있다.

재현 학습은 새 빈 폴더에서 별도 결과를 만든다. 기존 실험 결과는 보존한다.

```text
python -m src.surface_study --reproduce surface_repeat
python -m unittest discover -s tests -v
```

[NEU 저자 배포 페이지](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/) · [ResNet18 공식 모델](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

## 이전 슬래그 실험 기록



**목표:** 공개 주조 진동에서 슬래그를 덜 놓치면서 비슬래그 오경보를 늘리지 않는지 평가한다. 포스코 출강 자동화와 일시적 현상에 의한 오판에서 출발했으며, 실제 검증 단위는 SSFD에 기록된 상태별 0.5초 구간이다. 데이터는 전로 영상이 아니다. 실제 종료 시점, 유출량·사고·불량·원가 절감은 측정하지 않았다. 기존 v2의 모델 평가·입력 이상 시험 기능은 ‘검증 도구’에서 사용할 수 있다.

## 이번 실행본 · MiniRocket 재검증

Windows에서는 ZIP 전체를 풀고 `SlagWatch/START_WINDOWS.cmd`를 실행한다. 최초 설치에는 인터넷과 Python 3.11 64비트가 필요하다. 기존 CNN 설치 마커가 있어도 새 의존성을 확인하는 별도 마커를 사용한다. 직접 확인한 환경은 Linux x86-64, Python 3.11.17 CPU이며 구분기와 변환기는 미리 학습돼 있다.

```text
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.lock.txt
python -m streamlit run app.py
```

첫 화면의 ‘비교 실험’에서 **MiniRocket · 증강 비교**를 고른다. 수정 전 J를 고르면 J→R0, R0를 고르면 R0→R1을 재생한다. 저장된 실제 탐색이나 새 자동 탐색을 열고 ‘전체 회귀 사례’로 전환하면 원본과 네 교란의 회복·새 오류를 같은 모델 쌍으로 확인할 수 있다. 선택 사례 JSON과 전체 비교 CSV를 내보낼 수 있다. R0→R1 화면에서 수정의 근거로 인용한 창은 해당 학습 그룹 내부의 검증 사례다. 화면에서 재생하는 최종 비교 창은 그 fold의 학습 제외 그룹에서 가져온다. 기존 CNN 실험과 기존 schema 1 JSON도 ‘이전 CNN’과 같은 CLI에서 열린다.

```text
python -m src.pair_detector --replay experiments/rocket_v1/examples/R0_R1_search_failure.json
python -m src.pair_detector --replay experiments/rocket_v1/examples/R0_R1_new_FN.json
python -m src.pair_detector --csv new_record.csv --model R1
```

마지막 줄은 6,400 Hz의 단일 time,x,y,z 기록(양의 3,200행 배수)에 전체 데이터 학습 실험 모델을 명시적으로 적용한다. 각 구간만 사용하고 입력 오류는 판정·점수를 비운다. 모델의 Ridge 결정 점수에는 모델별 임계값을 적용한다. 확률 해석을 하지 않는다.

**설계 및 제한.** 48개 파일과 16개 번호 그룹을 유지했다. 파일 라벨을 상속한 S와 E/B의 0.5초 상태를 분류한다. 동일 SSFD 연구의 B/S·512샘플 실험과 평가 단위가 다르다. 다변량 원시 파형을 aeon 1.6.0 MiniRocket의 9,996개 특징으로 변환하고, 깨끗한 fit 그룹에만 표준화와 Ridge를 적합한 구성이 R0다. R1은 같은 변환기와 특징 scaler를 그대로 사용하며, 같은 Ridge에 원본·독립 가우시안 잡음·한 축의 순간 충격 파형을 0.5/0.25/0.25로 가중해 학습한다. 오경보 예산 15%는 각 모델의 내부 원본 OOF 임계값 선택에만 사용한다. 기존 J와 C1은 저장 모델을 재사용했다. 산업 연구의 진동 간섭은 합성 시험 설계의 동기이며, 웨이블릿 모델이나 연속 유입시점 검출은 이번 실험에 포함하지 않았다. [MiniRocket 공식 API](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.convolution_based.MiniRocket.html) · [공개 기준 구현](https://github.com/angus924/minirocket) · [SSFD v3](https://data.mendeley.com/datasets/mjnpsbs6fs/3) · [관련 SSFD 연구](https://arxiv.org/pdf/2509.00034)

**실제 결과.** 원본 검출/오경보는 J 49/47, C1 38/71, R0 27/44, R1 25/44였다. R1의 잡음·X·Y·Z 검출/오경보는 각각 22/44, 24/40, 21/44, 24/44다. X에서는 R0보다 검출 1개 증가·오경보 1개 감소였지만 잡음에서는 검출 10개 감소·오경보 26개 감소였다. 모든 조건의 슬래그 검출 목표 미달로 두 후보 모두 채택하지 않았다. A0는 목표 미달인 과거 기본값이다. 네 외부 학습 세트에 적합한 R0의 학습 AUC 평균 1.000, 제외 그룹 원본 AUC 평균 0.491이다. 같은 16개 그룹을 거듭 사용한 개발 평가로, 설비 적용이나 실제 사고 감소를 평가하지 않았다.

**재생 근거와 실험 기록.** 새 결과는 `experiments/rocket_v1/`에 있다. `protocol.json`은 동결한 설정·코드 해시·분할, `manifest.json`은 실행·모델 식별자다. `models/`의 각 파일 안에는 두 모델이 공유하는 변환기·scaler와 개별 Ridge·임계값이 들어 있다. `fit_0..3/` 및 `fit_full/`에는 내부 모델, 원본·교란 점수, 증강 seed와 위치, fit별 통계·시간을 저장했다. `summary.csv`, `by_fold.csv`, `by_group.csv`는 조건·fold·그룹별 성능, `changes.csv`는 네 오류 변화, `predictions.csv`는 파형 SHA와 구간별 예측이다. `inner_diagnostics.csv`와 `evidence.json`은 내부 실패 근거, `outer_scores.csv`는 외부 점수 분포·AUC/AP·실제 FPR, `verdict.json`은 목표와 비교 적격성을 개별 필드로 저장한다. `compilation.json`은 별도 캐시의 실제 Numba 컴파일 측정이다. 자동 탐색의 제한 시간은 모델 로드와 워밍업을 마친 뒤부터 센다.

재학습을 원할 때만 다음 순서로 실행한다. 기존 CNN과 이전 평가 결과는 읽기 전용 입력이다. 같은 소프트웨어 버전이 필요하다.

```text
python -m src.rocket_warmup
python -m src.rocket_study
python -m src.rocket_eval
python -m src.pair_search --run rocket_v1
python -m unittest discover -s tests -v
```

## 이전 CNN 실행 기록

### 원시 파형 CNN과 실패 재검증

ZIP 전체를 풀고 Windows에서는 `START_WINDOWS.cmd`를 실행한다. Python 3.11 64비트와 Python Launcher가 필요하다. 첫 의존성 설치에 인터넷이 필요하며 원본, 평가 모델, 실제 탐색 결과는 ZIP에 포함한다. 이번 실행 검증은 Linux x86_64, Python 3.11.17, CPU용 PyTorch 2.8.0에서 수행했다. 일반 실행은 아래 순서다.

```text
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.lock.txt
python -m streamlit run app.py
```

이전 CNN 실험에서 수정 전 J와 수정 후보 C1을 선택하면 사전 실행한 실제 탐색 결과가 열린다. ‘실패 조건 자동 탐색’은 새로 시험하며 ‘저장된 탐색 결과 열기’는 저장 결과로 돌아간다. ‘전체 회귀 사례’에서 조건과 고친 미검출·새 미검출·해소/새 오경보를 고른다. 각 기록의 학습 제외 fold에서 같은 입력에 두 모델을 다시 적용하며, 점수·임계값과 원래 경보를 함께 표시한다. JSON에는 원본/시험 배열 해시, 재생 설정, 실제 결과, 개발자의 원인 가설, 수정 내용, 해당 전체 결과의 경로와 모델별 해시가 담긴다. 실제 모델이 가설을 도출하거나 코드를 수정한 것은 아니다. A0는 과거 기본값이며 이 버전의 검출 목표를 통과하지 못했다. 이전 기록 비교, 게이트 및 구형 탐색기는 좌측의 보조 화면에 남겨 두었다.

**학습과 결과.** C0는 한 구간의 7개 겹치는 512샘플 패치에 같은 CNN을 적용해 logit을 평균한다. 보류 없는 한 구간 판정에 이 점수와 fold별 임계값을 쓴다. C0 내부 예측에서 RMS–점수의 순위 상관계수는 내부 분할에 따라 −0.98에서 +0.99까지 달랐다. B 가중치 조정의 사전 조건은 성립하지 않아 가중치 수정 모델은 학습하지 않았다. 외부 평가 전에 수정 후보 C1을 고정했다. C1은 학습 그룹의 축별 표준화 후 현재 창의 공통 RMS로 나누는 요소 하나만 바꿨다. 상대 축 크기는 유지되고 절대 진동 크기의 일부 정보는 사라진다. 학습 예산·모델 구조·seed는 같다. 내부 검증 손실로 early stopping하며 세 내부 최적 epoch의 중앙값으로 재학습한다. 15% FP 예산은 내부 원본에 적용한다. 학습 데이터는 원본만 사용했다. 별도 J 증강 진단은 이번 결과에 포함하지 않았다.

원본의 검출/오경보는 A1 23/36, J 49/47, C0 46/102, C1 38/71이었다. 교란에서 J가 내지 못하던 경보를 C1이 내지만 원본과 교란의 검출 목표 모두 미달해 채택하지 않았다. 모든 파일 라벨은 0.5초 창에 상속한다. 16개 그룹을 반복 사용한 개발용 교차검증이다. 전체 평가의 분모는 조건마다 S 160개와 E/B 320개다. 겹친 패치 또는 반복 교란을 독립 사건으로 세지 않는다.

아래 명령은 프로젝트 폴더에서 실행한다. 첫 줄은 배포된 실제 사례와 모델을 확인한다. 둘째 줄의 `new_record.csv`는 사용하려는 단일 기록 경로로 바꾼다. `--csv`는 전체 그룹으로 별도 재학습한 실험용 모델, `--replay`는 해당 파일을 학습하지 않은 평가 모델을 사용한다.

```text
python -m src.pair_detector --replay experiments/cnn_v1/examples/recovered.json
python -m src.pair_detector --csv new_record.csv --model C1
```

학습부터 전체 평가와 실제 탐색 결과를 다시 생성하는 명령이다. 결과는 `experiments/cnn_v1`에 갱신한다. 보관용 ZIP을 새 폴더에 풀어 실행할 수 있다.

```text
python -m src.cnn_train
python -m src.cnn_eval
python -m src.pair_search
python -m unittest discover -s tests -v
```

`protocol_C0.json`은 C0 시작 전 설정, `protocol.json`은 내부 검증 후 고정한 C1 변경과 판정 기준이다. `fit_0..3.json`에는 학습·검증 그룹, epoch별 검증 손실과 임계값이 있다. `fit_full.json`과 `models/full_*.pt`는 별도의 전체 학습용이며 평가 화면에서 쓰지 않는다. `predictions.csv`는 개별 예측, `summary.csv`, `by_fold.csv`, `by_group.csv`, `train_outer_auc.csv`는 분할·조건별 성능, `pair_summary.csv`와 `changes.csv`는 동일 입력 전후 회귀다. `search_*_*.json`은 탐색별 모든 시도, `examples/`는 바로 재생 가능한 개별 사례다. 점수 허용 오차는 절대 1e−5, 판정은 정확히 일치해야 한다.

## 이전 버전의 실험 기록

## 검출기 비교 실행본

`experiments/detector_v1/`에 이번 개발용 그룹 교차검증 결과를 저장했다. 기존 저장 모델과 이전 10% 결과도 포함한다. 각 fold의 학습 그룹 안에서 오경보 15% 이하인 경계 중 S 검출이 가장 많은 경계를 골랐다. 같은 규칙을 여섯 구성에 적용했고 10%도 함께 기록했다. 최종 비교는 모델당 조건별 480구간 전부, 보류 없이 수행했다.

| 원본 · 15% 내부 예산 | S 검출 | S 미검출 | 오경보 | 비슬래그 정답 |
|---|---:|---:|---:|---:|
| A1 기존 특징 RF | 23 | 137 | 36 | 284 |
| H 기존 특징 SVC | 36 | 124 | 35 | 285 |
| I 강건 특징 SVC | 42 | 118 | 40 | 280 |
| R 강건 특징 RF | 22 | 138 | 40 | 280 |
| J 세부 주파수 대역 SVC | 49 | 111 | 47 | 273 |
| K J의 최근 최대 2초 특징 중앙값 | 36 | 124 | 58 | 262 |
| 내부 선택 절차 (fold 0~3 순서로 J/A1/I/K) | 44 | 116 | 71 | 249 |

원본 목표는 검출률 70% 이상·오경보율 15% 이하다. 원본에서 검출이 가장 많았던 J도 49/160 검출에 머물렀다. J는 합성 잡음 및 X/Y/Z 스파이크에서 모든 입력을 비슬래그로 분류했다. 교란 시 가장 가까운 학습 서포트 벡터와의 유사도가 작아지고 점수가 모델의 절편 근처에 모여 네 fold 모두 원본에서 정한 임계값을 넘는 구간이 없었다. 서포트 벡터는 SVC가 분류 경계를 계산할 때 참조하는 학습 예시다. K의 과거 특징 집계도 목표 달성에 도움이 되지 않았다. 내부 선택 절차의 조건별 검출/오경보는 잡음 24/42, X 29/42, Y 22/32, Z 16/40이다. 목표는 조건마다 적어도 96/160 검출, 최대 64/320 오경보다.

**입력 확인.** 48개 원본의 해시, 열 순서 time/x/y/z, 5초당 32,000행, 16개 번호와 그룹 분리를 확인했다. 공식 축 단위는 m/s², 샘플링 주파수는 6,400Hz다. 기록 시각 문자열 간격은 중앙값 156µs다. E/B/S는 파일 라벨로, 0.5초 창마다 그 상태를 상속했다. csv에는 창 안의 전환 시각 정답이 없다. 모델에는 파형에서 얻은 특징만 넣었다. 식별자와 파일 내 위치는 분할·같은 파일의 순서 유지·교란 재생에만 사용한다. 문헌의 고성능 설정 M9는 B/S, 입력 길이 512, 단일 채널을 교대로 쓰는 CNN-LSTM 실험이며, 여기서는 E도 포함한다. 논문 모델이나 세부 데이터 로더를 재현하지 않았다. [데이터](https://data.mendeley.com/datasets/mjnpsbs6fs/3) · [축·단위·발생 번호 설명](https://link.springer.com/article/10.1186/s13104-025-07486-8) · [프리프린트 M9 실험](https://arxiv.org/pdf/2509.00034) · [출판 논문](https://doi.org/10.1016/j.eswa.2025.130457)

**수정 근거.** 원본 파형은 보정·클리핑하지 않는다. A1/H는 진폭 특징을 유지하고 FFT에서 평균을 제거한 뒤 8개 상대 대역 에너지를 계산한다. I/R은 MAD·IQR·중앙 90% RMS와 median Welch 상대 에너지 및 총 에너지를 쓴다. SVC의 표준화는 특징 추출 후 학습 그룹에만 적합한다. I에서 절대 진폭·총 에너지는 이미 보존된다. 학습 그룹에서는 넓은 고주파 대역 안의 세부 구간별 S와 E/B 차이가 양쪽 방향으로 나타났다. J는 기존 36개에 축별 50 Hz 폭의 64개 대역 로그 에너지 형태를 추가했다. 같은 파일 안의 세부 대역 편차가 파일 간 편차보다 작아 K는 다른 변경 없이 J의 최근 1~4개 벡터의 중앙값을 쓴다. 기록을 연결하거나 첫 창을 제외하지 않는다. 원본 파형의 전처리 제거 효과나 물리적 원인은 이 비교로 확인하지 않았다.

**실행.** Python 3.11 환경에서 프로젝트 폴더를 기준으로 실행한다.

```text
python -m pip install -r requirements.lock.txt
python -m streamlit run app.py
```

재학습과 평가를 재현하는 순서다. 결과는 detector_v1 폴더를 갱신한다.

```text
python -m src.detector_diagnostics
python -m src.detector_study
python -m src.detector_shift
python -m unittest discover -s tests -v
```

별도 기록은 `python -m src.detector "기록.csv"`로 추론한다. 단일 6,400 Hz 파일의 time,x,y,z 열과 3,200의 배수 행을 받는다. 매 0.5초의 끝에서 보류 없는 원래 판정을 출력한다. 입력 이상은 점수와 판정을 비워 둔다. 이 경로는 전체 16그룹의 3-fold 내부 검증으로 선택해 재학습한 H(임계값 0.425698)의 실험용 실행 모델을 쓴다. 앱의 평가·재생은 해당 번호를 학습하지 않은 fold 모델만 사용한다.

**결과 파일.** `protocol.json`과 `../hi_v1/protocol.json`에 구성·특징·분할·판정 목표, `manifest.json`에 버전과 해시, `inner_thresholds.csv`와 `selection.json`에 fold별 경계·선택을 기록했다. `predictions.csv`와 `summary.csv`는 앱에서 내려받을 수 있다. `by_fold.csv`와 `by_group.csv`는 fold 및 그룹별 결과, `inner_oof.csv.gz`는 선택에 사용한 내부 예측, `J_score_shift.csv`는 학습/평가 그룹에서 교란 후 점수와 경계까지의 차이를 확인할 수 있는 표다. AUC와 AP는 각 fold 안에서 계산하고 네 fold의 단순 평균을 보고한다. 여섯 고정 구성 외의 ‘selected’ 행은 fold마다 내부에서 고른 예측을 모은 것이다.

## 이전 10% 예산의 점수와 보류 점검

첫 화면은 입력 유효성 검사 후 원래 모델 판정을 그대로 표시한다. 기존 경계 거리와 제안 정책은 모두 실험용이며, ‘안전 지원 검증’의 사례 보기 역시 보류 없음으로 시작한다. 후보 I를 선택하면 첫 화면에서 내부 검증의 점수 경계별 검출률과 오경보율을 확인할 수 있다. 안전 지원 검증에서 I의 ‘A1이 놓치고 I가 검출’ 사례를 선택하면 같은 구간에서 A1의 원래 판정과 I의 경보·보류 여부를 확인할 수 있다.

저장 모델과 기존 검증 예측만으로 다시 점검하려면 프로젝트 폴더에서 `python -m src.score_audit`를 실행한다. `experiments/score_audit/`의 `replay.csv`는 두 검출기 전체 점수 재현, `paired_s.csv`는 조건·정책 방식별 실제 S 네 분류와 보류 수, `inner_points.csv`는 I의 내부 fold별 경계 두 곳에서의 검출률·오경보율·AUC다. 같은 fold의 AUC는 경계를 바꿔도 같다. 모든 임계값과 모델은 기존 실행본의 값을 사용한다.

## 추가 화면 · 안전 지원 검증

직접 사용자는 공정 AI 개발·검증 담당자다. 좌측 **안전 지원 검증**에서 검출기·시험 조건·전달 방식을 고른다. 보류 없는 분류 성능은 기존 첫 화면에서 그대로 확인한다. 선택한 사례에는 파일 라벨, 원래 모델 점수·판정, 보류/전달, 자동 전달값을 함께 표시한다. 현장 접근이나 설비 제어를 지시하지 않는다.

- 비교 범위: A0/A1/H/I 및 기존 E/G 검출기 × 원본·합성 잡음·X/Y/Z 스파이크, 각각 480구간. A0/A1/H/I의 기존 예측을 그대로 읽는다. E/G의 기존 80% 배치 및 학습 그룹 컷의 위험도·결정을 재사용한다. 기존 표에 빠진 E의 Z축만 같은 v4 모델·게이트·고정 컷으로 재추론했다. 새 분류기나 위험 예측 모델은 학습하지 않았다.
- 정책: E/G는 경계 거리 대 저장된 오류 위험 추정. A0/A1/H/I는 모델마다 다른 실제 검출 임계값까지의 거리 대 기존 모델 간 불일치 규칙을 순위화한 제안 정책이다. 점수 척도가 다른 SVC에 E/G의 학습된 오류 추정기를 적용하지 않았다. H/I 계열의 고정 보류 컷은 각 fold의 기존 내부 검증 점수만으로 정한다. 검출 임계값과 보류 컷은 서로 다르다.
- 정확한 80% 비교는 조건마다 별도로 재생하는 오프라인 배치다. 동일 fold 120개에서 같은 96개 수량을 전달한다. 정책 입력에 시험 조건명이나 정답을 주지 않는다. 모델·조건 선택은 사람의 결과 조회용이다. 고정 컷 방식은 평가 데이터에 적용한 실제 전달률을 보고한다. 같은 고정 난수 32개(seed 0~31)의 무작위 보류를 같은 모집단에 반복 적용한다. 무작위 평균·표본 표준편차·최솟값·최댓값이 저장된다. 고정 모드의 무작위 기준선은 확률 0.8을 사용한다.
- TP_pass는 실제 S의 맞는 경보 중 전달된 수, FN_pass는 실제 S의 잘못된 비슬래그 판단 중 전달된 수, H_TP/H_FN은 각각의 보류 수다. 이 네 값의 합은 항상 N_S다. FP_pass의 분모는 N_N이다. S/E/B와 E+B의 전달률도 별도로 기록한다. 빈 분모는 N/A로 처리한다. 누락·값 고정 등 모델 점수가 없는 입력의 품질 검사는 기존 별도 기능 시험이다.
- I 원본, 정확히 384건 전달: 보류 없음에서는 28건의 맞는 S 경보와 132건의 미검출이 있다. 기존 경계 거리 정책은 TP_pass 10 / FN_pass 108 / H_TP 18 / H_FN 24 / FP_pass 8. 불일치 우선은 6 / 110 / 22 / 22 / 4. 무작위 32회 평균은 21.97 / 106.16 / 6.03 / 25.84 / 13.78. 실제 경보가 가려지는 비용이 확인돼 두 보류 정책을 모두 실험용으로 둔다. 고정 컷 방식에서 I 원본 실제 전달 수는 경계 거리 391, 불일치 우선 383이다.
- 산업 원문: [포스코 공개특허 KR20040056051A](https://patents.google.com/patent/KR20040056051A/ko), [공개공보 PDF](https://patentimages.storage.googleapis.com/9b/b7/6b/7ca5bd6f38fe0f/KR20040056051A.pdf), [포스코 명장 인터뷰](https://newsroom.posco.com/kr/포스코명장-특별인터뷰㉓-포항-제강부-이영진-명장/), [Ergolines OSD](https://www.ergolines.it/products/ladle-optical-slag-detector-osd/), [Kiss KT2000](https://kisstechnologies.com/kt2000-ladleslag/). 특허의 래들→턴디시 진동 간섭 문제와 제안된 개선, 인터뷰의 전로→래들 자동화 사례, 공급업체의 제품 설명은 각각 해당 자료의 맥락이다. 본 시제품은 주변 진동의 실제 발생 빈도, 작업자 접근 감소, 사고 예방, 설비 정지, 유출량 감소를 측정하지 않는다.
- 재집계: `python -m src.safety_eval`. 결과 경로 `experiments/safety_support/`. `policies.csv`는 모델·조건·정책별, `group_runs.csv.gz`는 그룹별/seed별, `random_summary.csv` 및 `random_groups.csv.gz`는 무작위 분포다. `cases.csv.gz`는 사례의 원래 점수·예측·보류 정보, `training_cuts.csv`는 H/I 계열 보류 컷이다. 기존 출력은 보존된다. 모든 게이트 지표는 모델 출력의 대리지표다.

## 변경 및 결과

- 복구된 작업공간의 최신 완전한 기본 소스는 v4였다. Git 저장소가 아닌 ZIP 기반 작업공간이다. 원본 디렉터리와 ZIP을 비교한 뒤 별도 복사본에서 작업했다. v4 코드·데이터·모델·기존 결과물은 일괄 재학습하거나 대체하지 않았다.
- src/hi_audit.py는 원본 검증, 기록 시각 간격, v4의 저장 예측·E 필터·F 후보·선택된 2,000회 시험을 점검한다. 기존 주파수 RF의 원본은 TP/FN/FP/TN = 69/91/100/220. 오경보는 E 52건, B 48건. 8번 그룹은 S 10개를 전부 놓쳤고, 11번 그룹은 S를 모두 찾으면서 비슬래그 20개 중 15개에 오경보를 냈다.
- v4의 E 보정은 고립 피크를 바꾸는 입력 전처리다. X축 스파이크에서 동일 진폭 RF의 TP/FN/FP/TN은 70/90/144/176 → 57/103/119/201. 실제 S 중 원래 검출했지만 보정 후 놓친 경우는 17개, 반대로 회복한 경우는 4개다. 일부 S 판정에 필요한 입력 차이가 사라진 사례가 있지만, 어떤 피크가 물리적으로 슬래그에서 유래했는지는 확인할 수 없다. F의 X축 결과와 Y/Z축의 큰 편차, E의 선택된 2,000회 415→233 오류도 저장 결과와 일치했다.
- src/hi_models.py는 H의 v4와 동일한 진폭·상대 대역 에너지 36개, I의 축별 MAD·IQR·중앙 90% RMS·median Welch 기반 로그 에너지 특징 36개를 구현한다. H와 I 모두 StandardScaler + RBF SVC(C=1, gamma=scale, balanced, probability=False, seed=42). 원본만 학습하고 신호를 필터링하거나 정답을 변경하지 않는다. SVC의 판정 점수는 RF의 점수와 척도가 다르다.
- src/hi_study.py는 기존 4개의 outer 그룹 분할과 그 학습 그룹 안의 3-fold 그룹 검증을 사용한다. 내부 원본의 오경보율 10% 이하에서 TP 최대 → FP 최소 → 가장 높은 임계값 순으로 선택한다. 경계의 동점은 score >= threshold로 동일 처리한다. 앞선 대화에서 시연한 ‘음성 기록으로 예산만 맞추는’ 계산과 다른 규칙이다. A0는 기존 0.5, A1은 같은 저장 모델에 위 임계값만 적용한 비교 기준이다. 임계값은 평가 fold의 정답으로 맞추지 않고 같은 fold의 다섯 조건에 고정한다.
- 원본 전체에서 **A1은 TP 17·FP 22, H는 28·24, I는 28·17**이다. I는 기존에 놓친 S 17개를 회복하고 기존이 맞힌 S 6개를 새로 놓쳤다. E/B 오경보는 A1의 17/5에서 I의 2/15로 바뀌었다. I의 원본 TP 차이는 fold 0~3 각각 +2,+6,+2,+1이다. **잡음에서 I의 FP가 14→46(+10%p)**으로 늘어 범용 채택에 실패했다. H는 원본 FP 증가와 교란 악화로 실패했다. 후보 I도 실제 S를 28/160만 검출해 실용적 성능은 부족하다.
- 센서 시각 문자열의 관측 간격은 48파일 모두 156 µs, 첫 샘플부터 마지막 샘플은 4.991844초였다. 공개 사양은 6,400Hz(156.25 µs)이고 이 실험은 기존 기준선과 동일하게 공식 주파수와 샘플 번호를 사용한다. 같은 번호 E/B/S를 한 묶음으로 분리했으며 각 창의 라벨은 **파일 라벨을 상속**했다. 동일 파일의 창은 독립 조업 사건이 아니다. 번호 그룹들의 완전한 통계적 독립성도 입증되지 않았다.

## 재실행 및 산출물

다음 명령을 압축을 푼 프로젝트 루트에서, 패키지가 설치된 Python 3.11 가상환경으로 실행한다.

    python -m pip install -r requirements.lock.txt
    python -m src.hi_audit
    python -m src.hi_study
    python -m unittest discover -s tests -v
    python -m streamlit run app.py

기존 자료는 reports/, artifacts/, experiments/의 v4 파일이다. 이번 실행은 **experiments/hi_v1/** 안에서만 결과를 생성·갱신한다.

| 파일 | 내용 |
|---|---|
| protocol.json | 결과를 보기 전에 고정한 모델·특징·seed·fold·조건·채택 기준 및 기준 모델 해시 |
| thresholds.csv, inner_oof_clean.csv | fold별 임계값과 선택에 사용한 학습 그룹 내부 검증 점수 |
| models/fold_0~3.joblib | 실제 A0/A1/H/I 추론에 사용하는 RF, scaler/SVC, 모델별 기준과 메타정보 |
| predictions.csv | 4개 방식 × 5개 조건 × 480구간, 파형 SHA256, 점수·판정·라벨·그룹 |
| summary.csv, by_fold.csv, by_group.csv | 조건별 TP/FN/FP/TN, 재현율·오경보율·정밀도·균형 정확도, E/B 각각의 오경보, A1 대비 고친 오류·새 오류 |
| adoption.json | 원본 및 교란 채택 조건별 충족 여부와 fold별 검출 차이 |
| timestamp_audit.csv, v4_audit.csv, v4_clean_groups.csv | 원자료 시각, 기존 실험·오류 양상 검산 |

진동 배열 하나에 대한 추론에는 파일명, 라벨, 조건명이 필요 없다. 평가용 재생기에서만 기록 ID를 이용해 올바른 학습 제외 모델을 고르고 기존 방식의 교란 seed를 복원한다. 원본에서 복사한 동일 교란 배열을 네 방식이 공유한다. HOLD로 분모를 줄이지 않으며, 주 시험 밖의 잘못된 크기·비유한값·고정축 입력은 분류 전에 검사한다.

**검증:** Linux/Python 3.11에서 기존 테스트 12개와 새 핵심 검증 4개 통과. 독립적인 재실행에서 임계값, 9,600행 예측, 집계와 채택 판정 파일 일치. 실제 브라우저에서 조건·모델 전환, CSV 내려받기, 기존 회귀 화면 접근 확인. Windows 자체에서의 실행은 별도 확인 대상이다.

원자료: Sehri 외, Steel Slag Flow Dataset v3, DOI https://doi.org/10.17632/mjnpsbs6fs.3 , CC BY 4.0. data/raw/README_KO.md와 licenses/에 출처·사용 조건을 포함한다. 원본 분절·합성 교란·특징 및 예측은 프로젝트 파생 작업이다. 같은 16그룹을 여러 개발 실험에 사용했으며 독립 현장 검증 결과로 해석할 수 없다.
