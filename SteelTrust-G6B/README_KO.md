# SteelTrust G6 — SSFD 슬래그 유동 연구용 모델

## 1. 이 패키지는 무엇인가요?

SSFD 3축 진동 데이터의 **B(슬래그 직전) / S(슬래그 유동 중)** 구분을 연구하는 독립 실행 학습·평가 패키지입니다. 기존 G5 학습 결과의 높은 미검출과 출강 조건별 성능 편차를 줄이기 위해 개발했습니다.

**현재 성능이 가장 나은 것이 확인된 연구용 후보:** G6B RefSpec ExtraTrees (`trained_models/G6B_refspec_extratrees.joblib`). 전체 SSFD 16개 그룹으로 재학습해 저장한 가중치/모델입니다. **새 출강 그룹에서 검증된 양산용 모델이 아닙니다.**

**추가 딥러닝 후보:** G6 Contrastive Residual MLP (`g6_deep.py`). B 기준 주파수 변화량에 대한 신경망을 GPU에서 새로 학습합니다. 이전 G4/G5 CNN-GRU처럼 성능 개선이 보장되지 않으며, 현재 ZIP에는 전체 학습이 완료된 딥러닝 가중치가 없습니다.

## 2. 평가 결과 (직접 재계산)

| 항목 | 기존 G5-B SVM | 신규 G6-B ExtraTrees |
|---|---:|---:|
| 프로토콜 | B/S, 확인된 B 기준 2.5초 | 동일 |
| 평가 | 16개 출강 조건 4-fold 그룹 CV | 동일 |
| 평가 구간 | 0.5초 × 240개 | 동일 |
| 실제 S 중 감지 | 81 / 160 | **86 / 160** |
| 실제 B 중 오경보 | 5 / 80 | **5 / 80** |
| 슬래그 Recall | 50.6% | **53.8%** |
| B 오경보 FPR | 6.25% | **6.25%** |
| ROC-AUC | 0.755 | **0.789** |

G6는 G5가 틀렸던 20개 구간을 수정했으나, 새 오류도 15개 만들었습니다. 따라서 순개선은 5구간입니다. 출강 그룹 단위 부트스트랩의 검출률 차이 95% 구간은 약 -9.4~+13.8%p이며 0을 포함합니다. **통계적으로 확정된 개선이라 주장할 수 없습니다.**

**반드시 알아야 할 한계:** SSFD에는 독립적인 출강 조건이 16개뿐이고 G1~G6에서 반복적으로 실험 설정에 사용했습니다. 동일 데이터의 교차검증은 최종 독립 외부검증이 아닙니다. B/S 파일은 각각 5초 별도 녹음이므로 실제 슬래그 진입 시점을 예측했다는 뜻도 아닙니다.

## 3. 실제 변경 사항

1. B 구간 첫 2.5초를 **독립적으로 확인된 정상 기준 신호**로 사용하고 이 부분은 평가에서 제외합니다.
2. 입력 신호(6,400Hz, 3축)를 **0.5초 단위**로 분석합니다.
3. Welch 주파수 특징을 512·1,024 샘플의 두 해상도에서 계산하고, RMS 대체값(표준편차), 신호의 변화율·크기·왜도·첨도를 결합하여 75차원 특징을 만듭니다.
4. 타깃 구간의 특징에서 동일 출강의 B 기준 특징 중앙값을 빼서 **상대 변화량**을 계산합니다.
5. ExtraTrees 130개 결정나무를 학습합니다. 경보 임계값은 각 테스트 그룹을 제외한 학습 그룹의 3-fold 예측에서 정합니다. **외부 테스트 정답으로 임계값을 최적화하지 않습니다.**
6. 최종 모델은 모든 16개 그룹에서 재학습하고, 그룹 교차검증 예측만 평가 수치로 사용합니다.

## 4. Windows Python 3.12 실행 방법

압축을 해제한 뒤 **프로젝트 폴더 내부**에서 다음 파일을 실행하세요.

1. `SETUP_PY312_WINDOWS.cmd` — 가상환경과 PyTorch CUDA 12.6 설치
2. `CHECK_GPU_WINDOWS.cmd` — RTX 4070 Laptop 인식 확인
3. `QUICK_TEST_WINDOWS.cmd` — 파일, 추론, 딥러닝 순전파 기본 검사
4. `PREDICT_EXAMPLE_WINDOWS.cmd` — **학습 완료 G6 모델**로 예제 분석
5. `RETRAIN_G6_WINDOWS.cmd` — ExtraTrees 모델과 4폴드 평가 다시 만들기(선택)
6. `TRAIN_G6_DEEP_WINDOWS.cmd` — **신규 딥러닝 후보 전체 학습**(선택)
7. `STRESS_TEST_WINDOWS.cmd` — 센서 이득/잡음 시뮬레이션 재검사(선택)

명령줄 직접 실행:

```powershell
.venv\Scripts\python.exe g6.py predict --reference examples\B-1.csv --target examples\S-1.csv --persistence 2 --output runs\example_prediction.csv
.venv\Scripts\python.exe g6.py train --output runs\g6_retrained
.venv\Scripts\python.exe g6_deep.py --device cuda --epochs 200 --patience 30 --seeds 5 --output runs\g6_deep
```

`runs/g6_deep/summary.json`에는 **4개 평가 폴드를 모두 완료한 경우에만** 전체 지표가 기록됩니다. 깊은 신경망의 학습 시간이 길다고 해서 반드시 더 정확해지지 않습니다.

## 5. 실제 데이터 입력 조건

* 입력 CSV 필수 컬럼: `x,y,z` (추가 `time` 컬럼 가능)
* 샘플링: 6,400Hz, 데이터 단위는 SSFD와 일치해야 함
* `--reference`: **같은 출강 사건에서 슬래그가 아직 섞이지 않았다고 독립적으로 확인된 B 단계**의 진동 CSV(최소 16,000 샘플)
* `--target`: 판정할 진동 CSV(최소 3,200 샘플)
* 출강 상태 B가 언제인지 확인할 방법이 없다면 **G6-B를 사용해서는 안 됩니다.** G6는 E/B/S 3분류 모델이 아닙니다.
* 일반 `.pt` YOLO 영상 검출기와 입력 형식이 다르며 모델을 서로 교환할 수 없습니다.

### 출력 해석

`slag_score`는 ExtraTrees의 분류 점수이지 정확하게 보정된 사고 확률이 아닙니다. `raw_s_flag`는 구간별 연구용 판정, `persistent_alarm`은 연속 N개 구간의 점수가 임계값 이상인지 표시합니다. 자동 공정 중단에 연결하지 마세요.

## 6. 품질 검증 및 개선 한계

* 원본 SSFD 교차검증: Recall 53.8%, FPR 6.25% → **현장 자동차단에는 미달**
* 이득 ±5%: B와 타깃에 동일하게 적용한 단순 이득 변화 시험에서는 86/160, 5/80으로 동일했습니다.
* 센서 잡음 2%(기준 B 진동 표준편차 기준): 86/160, 7/80
* 센서 잡음 5%: 90/160, 8/80. 경보 증가가 반드시 실제 개선은 아닙니다.
* 이는 생성한 잡음의 모의시험이며 실제 설비 조건을 보증하지 않습니다.
* 다음에 필요한 것은 **제철소 신규 출강 기록**, 실제 진입 타임스탬프 및 정밀한 B 기준 확보 가능성 점검입니다.

## 7. 폴더 구성

```
SteelTrust_G6_SSFD/
  g6.py                 # G6 모델 재학습, 평가, 추론
  g6_deep.py            # RTX 4070 GPU 기반 딥러닝 후보 학습
  g6_stress.py          # 환경 변형 강건성 시험
  g6_report.py          # 과거 G5 대비 오류 변화 분석
  trained_models/
    G6B_refspec_extratrees.joblib   # 이미 학습된 후보
    G5B_archive.joblib             # 이전 G5 백업
    cv_metrics.json
    oof_predictions.csv
  reports/              # 성능·그룹별 결과·스트레스·오류 기록
  data/SSFD.zip         # 공개 SSFD v3 데이터
  examples/B-1.csv, S-1.csv
  tests/test_g6.py
```

## 8. 데이터 저작자 및 보안

SSFD v3: Sehri, Mert 등, *Steel Slag Flow Dataset*, Mendeley Data, DOI: 10.17632/mjnpsbs6fs.3, CC BY 4.0. https://data.mendeley.com/datasets/mjnpsbs6fs/3

`.joblib`는 Pickle 형식이므로 **출처를 신뢰하는 모델만 로딩**하세요. 모델 입력 데이터를 외부 인터넷에 전송하지 않습니다. 기존 G1~G5 파일은 수정하지 않았습니다.

딥러닝 전체 학습 후 `REVIEW_DEEP_WINDOWS.cmd`를 실행하면 G6 ExtraTrees와 동일 구간 비교 및 연구 후보 채택 기준을 확인할 수 있습니다. `VALIDATION_KO.md`에서 그룹별 오판 및 신뢰성 분석을 확인하세요.
