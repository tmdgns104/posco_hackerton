# Kolektor Surface-Defect Dataset 2 (KSDD2) — 출처와 이용 조건

제공·주석: **Kolektor Group d.o.o.** 배포: **Visual Cognitive Systems Laboratory (ViCoS), University of Ljubljana**. Jakob Božič, Domen Tabernik, Danijel Skočaj, “Mixed supervision for surface-defect detection: from weakly to fully supervised learning”, *Computers in Industry* (2021), [연구 논문](https://prints.vicos.si/publications/385). 본 시제품에 대한 원저자의 보증이나 제휴를 의미하지 않는다.

- [공식 데이터 설명, 다운로드, 라이선스](https://www.vicos.si/resources/kolektorsdd2/) → [공식 단축 링크](https://go.vicos.si/kolektorsdd2) → [실제 ZIP](https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip).
- 공식 다운로드 2026-11-08 UTC. 별도 버전 번호는 공표되지 않았다. **2023-09-28 10:40:04 UTC**는 HTTP Last-Modified이며 연구 발표일이나 수집일이 아니다. ETag: `"65155804-32d9ad9b"`. 원본 853,126,555바이트; SHA-256 `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`.
- 공식 원본 이미지·마스크와 여기에서 만든 이미지 단위 정답, 예시 이미지, 교란 영상 자료 및 그 자료를 담은 사례의 이미지에는 [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)을 적용한다. [라이선스 전문(오프라인)](inspection_license/CC-BY-NC-SA-4.0.txt). **비상업적 용도**로만 사용하고 저작자·출처·라이선스와 변경 사항을 표시한다. 변경 자료를 배포할 때에는 같은 라이선스를 적용한다. 라이선스가 허용하는 권리를 추가로 제한하지 않는다. 상업적 이용 문의는 공식 페이지의 Danijel Skočaj에게 한다.
- 원본 PNG와 마스크는 변경하지 않았다. 새로 한 일: 이미지 단위 라벨 변환, 중복 점검, 공식 학습 데이터의 학습/검증 구분, 예시 이미지 축소·병치·작은 결함 위치 표시, 시험 조건에 따른 밝기 및 잡음 변환. 공식 test의 구성과 정답은 보존한다. 이 출처 표시는 KSDD2 데이터에 관한 것으로, 패키지의 다른 데이터·소프트웨어 전체를 같은 라이선스로 재허가한다는 뜻이 아니다.

## 정답·분할의 근거

[저자 제공 로더](https://github.com/vicoslab/mixed-segdec-net-comind2021/blob/21583eb22e719a70fee388ce45f0f7f27f529926/data/input_ksdd2.py#L29-L42)는 공식 분할별 번호에 대해 원본과 단일채널 `_GT.png`를 쌍으로 읽는다. [저자의 마스크 처리](https://github.com/vicoslab/mixed-segdec-net-comind2021/blob/21583eb22e719a70fee388ce45f0f7f27f529926/data/dataset.py#L102-L108)는 비영(非零) 픽셀이 있으면 positive로 정의한다. 여기서는 리사이즈 전 원본에서 이 규칙을 검사한다: 모든 공식 마스크는 같은 크기의 8비트 회색조이고 값은 0 또는 255다. **흰 픽셀 유무만** 이미지 라벨로 변환한다. 누락·형식 오류·중간 계조·크기 불일치는 오류이며 정상으로 채우지 않는다. 모델에는 RGB 이미지 배열만 전달하고 라벨·마스크·파일 경로·ID는 입력 특성이 아니다.

ZIP의 일반 파일은 **6,677개**(이미지 3,336, 마스크 3,336, 분할 메타데이터 5). 이 중 공식 `split_weakly_246.pyb`의 두 ID 목록과 일치하는 원본은 **3,335쌍**이다. 작성 당시 `10301 (copy).png`와 그 마스크는 정식 `10301`과 각각 파일 SHA-256까지 같고 공식 목록에 없다. 원본 디렉터리에는 남기고 [exclusions.csv](exclusions.csv)에 제외 이유를 기록했다. 주석 종류를 나타내는 분할 메타데이터의 불리언을 정상/불량 라벨로 해석하지 않는다.

| 구분 | 정상 | 불량 | 합계 |
|---|---:|---:|---:|
| 공식 train | 2,085 | 246 | 2,331 |
| ↳ 모델 학습 | 1,668 | 197 | 1,865 |
| ↳ 검증 | 417 | 49 | 466 |
| 공식 test (보존) | 894 | 110 | 1,004 |
| 별도: 공식 목록 밖 복사본 | 0 | 1 | 1 |

[manifest.csv](manifest.csv)에는 제외 행까지 담는다. `source_id`는 고정한 배포본, `part_id`는 공식 샘플 번호(제품·코일 식별자가 아님), `group_id`는 같은 크기·RGB 픽셀 내용의 회전/반사 동등류다. 파일 SHA-256, 디코딩한 RGB SHA-256, 정사각 격자의 8가지 회전·반사 중 최소 SHA-256을 검사했다. 공식 train/test 간 겹침은 이 범위에서 0개다. 충돌 발생 시 공식 test는 그대로 두고 train 쪽을 제외한다. 같은 영상에 정답 충돌이 있으면 중단한다. 학습/검증은 공식 train 안에서 정상/불량별로 그룹을 `sha256("42:" + group_id)` 오름차순 정렬한 뒤 각각 20% 반올림을 검증에 할당했다. 하나의 원본에서 만든 타일·교란 입력은 원본의 분할을 따른다. **제품·코일·촬영 배치 정보는 제공되지 않았다.** 이 그룹은 다른 촬영분의 동일 실물이나 임의 크롭·리사이즈를 식별한다는 보장이 없다.

[학습 사례의 원본과 마스크를 나란히 보기](train_pair_review.png). `ksdd2-10000`은 빈 마스크, `ksdd2-10021`은 왼쪽 상단의 표면 흔적 3,346픽셀, `ksdd2-10045`는 오른쪽 하단의 작은 결함 23픽셀(7×5 경계 상자)이다. 크기는 원본 폭 184–241, 높이 597–665픽셀. 학습 불량 197장의 결함 픽셀 수 중앙값은 2,293개이고 가장 작은 결함은 영상의 0.017%뿐이다. [audit.json](audit.json)에 분할별 개수·분위수·해시를 기록했다.

## 전체 복원과 검증

프로젝트 루트에서 Python 3.11 이상 및 Pillow로 실행한다. 아래 ZIP 경로는 프로젝트 외부 예시이며 Windows에서도 쓸 수 있다. 시연본에 담은 특정 분할의 유효성 검사와 전체 복원을 구분한다.

```console
python -m src.inspection_data verify --split validation
python -m src.inspection_data fetch --archive ../KolektorSDD2.zip
python -m src.inspection_data index --archive ../KolektorSDD2.zip
python -m src.inspection_data verify --split all
```

`fetch`는 원본 바이트 수·SHA-256을 확인한다. `index`는 확인된 공식 ZIP만 열어 원본을 복원하고 정답·분할·인덱스를 다시 생성한다. 시연본에 이미 있는 파일은 같은 SHA-256이면 그대로 두고, 원본과 다르면 검증된 공식 바이트로 복원한다. `verify`는 동결한 manifest의 SHA-256과 요청한 분할의 모든 원본/마스크 해시를 다시 비교한다. 공식 test의 주석 및 무결성 확인은 모델 성능 평가나 예측 실행이 아니다.

동결 manifest SHA-256: `8c85620d838a4916afc94c365593bf351e09588d65b733f8bbc96e67d44faa68`. 전체 복원에는 원본 ZIP과 압축을 푼 PNG를 위해 약 1.8 GB의 여유 공간을 권장한다.
