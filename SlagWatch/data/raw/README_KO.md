# Steel Slag Flow Dataset (SSFD) V3 — 공개 데이터 확보본

확인·다운로드 일자: 2026-11-07 (한국 시간 기준)

## 원출처 및 저작자

Sehri, Mert; Ayres, Lucas Mantuan; Boldt, Francisco de Assis; Dumond, Patrick; Cuadros, Marco Antonio de Souza Leite (2026), “Steel Slag Flow Dataset (SSFD)”, Mendeley Data, V3.

- 데이터 원문: https://data.mendeley.com/datasets/mjnpsbs6fs/3
- DOI: https://doi.org/10.17632/mjnpsbs6fs.3
- 데이터 설명 논문: https://pmc.ncbi.nlm.nih.gov/articles/PMC12495644/
- 라이선스: Creative Commons Attribution 4.0 International (CC BY 4.0)
- 라이선스 원문: https://creativecommons.org/licenses/by/4.0/

공식 저장소의 Download All 기능으로 내려받은 자료다. 이 묶음에는 원본 CSV 48개와 JPG 3개를 내용 변경 없이 수록하고, 이 설명과 파일 확인용 manifest.json을 추가했다. 원저작자와 원출처를 표시하고, 파생 데이터를 만들 때 변경 내용을 명시해야 한다.

## 데이터 구성

- E-1.csv ~ E-16.csv: 초기 슬래그가 흐르지 않는 상태
- B-1.csv ~ B-16.csv: 슬래그 흐름 이전 상태
- S-1.csv ~ S-16.csv: 슬래그가 흐르는 상태
- 컬럼: time, x, y, z
- 공식 설명: 6,400Hz, 파일별 약 5초, 3축 진동
- 파일 번호: 동일 발생/조건의 그룹을 나타내므로 학습·평가를 이 번호 단위로 분리한다.

## 프로젝트에 사용할 범위

공개 진동 기록으로 슬래그 유동 상태를 분류하고, 모의 잡음·누락·값 고정을 주입해 모델과 입력 상태 처리의 변화를 비교하는 개발 실험에 사용할 수 있다.

이 데이터는 포스코의 전로 출강 영상이나 실제 출강 종료 제어 로그가 아니다. 데이터 설명 논문은 용강을 주형으로 붓는 과정에서 유동 제어 축에 부착한 센서로 기록했다고 설명한다. 라벨별로 잘라 놓은 기록이므로 전로의 최적 종료 시각이나 슬래그 유동 시작 전 잔여 시간을 정답으로 사용하지 않는다.

E/B/S 파일을 이어 붙여 데모를 만들 경우 실제 연속 기록과 구분해서 모의 재생이라고 표시한다. 창으로 잘라 만든 입력 수를 독립적인 공정 사례 수로 해석하지 않는다. 센서 이상을 추가한 데이터는 실제 수집된 장애 사례가 아닌 합성 시험 입력이다.

## 구현 계획

1. 원본 파일 구조와 결측·시간 순서 확인.
2. 파일 번호로 학습·조정·최종 평가 그룹 분리.
3. 최근 관측 구간의 RMS·표준편차·최대 진폭 등 특징 계산.
4. 우선 S 대 E/B의 이진 분류 모델과 기준 규칙 비교.
5. 원본 기록 재생 화면과 합성 입력 이상 시험 기능 구성.
6. 슬래그 미검출·오경보·그룹별 성능을 기록.

이 데이터 묶음은 자료 확보 결과이며, 완성된 프로젝트나 현장 성능 검증 결과를 포함하지 않는다.
