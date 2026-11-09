# G7 + GitHub dohyeon 브랜치: 통합 선택 근거

확인한 GitHub URL: https://github.com/tmdgns104/posco_hackerton/tree/dohyeon

## 확인한 공개 코드
- `SlagWatch/app_demo.py`: 검증 워크스페이스, G1 사례, 모델 등록/교체, 로그인 및 감사 기능에 대한 내비게이션
- `SlagWatch/src/demo_ui.py`: 동결된 G1 결과, 오류 변화, 사례 재현 화면
- `SlagWatch/src/inspection_ui.py`: 동일 샘플과 정답에서 기존 모델/후보 모델의 FN/FP/새 오류 비교
- `SlagWatch/src/p1_tabular_ui.py`: 모델 등록/교체 실험 (단, 업로드 joblib 실행 위험)
- `SlagWatch/src/security.py`: 계정·감사 및 외부 기록에 관한 구현
- `README.md`: G1/NEU/K0·K1/원본 데이터 보존 계획

## G8에서 채택한 장점
1. 시작화면을 '오류 복구와 신규 오류' 중심으로 단순화.
2. K0/K1의 **오경보와 미검출을 따로 평가**하고 기존/후보 비교를 동일 3,262회 판정으로 검증.
3. G1 사례 ZIP으로 재검증할 수 있는 기존 기능을 간편 UI에서 노출.
4. G7의 OOF 동등 구간 G5/G6 비교, 무작위 vs RF 능동 탐색을 유지.
5. 평가 CSV·원본 모델·가중치는 변경하지 않음.
6. 임의 업로드 `.joblib` 실행 및 화면의 생산 설비 모델 변경 버튼은 기본적으로 제외.

## 미통합 및 제한
- GitHub 원격 저장소의 **전체 파일 트리/체크포인트를 자동 다운로드하여 바이트 단위 비교하거나 병합한 것은 아님**. 확인 가능한 Raw 공개 파일과 사용자 제공 G7 ZIP에 한정.
- GitHub `dohyeon` 브랜치의 로그인/원격 감사/모델 등록 코드를 그대로 복사하지 않았고 새 인증 체계를 구현한 것으로 주장하지 않음.
- GitHub에 있는 KSDD2 원본 이미지 전체는 이번 G7 원본에 없으므로 추가하지 않음.
- G7의 모든 14종 모델을 새 메인 화면에서 개별 실시간 평가하는 기능은 제공하지 않으며, 복잡한 이전 연구 화면은 별도 실행기로 남김.
