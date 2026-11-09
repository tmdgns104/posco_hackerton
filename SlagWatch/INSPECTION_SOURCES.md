# 검사 AI 변경 검증의 근거

원문 확인일은 2026-11-09(한국시간)이다.

| 자료 | 발행 시점 | 확인 내용과 구현에 반영한 범위 |
|---|---|---|
| [포스코그룹 뉴스룸 · 포항제철소 현장 주도 스마트 혁신](https://newsroom.posco.com/kr/ai가-불량-먼저-잡고-제품-출하-순서까지-정한다-포항/) | 2026-06-30 | 포항제철소는 후속 공정에서 발견된 결함을 추적하고 제품을 되돌리는 부담과, 이를 줄이기 위해 도입한 조기 선별·선택적 재검사를 소개했다. 이 사례에서 검사 AI가 놓치는 불량과 정상 제품에 대한 경보를 따로 살펴볼 필요성을 도출했다. 기사에 나오는 크레인 출하 최적화와 안전 CCTV의 성과는 별개다. |
| [MVTec AD 2 데이터 설명](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) · [공식 발표](https://www.mvtec.com/knowledge-base/news/article/advancing-anomaly-detection-vand-2025-challenge-at-cvpr-2025) | 데이터 공개 2025-04-01, 설명 기사 2025-05-16 | MVTec은 학습 데이터에 없을 수 있는 여러 조명 조건에서 정상·이상 샘플을 평가한다. 이 설계를 참고해 원본과 합성 밝기·잡음 조건의 성능을 개별 측정한다. 이 프로젝트의 측정 데이터는 KolektorSDD2와 NEU-CLS다. |
| [Breck 외, The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) · [원문 PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/4156.pdf) | IEEE Big Data, 2017 | Model 6은 전체 평균에 가려진 중요 부분집합의 성능을 평가한다. Infra 4는 배포 전 이전 모델의 예측과 비교하는 검사를 다룬다. 같은 입력에 대한 기존·후보 모델의 결과를 짝짓고, 변화가 생긴 조건·사례와 원본 단위 분모를 저장하는 설계에 반영했다. |
| [ViCoS Lab / Kolektor Group, KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | 관련 논문 Božič, Tabernik, Skočaj, *Mixed supervision for surface-defect detection: from weakly to fully supervised learning*, Computers in Industry, 2021. 공식 웹페이지의 별도 발행일은 미표기 | 통제된 산업 환경에서 수집한 부품 표면 이미지와 주석을 사용한다. 원 제작자가 제공한 학습/시험 분할과 픽셀 마스크를 확인해 이미지 단위의 정상/불량 정답을 만든다. 배포물의 데이터와 파생 사례는 저작자표시·비영리·동일조건변경허락에 따른다. |

## 직접 측정한 것

검사 단위는 사진 한 장이다. 한 조건에서 불량을 놓친 수와 정상에 경보를 낸 수, 모델을 교체하며 새로 생긴 오류와 해소한 오류, 저장 입력의 재현 여부를 측정한다. 동일 원본을 여러 조건에서 반복한 횟수도 함께 기록한다. 생산 정지 시간, 재검사 비용, 작업자 안전, 실제 포스코 공정의 불량 유출량은 측정하지 않았다. 합성 조건은 검사 절차를 시험하기 위한 것으로 현장 간섭의 강도나 발생 빈도를 나타내지 않는다.
