## 001_모델튜닝챌린지_RF_하이퍼파라미터_최적화 
- 모델 튜닝: GridSearchCV, RandomnizedSearchCV, Optuna => Optuna(현재까지 나온 베이지안 방식들 서포트함) 성능 향상에 도움줌
- 샘플링: recall 높이려고 오버샘플링, 언더샘플링, 테스트 코드에서 못맞추는 데이터 뽑아서 오버 샘플링등 다양한 방법 시도 => 학습데이터 과적함(acc 96%), 뉴 데이터 수집아니면 크게 의미 없음
- 이상치 제거: 데이터 plt로 그렸을때 이상치 있어서 제거하였으나 학습데이터에 과적합됨
- 수치형 컬럼 scaler 적용 => 성능향상에 도움됨
- 데이터 수량이 적어서 그런지 n_estimators 높게 가져가면 안됬음, ROC AUC 점수도 0.8 부근이 실제 테스트 데이터에 적합했음
- 결과: 상위 29%
