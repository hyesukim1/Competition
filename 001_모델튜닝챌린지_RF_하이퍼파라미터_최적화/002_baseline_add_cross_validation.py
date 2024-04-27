import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from datetime import datetime
from test import result_test


data = pd.read_csv('./open/train.csv')

# person_id 컬럼 제거
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [10, 50, 100, 150, 200, 250, 300],
    'max_depth': [None, 10, 30, 50, 100],
    'min_samples_split': [0.01, 0.05, 0.1],
    'max_features': ['log2', 'sqrt'],
}
# kfold = KFold(n_splits=20, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=skf, n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
result_test(best_params)


submit = pd.read_csv('./open/sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'skf_20'+'.csv'
submit.to_csv('./result/'+file_name, index=False)

'''
cross_validation은 둘중에 암거나 써도 크게 차이 없는 듯

kfold:: n_splits=2
Best Parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_split': 0.1, 'n_estimators': 50}
Best Score: 0.8039223600171546
ROC AUC Score: 0.6810622033239496
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      1163
           1       0.86      0.37      0.52       146

    accuracy                           0.92      1309
   macro avg       0.89      0.68      0.74      1309
weighted avg       0.92      0.92      0.91      1309

False Positive: [0, 6, 7, 11, 17, 23, 26, 43, 48, 50, 52, 60, 74, 91, 100, 101, 102, 110, 114, 119, 125, 133, 134, 135, 139, 146, 157, 159, 165, 170, 179, 180, 181, 193, 200, 203, 204, 209, 219, 224, 228, 229, 244, 264, 266, 267, 271, 293, 294, 297, 308, 310, 332, 350, 355, 362, 378, 393, 398, 405, 414, 429, 452, 457, 459, 531, 533, 594, 595, 606, 610, 621, 624, 643, 655, 677, 682, 704, 839, 865, 882, 891, 896, 905, 919, 963, 987, 988, 1015, 1138, 1172, 1204]

kfold:: n_splits=20
Best Parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_split': 0.01, 'n_estimators': 10}
Best Score: 0.827445470371168
ROC AUC Score: 0.7200084806652611
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      1163
           1       0.82      0.45      0.58       146

    accuracy                           0.93      1309
   macro avg       0.88      0.72      0.77      1309
weighted avg       0.92      0.93      0.92      1309

skf =2
Best Parameters: {'max_depth': 10, 'max_features': 'log2', 'min_samples_split': 0.05, 'n_estimators': 50}
Best Score: 0.8285717870979551
ROC AUC Score: 0.6939009882330769
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      1163
           1       0.84      0.40      0.54       146

    accuracy                           0.92      1309
   macro avg       0.88      0.69      0.75      1309
weighted avg       0.92      0.92      0.91      1309

skf = 20

Best Parameters: {'max_depth': None, 'max_features': 'log2', 'min_samples_split': 0.05, 'n_estimators': 10}
Best Score: 0.8055313047925191
ROC AUC Score: 0.6900464080849009
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      1163
           1       0.83      0.39      0.53       146

    accuracy                           0.92      1309
   macro avg       0.88      0.69      0.74      1309
weighted avg       0.92      0.92      0.91      1309

'''