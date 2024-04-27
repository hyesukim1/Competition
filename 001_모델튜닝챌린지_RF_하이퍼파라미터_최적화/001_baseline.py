import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from test import result_test

data = pd.read_csv('./open/train.csv')

# person_id 컬럼 제거
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4]
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print('best_params:', best_params)
print('best_score:', best_score)
result_test(best_params)


submit = pd.read_csv('./open/sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit'+'.csv'
submit.to_csv('./result/'+file_name, index=False)

'''
best_params: {'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50}
best_score: 0.7474650215816369
ROC AUC Score: 0.6951907560748655
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      1163
           1       0.88      0.40      0.55       146

    accuracy                           0.93      1309
   macro avg       0.90      0.70      0.75      1309
weighted avg       0.92      0.93      0.91      1309

False Positive: [0, 6, 7, 11, 17, 23, 26, 30, 43, 48, 50, 52, 60, 74, 100, 101, 102, 114, 119, 125, 133, 134, 135, 139, 146, 157, 159, 165, 170, 179, 180, 181, 193, 200, 203, 204, 209, 219, 224, 228, 229, 264, 266, 267, 271, 293, 294, 297, 308, 310, 332, 350, 362, 378, 393, 398, 405, 414, 429, 452, 457, 531, 533, 594, 595, 606, 610, 621, 624, 643, 655, 677, 682, 704, 839, 865, 882, 891, 896, 905, 919, 963, 987, 988, 1015, 1138, 1172, 1204]
False Negative: [20, 24, 37, 55, 67, 105, 108, 171]
'''