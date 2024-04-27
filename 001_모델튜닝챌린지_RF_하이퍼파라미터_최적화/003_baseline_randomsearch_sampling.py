import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from datetime import datetime
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from test import result_test
data = pd.read_csv('./open/train.csv')

# SMOTE
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']
smote = SMOTE(random_state=0)
X_train,y_train = smote.fit_resample(X_train, y_train)

# resample
# num_0 = len(data[data['login']==0])
# num_1 = len(data[data['login']==1])
# oversampled_data = pd.concat([ data[data['login']==0] , data[data['login']==1].sample(num_0, replace=True) ])
#
# X_train = oversampled_data.drop(['person_id', 'login'], axis=1)
# y_train = oversampled_data['login']

estimators = [] # the number of trees in our random forest
for x in range(49, 80):
    estimators.append(int(x))

max_depth = [None]
# for x in range(10, 100, 10):
#     max_depth.append(int(x))

min_samples_split = np.arange(0.001, 0.05, 0.005).tolist()

min_samples_leaf = [1]

bootstrap = [False]

param_search_space = {
    'n_estimators': estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

kfold = KFold(n_splits=20, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)
grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_search_space, cv=kfold, n_jobs=-1, verbose=2, scoring='roc_auc')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
result_test(best_params)

submit = pd.read_csv('./open/sample_submission.csv')


for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'randomsearch_SMOTE'+'.csv'
submit.to_csv('./result/'+file_name, index=False)
'''
Resample
Best Parameters: {'n_estimators': 72, 'min_samples_split': 0.001, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'bootstrap': False}
Best Score: 0.9520257180564864
ROC AUC Score: 0.8416059081967986
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      1163
           1       0.98      0.68      0.81       146

    accuracy                           0.96      1309
   macro avg       0.97      0.84      0.89      1309
weighted avg       0.96      0.96      0.96      1309

SMOTE
Best Parameters: {'n_estimators': 78, 'min_samples_split': 0.001, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
Best Score: 0.9485363548042456
ROC AUC Score: 0.8416059081967986
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      1163
           1       0.98      0.68      0.81       146

    accuracy                           0.96      1309
   macro avg       0.97      0.84      0.89      1309
weighted avg       0.96      0.96      0.96      1309

'''

