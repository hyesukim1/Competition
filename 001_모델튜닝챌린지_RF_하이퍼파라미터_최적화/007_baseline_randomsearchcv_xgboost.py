import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from datetime import datetime
import xgboost as xgb
from imblearn.over_sampling import SMOTE

data = pd.read_csv('./open/train.csv')

# SMOTE
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']
smote = SMOTE(random_state=0)
X_train,y_train = smote.fit_resample(X_train, y_train)

estimators = [] # the number of trees in our random forest
for x in range(100, 300, 10):
    estimators.append(int(x))

max_depth = [None]
for x in range(10, 100, 10):
    max_depth.append(int(x))

min_samples_split = np.arange(0.01, 1, 0.05).tolist()

min_samples_leaf = [1, 2, 3]

bootstrap = [True, False]

param_grid = {
    'min_child_weight':[1,3,5],
    'gamma':[0,1,2,3],
    'nthread':[4],
    'colsample_bytree':[0.5,0.8],
    'colsample_bylevel':[0.9],
    'n_estimators': estimators,
    'max_depth': max_depth,
    'learning_rate': [0.01, 0.1, 0.5]
}

xgb_classifier = xgb.XGBClassifier(eval_metric='logloss')

grid_search = RandomizedSearchCV(estimator=xgb_classifier, param_distributions=param_grid, scoring='roc_auc', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

submit = pd.read_csv('./open/sample_submission.csv')


for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'randomsearch_xgboost'+'.csv'
submit.to_csv('./result/'+file_name, index=False)

# Best Parameters: {'nthread': 4, 'n_estimators': 260, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.9}
# Best Score: 0.9258313346642268
