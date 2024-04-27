import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from datetime import datetime
from imblearn.over_sampling import SMOTE
from test import result_test

data = pd.read_csv('./open/train.csv')

# SMOTE
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']
smote = SMOTE(random_state=0)
X_train,y_train = smote.fit_resample(X_train, y_train)

estimators = [] # the number of trees in our random forest
for x in range(10, 50):
    estimators.append(int(x))

min_samples_split = np.arange(0.01, 1, 0.01).tolist()

param_search_space = {
    'n_estimators': estimators,
    'min_samples_split': min_samples_split,
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [1],
    'bootstrap': [True, False]
}

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
rf = RandomForestClassifier(warm_start=True, max_features=None, random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=skf, n_jobs=-1, verbose=2, scoring='roc_auc')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
result_test(best_params)
submit = pd.read_csv('./open/sample_submission.csv')


for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'gridsearch_detail'+'.csv'
submit.to_csv('./result/'+file_name, index=False)

# Best Parameters: {'bootstrap': False, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 0.01, 'n_estimators': 45}
# Best Score: 0.9430978701255686
