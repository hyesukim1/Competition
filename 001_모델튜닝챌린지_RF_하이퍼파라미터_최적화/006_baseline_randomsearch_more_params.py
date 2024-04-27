import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from datetime import datetime

data = pd.read_csv('./open/train.csv')


X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']

estimators = [] # the number of trees in our random forest
for x in range(100, 300, 10):
    estimators.append(int(x))

max_depth = [None]
for x in range(10, 100, 10):
    max_depth.append(int(x))

min_samples_split = np.arange(0.01, 1, 0.05).tolist()

min_samples_leaf = [1, 2, 3]

bootstrap = [True, False]

param_search_space = {
    'n_estimators': estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)
grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_search_space, cv=skf, n_jobs=-1, verbose=2, scoring='roc_auc')
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
# Best Parameters: {'n_estimators': 160, 'min_samples_split': 0.21000000000000002, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 40, 'bootstrap': True}
# Best Score: 0.8155998011048775

submit = pd.read_csv('./open/sample_submission.csv')


for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'randomsearch_more_prams_'+'.csv'
submit.to_csv('./result/'+file_name, index=False)