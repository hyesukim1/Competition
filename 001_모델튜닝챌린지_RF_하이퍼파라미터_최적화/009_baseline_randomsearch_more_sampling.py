import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from datetime import datetime
from imblearn.over_sampling import SMOTE
from test import result_test
from sklearn.utils import resample

data = pd.read_csv('./open/train.csv')

# False Positive ind
FP_ind = [0, 6, 7, 11, 23, 26, 50, 52, 60, 100, 101, 102, 114, 119, 133, 134, 135, 139, 165, 179, 200, 204, 209, 219, 224, 228, 229, 271, 293, 294, 297, 308, 310, 332, 362, 378, 393, 398, 405, 414, 429, 452, 531, 594, 595, 606, 610, 621, 624, 643, 655, 677, 704, 839, 865, 882, 891, 896, 905, 919, 963, 987, 988, 1015, 1138, 1172, 1204]
specific_samples = data[data['person_id'].isin(FP_ind)]

upsampling_df = resample(specific_samples, replace=True, n_samples=len(FP_ind)*1000, random_state=0)
augmented_df = pd.concat([data, upsampling_df ], ignore_index=True)
X_train1 = augmented_df.drop(['person_id', 'login'], axis=1)
y_train1 = augmented_df['login']

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train1, y_train1)


estimators = [] # the number of trees in our random forest
for x in range(49, 100):
    estimators.append(int(x))

min_samples_split = np.arange(0.001, 0.01, 0.001).tolist()

param_search_space = {
    'n_estimators': estimators,
    'min_samples_split': min_samples_split,
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [1],
    'bootstrap': [False]
}

skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
rf = RandomForestClassifier(warm_start=True, max_features=None, random_state=42)
grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_search_space, cv=skf, n_jobs=-1, verbose=2, scoring='roc_auc')
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

# Best Parameters: {'bootstrap': True, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 0.01, 'n_estimators': 38}
# Best Score: 0.9998321974200198
# ROC AUC Score: 0.7290074088034018

'''
Best Parameters: {'n_estimators': 70, 'min_samples_split': 0.004, 'min_samples_leaf': 1, 'max_features': 'log2', 'bootstrap': False}
Best Score: 0.9998565536544313
ROC AUC Score: 0.8343266705143759
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      1163
           1       0.97      0.67      0.79       146

    accuracy                           0.96      1309
   macro avg       0.97      0.83      0.89      1309
weighted avg       0.96      0.96      0.96      1309

'''