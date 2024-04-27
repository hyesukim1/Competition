import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from datetime import datetime
from imblearn.over_sampling import SMOTE
from test import result_test
from sklearn.utils import resample
from skopt import BayesSearchCV

data = pd.read_csv('./open/train.csv')

# False Positive ind
FP_ind = [0, 6, 7, 11, 23, 26, 50, 52, 60, 100, 101, 102, 114, 119, 133, 134, 135, 139, 165, 179, 200, 204, 209, 219, 224, 228, 229, 271, 293, 294, 297, 308, 310, 332, 362, 378, 393, 398, 405, 414, 429, 452, 531, 594, 595, 606, 610, 621, 624, 643, 655, 677, 704, 839, 865, 882, 891, 896, 905, 919, 963, 987, 988, 1015, 1138, 1172, 1204]
specific_samples = data[data['person_id'].isin(FP_ind)]

upsampling_df = resample(specific_samples, replace=True, n_samples=len(FP_ind)*500, random_state=0)
augmented_df = pd.concat([data, upsampling_df ], ignore_index=True)
X_train1 = augmented_df.drop(['person_id', 'login'], axis=1)
y_train1 = augmented_df['login']

X_train1 = data.drop(['person_id', 'login'], axis=1)
y_train1 = data['login']
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train1, y_train1)

min_samples_split = np.arange(0.001, 0.05, 0.001).tolist()
estimators = []

for x in range(1, 30):
    estimators.append(int(x))

param_search_space = {
    'n_estimators': estimators,  # 10에서 1000까지의 정수
    'criterion': ['gini'],
    # 'max_depth': [1, 2, 3, 4, 5],  # None을 포함한 1부터 50까지의 정수
    'min_samples_split': np.arange(0.001, 0.05, 0.001).tolist(),
    'min_samples_leaf': [1],
    'min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.1).tolist(),
    'max_features': ['log2', 'sqrt'],
    # 'max_leaf_nodes': list(range(1, 51)),
    'min_impurity_decrease': np.arange(0.0, 0.5, 0.1).tolist(),
    'bootstrap': [False]
}

skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
rf = RandomForestClassifier(warm_start=True, max_features=None, random_state=42)
# grid_search = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=skf, n_jobs=-1, verbose=2, scoring='roc_auc')
# grid_search.fit(X_train, y_train)


bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42), #class_weight='balanced'
    search_spaces=param_search_space,
    n_iter=32,
    cv=skf,
    n_jobs=-1,
    random_state=0,
    verbose=2
)
bayes_search.fit(X_train, y_train)

best_params = bayes_search.best_params_
best_score = bayes_search.best_score_
best_model = bayes_search.best_estimator_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

result_test(best_params)

submit = pd.read_csv('./open/sample_submission.csv')


for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'bayssearch_allOfparams_1'+'.csv'
submit.to_csv('./result/'+file_name, index=False)



'''
param_search_space = {
    'n_estimators': estimators,  # 10에서 1000까지의 정수
    'criterion': ['gini'],
    # 'max_depth': list(range(1, 51)),  # None을 포함한 1부터 50까지의 정수
    'min_samples_split': min_samples_split,
    # 'min_samples_leaf': np.arange(0.1, 0.5, 0.1).tolist(),
    # 'min_weight_fraction_leaf': np.arange(0.0, 0.5, 0.1).tolist(),
    'max_features': ['log2'] + list(range(1, 31)),
    # 'max_leaf_nodes': list(range(1, 51)),
    # 'min_impurity_decrease': np.arange(0.0, 0.5, 6).tolist(),
    'bootstrap': [False]
}
Best Parameters: {'n_estimators': 58, 'min_samples_split': 0.028, 'max_features': 2, 'criterion': 'gini', 'bootstrap': False}
Best Score: 0.9998060572184876
ROC AUC Score: 0.7247229060412961
              precision    recall  f1-score   support

           0       0.94      0.99      0.96      1163
           1       0.86      0.46      0.60       146

    accuracy                           0.93      1309
   macro avg       0.90      0.72      0.78      1309
weighted avg       0.93      0.93      0.92      1309

# mean_sample_leaf 추가시 성 능하락 
Best Parameters: {'n_estimators': 69, 'min_samples_split': 0.032, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 2, 'criterion': 'gini', 'bootstrap': False}
Best Score: 0.9896487946666165
ROC AUC Score: 0.656659677970294
              precision    recall  f1-score   support

           0       0.92      0.99      0.95      1163
           1       0.82      0.32      0.46       146

    accuracy                           0.92      1309
   macro avg       0.87      0.66      0.71      1309
weighted avg       0.91      0.92      0.90      1309

#min_weight_fraction_leaf 있을수 록 성능하.락
Best Parameters: {'n_estimators': 86, 'min_weight_fraction_leaf': 0.0, 'min_samples_split': 0.034, 'max_features': 2, 'max_depth': 2, 'criterion': 'gini', 'bootstrap': False}
Best Score: 0.9815440998394832
ROC AUC Score: 0.6215679807771588
              precision    recall  f1-score   support

           0       0.91      1.00      0.95      1163
           1       0.90      0.25      0.39       146

    accuracy                           0.91      1309
   macro avg       0.91      0.62      0.67      1309
weighted avg       0.91      0.91      0.89      1309

# max leaf node 추가하.면..성능하락
Best Parameters: {'n_estimators': 85, 'min_samples_split': 0.045, 'max_leaf_nodes': 7, 'max_depth': 2, 'criterion': 'gini', 'bootstrap': False}
Best Score: 0.9225253762776765
ROC AUC Score: 0.6151485883225951
              precision    recall  f1-score   support

           0       0.91      1.00      0.95      1163
           1       0.92      0.23      0.37       146

    accuracy                           0.91      1309
   macro avg       0.92      0.62      0.66      1309
weighted avg       0.91      0.91      0.89      1309

# min_impurity_decrease 
Best Parameters: {'n_estimators': 54, 'min_samples_split': 0.010000000000000002, 'min_impurity_decrease': 0.0, 'max_depth': 2, 'criterion': 'gini', 'bootstrap': False}
Best Score: 0.9225253762776765
ROC AUC Score: 0.6190031684707712
              precision    recall  f1-score   support

           0       0.91      1.00      0.95      1163
           1       0.95      0.24      0.38       146

    accuracy                           0.91      1309
   macro avg       0.93      0.62      0.67      1309
weighted avg       0.92      0.91      0.89      1309

### 줄일수록 좋음
param_search_space = {
    'n_estimators': estimators,  # 10에서 1000까지의 정수
    'criterion': ['gini'],
    'min_samples_leaf': [1],
    'max_features': ['log2', 'sqrt'],
    'bootstrap': [False]
}

Best Parameters: {'n_estimators': 96, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'criterion': 'gini', 'bootstrap': False}
Best Score: 0.9460647634978537
ROC AUC Score: 0.8416059081967986
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      1163
           1       0.98      0.68      0.81       146

    accuracy                           0.96      1309
   macro avg       0.97      0.84      0.89      1309
weighted avg       0.96      0.96      0.96      1309

Best Parameters: OrderedDict([('bootstrap', False), ('criterion', 'gini'), ('max_features', 'log2'), ('min_impurity_decrease', 0.0), ('min_samples_leaf', 1), ('min_samples_split', 0.004), ('min_weight_fraction_leaf', 0.0), ('n_estimators', 25)])
Best Score: 0.8925361037430003
ROC AUC Score: 0.8338967479004464
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      1163
           1       0.96      0.67      0.79       146

    accuracy                           0.96      1309
   macro avg       0.96      0.83      0.88      1309
weighted avg       0.96      0.96      0.96      1309

'''