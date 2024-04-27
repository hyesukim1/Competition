import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history
from test import result_test
from datetime import datetime

data = pd.read_csv('./open/train.csv')

# 정규화 전에 이상치 제거
# data = data[data['past_login_total'] < 100]
# data = data[data['past_1_month_login'] < 90]
# data = data[data['past_1_week_login'] < 17]

X_train1 = data.drop(['person_id', 'login'], axis=1)
y_train1 = data['login']

scaler = MinMaxScaler()
numeric = X_train1.select_dtypes(exclude='object').columns
print(numeric)
X_train1[numeric] = scaler.fit_transform(X_train1[numeric])

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 5, 50),
        'criterion': 'gini',
        'max_depth': trial.suggest_int(name="max_depth", low=40, high=60, step=1),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.0001, 0.005),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 9, 20),
        # 'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        'max_features': trial.suggest_categorical(name="max_features", choices=['log2', 'sqrt']),
        # 'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 50),
        # 'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.9)
    }

    class_weights = trial.suggest_categorical('class_weights', [None, 'balanced'])

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight=class_weights)

    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True)

    score = cross_val_score(model, X_train1, y_train1, cv=stratified_kfold, scoring='roc_auc').mean()

    return score

sampler = TPESampler(**TPESampler.hyperopt_parameters())
# Optuna 스터디 생성
study = optuna.create_study(direction='maximize', sampler=sampler, study_name='RandomForestOptimization')

# 목적 함수 최적화, n_trials 100이하로 해도 됨
study.optimize(objective, n_trials=500, n_jobs=-1)
best_params = study.best_params


# 최적의 하이퍼파라미터로 모델 생성
best_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    criterion='gini',
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    # min_weight_fraction_leaf=best_params['min_weight_fraction_leaf'],
    max_features=best_params['max_features'],  # float type
    # max_leaf_nodes=best_params['max_leaf_nodes'],
    # min_impurity_decrease=best_params['min_impurity_decrease'],
    random_state=42,
    n_jobs=-1,
    class_weight=best_params['class_weights']
)


submit = pd.read_csv('./open/sample_submission.csv')


for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value
        # print(submit[param])

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'new_randomforest'+'.csv'
submit.to_csv('./result/'+file_name, index=False)

plot_optimization_history(study)
result_test(best_params)


'''
/home/hskim/Desktop/pythonProject/Dacon/result/2024-03-28_baseline_submit_new_randomforest1.csv
ROC AUC Score: 0.8530665850010012
              precision    recall  f1-score   support

           0       0.97      0.95      0.96      1163
           1       0.67      0.75      0.71       146

    accuracy                           0.93      1309
   macro avg       0.82      0.85      0.83      1309
weighted avg       0.93      0.93      0.93      1309

/home/hskim/Desktop/pythonProject/Dacon/result/2024-03-28_baseline_submit_new_randomforest2.csv
ROC AUC Score: 0.8620655131391417
              precision    recall  f1-score   support

           0       0.97      0.96      0.96      1163
           1       0.69      0.77      0.73       146

    accuracy                           0.94      1309
   macro avg       0.83      0.86      0.85      1309
weighted avg       0.94      0.94      0.94      1309

ROC AUC Score: 0.8821982591078811
              precision    recall  f1-score   support
           0       0.97      0.96      0.97      1163
           1       0.73      0.80      0.76       146
    accuracy                           0.94      1309
   macro avg       0.85      0.88      0.87      1309
weighted avg       0.95      0.94      0.95      1309
'''