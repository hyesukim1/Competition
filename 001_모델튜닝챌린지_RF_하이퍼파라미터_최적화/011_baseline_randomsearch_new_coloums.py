import pandas as pd
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from test import result_test
from skopt import BayesSearchCV
from mycode import *
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler

dt = pd.read_csv('./open/train.csv')

def new_col(df):
    # 1. 로그인 증가율
    df['login_increase_rate'] = (df['past_1_week_login'] / df['past_1_month_login']).replace([np.inf, np.nan], 0)

    # 2. 평균 주간 로그인 횟수
    df['average_weekly_logins'] = (df['past_1_month_login'] / 4).replace([np.inf, np.nan], 0)

    # 3. 장기 대비 단기 로그인 비율
    df['long_vs_short_term_login_ratio'] = (df['past_login_total'] / (df['past_1_month_login'] + df['past_1_week_login'])).replace([np.inf, np.nan], 0)

    # 4. 활동 감소율
    df['activity_decline_rate'] = ((df['past_1_month_login'] - df['past_1_week_login'] * 4) / df['past_1_month_login']).replace([np.inf, np.nan], 0)

    # 5. 정기적인 사용자 여부 판단
    df['is_regular_user'] = (df['past_1_week_login'] >= 3) & (df['past_1_month_login'] >= 12)
    return df

data = new_col(dt)
print(data)

X_train = data.drop(['person_id', 'login'], axis=1) # 'phone_rat', 'sub_size' 핏쳐 중요도로 뺌
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
y_train = data['login']


X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

estimators = [] # the number of trees in our random forest
for x in range(49, 80):
    estimators.append(int(x))

min_samples_split = np.arange(0.001, 0.05, 0.001).tolist()

param_search_space = {
    'n_estimators': estimators,
    'min_samples_split': min_samples_split,
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [1],
    'bootstrap': [True, False]
}

skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

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
feature_importance(X_train, y_train, best_params)

submit = pd.read_csv('./open/sample_submission.csv')


for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

now = datetime.now()
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'new_colums'+'.csv'
submit.to_csv('./result/'+file_name, index=False)

'''
Best Parameters: OrderedDict([('bootstrap', True), ('max_features', 'sqrt'), ('min_samples_leaf', 1), ('min_samples_split', 0.046), ('n_estimators', 62)])
Best Score: 0.7740476190476191
ROC AUC Score: 0.6943309108470064
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      1163
           1       0.85      0.40      0.54       146

    accuracy                           0.93      1309
   macro avg       0.89      0.69      0.75      1309
weighted avg       0.92      0.93      0.91      1309
'''