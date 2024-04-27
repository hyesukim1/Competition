from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from test import result_test
from skopt import BayesSearchCV
from mycode import *
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv('./open/train.csv')


data = data[data['past_login_total'] < 100]
data = data[data['past_1_month_login'] < 90]
data = data[data['past_1_week_login'] < 17]


fp_ind = [0, 6, 7, 11, 23, 26, 50, 52, 60, 74, 100, 101, 102, 114, 119, 125, 133, 134, 135, 139, 146, 157, 165, 170, 179, 180, 181, 193, 200, 204, 209, 219, 224, 228, 229, 244, 266, 267, 271, 293, 294, 297, 308, 310, 332, 350, 362, 378, 393, 398, 405, 414, 429, 452, 457, 531, 533, 594, 595, 606, 610, 621, 624, 643, 655, 677, 682, 704, 839, 865, 882, 891, 896, 905, 919, 963, 987, 988, 1015, 1138, 1172, 1204]
fp_data = data.loc[fp_ind]
new_df = data[data['login'] == 0]
data = pd.concat([fp_data, new_df], axis=0)

X_train = data.drop(['person_id', 'login'], axis=1) # 'phone_rat', 'sub_size' 핏쳐 중요도로 뺌
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
y_train = data['login']

# X_train, y_train = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

estimators = [] # the number of trees in our random forest
for x in range(49, 100):
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
file_name = now.strftime('%Y-%m-%d')+'_baseline_submit_'+'more_min_samples_split_1'+'.csv'
submit.to_csv('./result/'+file_name, index=False)

# Best Parameters: OrderedDict([('bootstrap', False), ('max_features', 'log2'), ('min_samples_leaf', 1), ('min_samples_split', 0.001), ('n_estimators', 94)])
# Best Score: 0.9193668954996184
# ROC AUC Score: 0.8416059081967986

# recall 값을 높이면 성능이 높아질 줄 알았는데 실제는 그렇지 않았음, 오히려 기법이 추가되지 않아야 성능이 좋음
# 언더샘플링을하면 학습 데이터 성능에 좋아짐 근데 실제 올리는 데이터에는 부정확함(오버 샘플링보다 언더 샘플링이 효과가 좋음)
'''
Best Parameters: OrderedDict([('bootstrap', False), ('max_features', 'log2'), ('min_samples_leaf', 1), ('min_samples_split', 0.013000000000000001), ('n_estimators', 56)])
Best Score: 0.6652777777777777
ROC AUC Score: 0.7461306964746346
              precision    recall  f1-score   support

           0       0.94      0.99      0.97      1163
           1       0.89      0.50      0.64       146

    accuracy                           0.94      1309
   macro avg       0.92      0.75      0.80      1309
weighted avg       0.93      0.94      0.93      1309

'''