import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


def result_test(best_params):
       data = pd.read_csv('./open/train.csv')

       # person_id 컬럼 제거
       X_train = data.drop(['person_id', 'login'], axis=1) # 'phone_rat', 'sub_size' 핏쳐 중요도로 뺌
       y_train = data['login']
       model = RandomForestClassifier(
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
       model.fit(X_train, y_train)
       y_pred = model.predict_proba(X_train)[:, 1]
       y_pred = np.where(y_pred > 0.5, 1 , 0)


       roc_auc = roc_auc_score(y_train, y_pred)
       print(f"ROC AUC Score: {roc_auc}")

       feature_name=['Sex', 'past_login_total', 'past_1_month_login', 'past_1_week_login',
              'sub_size', 'email_type', 'phone_rat', 'apple_rat']

       plt.figure(figsize=(100, 80))
       cm = confusion_matrix(y_train, y_pred)
       print(classification_report(y_train, y_pred))

       # 혼동 행렬 시각화
       plt.figure(figsize=(10, 7))
       sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['log_o', 'log_x'], yticklabels=['log_o', 'log_x'])
       plt.xlabel('Predicted')
       plt.ylabel('True')
       plt.title('Confusion Matrix')
       plt.show()


       fp_indices = [i for i, (actual, pred) in enumerate(zip(y_train, y_pred)) if actual == 1 and pred == 0]
       test_indices = X_train.index
       fp_df = data.loc[test_indices[fp_indices]]
       print("False Positive:", fp_indices)
       print(fp_df)

       fn_indices = [i for i, (actual, pred) in enumerate(zip(y_train, y_pred)) if actual == 0 and pred == 1]
       test_indices = X_train.index
       fn_df = data.loc[test_indices[fn_indices]]
       print("False Negative:", fn_indices)
       print(fn_df)


# best_params = {'bootstrap': False, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 0.05, 'n_estimators': 50}
# best_params = {'bootstrap': False, 'max_depth': 14, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 0.05, 'n_estimators': 59}
# best_params = {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 14, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 0.05, 'n_estimators': 54}
# best_params = {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 14, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 0.05, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 44}
# best_params = {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 13, 'max_features': 'log2', 'max_leaf_nodes': 50, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 0.05, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 41}
# best_params ={'n_estimators': 108, 'min_weight_fraction_leaf': 0.005, 'min_samples_split': 0.04, 'min_samples_leaf': 1, 'min_impurity_decrease': 0.0, 'max_leaf_nodes': 200, 'max_features': 'log2', 'max_depth': 13, 'criterion': 'entropy', 'bootstrap': True}
# best_params = {'n_estimators': 181, 'min_weight_fraction_leaf': 0.0, 'min_samples_split': 0.04, 'min_samples_leaf': 1, 'min_impurity_decrease': 0.0, 'max_leaf_nodes': 200, 'max_features': 'log2', 'max_depth': 10, 'criterion': 'entropy', 'bootstrap': True}

# best_params = {'n_estimators': 73, 'min_weight_fraction_leaf': 0.0, 'min_samples_split': 0.04, 'min_samples_leaf': 1, 'min_impurity_decrease': 0.0, 'max_leaf_nodes': None, 'max_features': 'sqrt', 'max_depth': 61, 'criterion': 'entropy', 'bootstrap': False}
# best_params = {'n_estimators': 462, 'min_weight_fraction_leaf': 0.0, 'min_samples_split': 0.01, 'min_samples_leaf': 1, 'min_impurity_decrease': 0.0, 'max_leaf_nodes': 50, 'max_features': 'sqrt', 'max_depth': 207, 'criterion': 'gini', 'bootstrap': True}
# best_params = {'bootstrap': False, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 0.01, 'n_estimators': 45}
#
# best_params = {'bootstrap': True, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 0.01, 'n_estimators': 38}

# result_test(best_params)
k=10