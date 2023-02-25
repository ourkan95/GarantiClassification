import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report 

target = 'moved_after_2019'

def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

#MODELS
df = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\train_final.csv')
df_test = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\test_final.csv')
df_subm = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\garanti-bbva-data-camp\submission.csv')
X, y = df.drop(columns=['moved_after_2019']), df['moved_after_2019']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=17)


# TEK GRID
"""
ParamsXGB = {
    # 'gamma':[i/10.0 for i in range(0,3)],
    # 'subsample':[0.8,0.9,1],
    # 'colsample_bytree':[0.8,0.9,1],
    'reg_alpha':[0,0.3,0.5 ],
    'reg_lambda':[0,0.3,0.5],
        # 'max_depth': [13, 15, 17],
        # 'n_estimators': [1500, 2500, 5000],
        # 'learning_rate': [0.005, 0.01, 0.02]
}

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

gridxg = GridSearchCV(estimator = XGBClassifier(tree_method='gpu_hist', gpu_id=0, learning_rate = 0.01, max_depth = 15, n_estimators = 2500,
                      min_child_weight = 1,  objective= 'binary:logistic', scale_pos_weight=1, subsample = 1, colsample_bytree = 1 
                      , n_jobs = -1, seed=27),
                      param_grid = ParamsXGB, scoring='accuracy', cv=kfold, verbose = 3)

gridxg.fit(X_train, y_train)

print(gridxg.best_params_)
print(gridxg.best_score_)
print(gridxg.cv_results_)
# 0.7364804785385175 {'learning_rate': 0.01, 'max_depth': 11, 'min_child_weight': 1, 'n_estimators': 5000}
"""
xgbfinal = XGBClassifier(tree_method='gpu_hist', gpu_id=0, learning_rate = 0.01, max_depth = 11, n_estimators = 5000,
                      min_child_weight = 1,  objective= 'binary:logistic', subsample = 1, colsample_bytree = 1, reg_alpha = 0.3, reg_lambda =0.3,
                      scale_pos_weight=1, n_jobs = -1, seed=27)
xgbfinal.fit(X,y)



#RF

from sklearn.ensemble import RandomForestClassifier
"""
ParamRF = {
            'n_estimators':  [800,900, 1000],
            'criterion' : ['gini'],
            'class_weight': ['balanced_subsample']
            }
kfold = StratifiedKFold(shuffle=True, random_state=42)
gridrf = GridSearchCV(estimator = RandomForestClassifier(n_jobs=(-1),verbose = 0),
                      param_grid = ParamRF, scoring='accuracy', cv=kfold, verbose = 3)

gridrf.fit(X_train,y_train)

print(gridrf.best_params_)
print(gridrf.best_score_)
print(gridrf.cv_results_)
"""
rffinal = RandomForestClassifier(n_estimators=800, class_weight='balanced_subsample', random_state=42)

rffinal.fit(X,y)

# LIGHTGBM
"""
lightgbm_params = {
                    'reg_lambda': [0,0.5,1],
                    'reg_alpha': [0,0.5,1]                                    
                   }

kfold = StratifiedKFold(shuffle=True, random_state=42)
gridLGBM = GridSearchCV(estimator = LGBMClassifier(seed = 27 ,force_col_wise=True , learning_rate = 0.1, n_estimators = 1800
                                                   ,verbose = 0),
                      param_grid = lightgbm_params, scoring='accuracy', cv=kfold, verbose = 3)

gridLGBM.fit(X_train, y_train)

print(gridLGBM.best_params_)
print(gridLGBM.best_score_)
print(gridLGBM.cv_results_)
"""
lgbfinal = LGBMClassifier(seed =27 ,force_col_wise=True , learning_rate = 0.1, n_estimators = 1800, reg_alpha = 0, reg_lambda = 0)
lgbfinal.fit(X,y)

# CATBOOST
"""
cat_params = {
        # 'iterations': [325,350,375],
        # 'learning_rate': [0.03, 0.05, 0.07],
        # 'max_depth': [12,13,14],
        'l2_leaf_reg': [0, 0.5, 1]
        }

valid = Pool(X_test.iloc[0:1500,:], label=y_test.iloc[0:1500])

kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
gridCAT = GridSearchCV(estimator = CatBoostClassifier(random_seed=23, verbose = 0,iterations = 350, learning_rate = 0.05, max_depth=13,
                loss_function="CrossEntropy", eval_metric="CrossEntropy",  task_type="GPU"),
                      param_grid = cat_params, scoring='accuracy', cv=kfold, verbose = 3)

gridCAT.fit(X_train, y_train)
print(gridCAT.best_params_)
print(gridCAT.best_score_)
print(gridCAT.cv_results_)
"""

catfinal = CatBoostClassifier(random_seed=23, verbose = 0, iterations = 350, learning_rate = 0.05, max_depth=13, l2_leaf_reg = 0.5,
                loss_function="CrossEntropy", eval_metric="CrossEntropy",  task_type="GPU")
catfinal.fit(X,y)


print(accuracy_score(y, catfinal.predict(X)))
print(accuracy_score(y, xgbfinal.predict(X)))
# print(accuracy_score(y, lgbfinal.predict(X)))
print(accuracy_score(y, rffinal.predict(X)))


# ('LightGBM', lgbfinal),

voting_clf = VotingClassifier(estimators=[('RF', rffinal), ('XGB', xgbfinal)],voting='soft',n_jobs=(-1),verbose = True).fit(X, y)

# print(accuracy_score(y, voting_clf.predict(X)))
# 0.9915690601482487, rfxgcat 0.9951149587883589, rfxg 0.9971519643901243



#Prediction
df_subm['moved_after_2019'] = voting_clf.predict(df_test)

df_subm['moved_after_2019'] = df_subm['moved_after_2019'].astype(int)
df_subm['moved_after_2019'].value_counts()

df_subm.to_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\submission00rfxg2.csv', index = False)


