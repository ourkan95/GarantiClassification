import numpy as np
import pandas as pd


df_education = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\education.csv',index_col=(0))
df_exp = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\exp.csv',index_col=(0))
df_language = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\language.csv',index_col=(0))                           
df_skills = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\skills.csv',index_col=(0))                           
df_train = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\train_users.csv',index_col=(0))                           
df_test = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\test_users.csv',index_col=(0))
df_subm = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\garanti-bbva-data-camp\submission.csv')
                          
df_train[df_education.columns]   = df_education[df_education.columns]
df_train[df_language.columns]  = df_language[df_language.columns]
df_train[df_skills.columns] = df_skills[df_skills.columns]
df_train[df_exp.columns]   = df_exp[df_exp.columns]

df_test[df_education.columns]   = df_education[df_education.columns]
df_test[df_language.columns]  = df_language[df_language.columns]
df_test[df_skills.columns] = df_skills[df_skills.columns]
df_test[df_exp.columns]   = df_exp[df_exp.columns]

cat_cols = [col for col in df_test.columns if df_test[col].dtype == 'object']
num_cols = [col for col in df_test.columns if df_test[col].dtype != 'object']

for col in cat_cols:
    train_cats  = set(df_train[col].unique())
    test_cats   = set(df_test[col].unique())
    common_cats = set.intersection(train_cats, test_cats)
    
    df_train.loc[~df_train[col].isin(common_cats), col] = 'other'
    df_test.loc[~df_test[col].isin(common_cats), col] = 'other'

df_all = pd.concat([df_train, df_test], axis=0)

for col in cat_cols:
    df_all[col] = df_all[col].factorize()[0]

df_all[cat_cols] = df_all[cat_cols].astype('category')
df_all[num_cols] = df_all[num_cols].fillna(-1)

df_train = df_all.loc[df_train.index, df_train.columns]
df_test = df_all.loc[df_test.index, df_test.columns]

df_train.to_csv('train_final.csv',index = False)
df_test.to_csv('test_final.csv', index = False)


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X, y = df_train.drop(columns=['moved_after_2019']), df_train['moved_after_2019']
clf = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=42)
cv  = StratifiedKFold(shuffle=True, random_state=42)

scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f'Scores mean: {np.mean(scores):.4f}')
print(f'Scores std: {np.std(scores):.4f}')

clf.fit(X, y)
df_subm['moved_after_2019'] = clf.predict(df_test)

df_subm['moved_after_2019'] = df_subm['moved_after_2019'].astype(int)
df_subm['moved_after_2019'].value_counts()
df_subm.to_csv('submission.csv', index = False)






















             