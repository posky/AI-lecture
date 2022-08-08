import os
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import sklearn as sk
import flask

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


print(pd.__version__)
print(np.__version__)
print(sns.__version__)
print(sk.__version__)
print(flask.__version__)


# read data
# getcwd - get current working directory
print(os.getcwd())
print(os.path.dirname(__file__))

INPUT_PATH = os.path.dirname(__file__)
train = pd.read_csv(INPUT_PATH + '/data/train.csv')
test = pd.read_csv(INPUT_PATH + '/data/test.csv')
sub = pd.read_csv(INPUT_PATH + '/data/sample_submission.csv')

print(train.shape, test.shape, sub.shape)
print(train.columns)
print(train.info())
print(train.head())

# model selection - train, test
"""
    'id', 'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
"""
train.loc[train['income'] == '>50K', 'target'] = 1
train.loc[train['income'] == '<=50K', 'target'] = 0
train['target'] = train['target'].astype('int')

sel = ['age', 'fnlwgt', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss']
X = train[sel]
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=train['target'],
    random_state=0
)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X.describe())

# model
# random forest
# rf = RandomForestClassifier(n_jobs=-1, random_state=0)
# rf.fit(X_train, y_train)
# print('\n\nrandom forest')
# print(f'train set acc: {rf.score(X_train, y_train)}')
# print(f'test set acc: {rf.score(X_test, y_test)}')
# cvs = cross_val_score(rf, X, y, n_jobs=-1, scoring='roc_auc')
# print(f'cross validation score(AUC): {cvs}')
# print(f'mean cross validation score(AUC): {cvs.mean()}')

# grandieng boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    random_state=0
)
gb.fit(X_train, y_train)
print('\n\ngradient boosting')
# print(f'train set acc: {gb.score(X_train, y_train)}')
# print(f'test set acc: {gb.score(X_test, y_test)}')
cvs = cross_val_score(gb, X, y, cv=5, n_jobs=-1, scoring='roc_auc')
# print(f'cross validation score(AUC): {cvs}')
print(f'mean cross validation score(AUC): {cvs.mean()}')

# logistic regression
# logreg = LogisticRegression(n_jobs=-1)
# logreg.fit(X_train, y_train)
# print('\n\nlogictic regression')
# print(f'train set acc: {logreg.score(X_train, y_train)}')
# print(f'test set acc: {logreg.score(X_test, y_test)}')
# cvs = cross_val_score(logreg, X, y, n_jobs=-1, scoring='roc_auc')
# print(f'cross validation score(AUC): {cvs}')
# print(f'mean cross validation score(AUC): {cvs.mean()}')


model = GradientBoostingClassifier(random_state=0)
model.fit(X_train, y_train)
score = cross_val_score(model, X_test, y_test, cv=5, n_jobs=-1, scoring='roc_auc')
print(f'cross validation score(AUC): {score.mean()}')

pickle.dump(model, open(INPUT_PATH + '/model/income_base.pkl', 'wb'))
