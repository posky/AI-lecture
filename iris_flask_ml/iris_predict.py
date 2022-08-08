import os
import pickle

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    stratify=iris.target,
    random_state=0
)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print('\n\nrandom forest')
model = RandomForestClassifier(n_jobs=-1, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


print('\n\ngradient boosting')
model = GradientBoostingClassifier(random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

pickle.dump(model, open(os.path.dirname(__file__) + '/model/iris_base.pkl', 'wb'))

print(iris.keys())
print(iris.target_names)
print(iris.feature_names)
print(type(iris.data))
df = pd.DataFrame(iris.data)
print(df.describe())
