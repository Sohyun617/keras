import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

import warnings
warnings.filterwarnings(action='ignore')

iris_data =pd.read_csv("./data/iris2.csv",encoding='utf-8')

y=iris_data.loc[:,'Name']#.values
x=iris_data.iloc[:,:-1]#.values

warnings.filterwarnings(action='ignore')
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
    {'clf__C': param_range, 'clf__kernel': ['linear']},
    {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid,
                  scoring='accuracy', cv=10, n_jobs=1)
#%time gs = gs.fit(x_train, y_train)

print(gs.best_score_)
print(gs.best_params_)