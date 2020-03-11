import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

#데이터 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding = 'utf-8')

#붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y= iris_data.loc[:,"Name"]
x= iris_data.loc[:,["SepalLength","SepalWidth","PetalLength",
                    "PetalWidth"]]

#학습전용과 테스트 전용 분리하기
warnings.filterwarnings('ignore')
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,
                        train_size=0.7, shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

model = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

model.fit(x_train,y_train)
print("테스트 점수: ", model.score(x_test, y_test))