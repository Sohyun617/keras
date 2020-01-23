#1. 데이터
import numpy as np

x= np.array([range(1,101),range(101,201)])
y= np.array([range(1,101),range(101,201)])

x=x.reshape(100,2)
y=y.reshape(100,2)

# x= np.transpose(x) #reshape과 같은 기능
# y= np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test,  y_train, y_test = train_test_split(
    x, y, train_size=0.6, random_state=66, shuffle = False)

x_test, x_val,  y_test, y_val = train_test_split(
    x_test, y_test, test_size=0.5, random_state=66, shuffle = False)

print(x_train)
print(x_test)
print(x_val)

# print(x.shape) 
# print(y.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim =2))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))

# model.summary()


#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs =300, batch_size=10, validation_data= (x_val, y_val))

#4. 평가
loss, mse = model.evaluate(x_test, y_test, batch_size=10)
print('mse: ', mse)

x_prd = np.array([[201,202,203],[204,205,206]])
x_prd= x_prd.reshape(3,2)
aaa= model.predict(x_prd, batch_size=10)
print(aaa)

y_predict= model.predict(x_test, batch_size=10)

#RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2:", r2_y_predict)