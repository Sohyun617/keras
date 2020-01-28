#1. 데이터
import numpy as np

x1= np.array([range(1,101),range(101,201),range(301,401)])
y1= np.array([range(1,101)])

x2= np.array([range(1001,1101),range(1101,1201),range(1301,1401)])
#y2= np.array([range(1101,1201)])

print(x1.shape) # (3,100)
print(y1.shape)  # (1,100)
#print(y2.shape)  # (1,1)

# x=x.reshape(100,3)
# y=y.reshape(100,1)
# y2=y2.reshape(1,1)

x1= np.transpose(x1) #reshape과 같은 기능
y1= np.transpose(y1)
x2= np.transpose(x2)


from sklearn.model_selection import train_test_split
x2_train, x2_test, x1_train, x1_test,  y_train, y_test,  = train_test_split(
    x2, x1, y1, train_size=0.6, random_state=66, shuffle = False)

x2_test, x2_val, x1_test, x1_val,  y_test, y_val = train_test_split(
    x2_test, x1_test, y_test, test_size=0.5, random_state=66, shuffle = False)

print(x1_train)
print(x1_test)
print(x1_val)

# print(x.shape) 
# print(y.shape)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2 = Input(shape=(3,))
dense21 = Dense(7)(input2)
dense22 = Dense(4)(dense21)
output2 = Dense(5)(dense22)


from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2]) #output1과 2를 합치다

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output3 = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = output3)
model.summary()


#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit([x1_train,x2_train], y_train, epochs =100, batch_size=10,
           validation_data= ([x1_val, x2_val], y_val))

#4. 평가
loss, mse = model.evaluate([x1_test, x2_test], y_test, batch_size=10)
print('mse: ', mse)

x_prd1 = [[201,202,203],[204,205,206],[207,208,209]]
x_prd2 = [[301,302,303],[304,305,306],[307,308,309]]
x_prd1= np.transpose(x_prd1)
x_prd2= np.transpose(x_prd2)

aaa= model.predict([x_prd1, x_prd2], batch_size=10)
print(aaa)

y_predict= model.predict([x1_test, x2_test], batch_size=10)

#RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2:", r2_y_predict)