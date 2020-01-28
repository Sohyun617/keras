#1. 데이터
import numpy as np

x= np.array([range(1,101),range(101,201),range(301,401)])
y= np.array([range(1,101)])
y2= np.array([range(101,102)])

print(x.shape) # (3,100)
print(y.shape)  # (1,100)
#print(y2.shape)  # (1,1)

# x=x.reshape(100,3)
# y=y.reshape(100,1)
# y2=y2.reshape(1,1)

x= np.transpose(x) #reshape과 같은 기능
y= np.transpose(y)
# y2= np.transpose(y2)



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
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()

# input1 = Input(shape=(3,))
# dense1 = Dense(5)(input1)
# dense2 = Dense(2)(dense1)
# dense3 = Dense(3)(dense2)
# output1 = Dense(1)(dense3)

input1 = Input(shape=(3,))
x = Dense(5)(input1)
x = Dense(2)(x)
x = Dense(3)(x)
output1 = Dense(1)(input1)
model = Model(inputs = input1, outputs = output1)

model.summary()


#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs =100, batch_size=30, validation_data= (x_val, y_val))

#4. 평가
loss, mse = model.evaluate(x_test, y_test, batch_size=30)
print('mse: ', mse)

x_prd = np.array([[201,202,203],[204,205,206],[207,208,209]])
x_prd= x_prd.reshape(3,3)
aaa= model.predict(x_prd, batch_size=30)
print(aaa)

y_predict= model.predict(x_test, batch_size=30)

#RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2:", r2_y_predict)