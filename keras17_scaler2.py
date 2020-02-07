import numpy as np
from numpy import array
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Input


x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20000,30000,40000],[30000,40000,50000],
           [40000,50000,60000],[100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
# print(x2)

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
# print(x1)
x= x.reshape(x.shape[0], x.shape[1], 1)

x_train = x[:10]
x_test = x[10:]
y_train = y[:10]
y_test = y[10:]

#---------------------------------------------------------
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1)))
model.add(Dense(5)) #(5,3) 
model.add(Dense(1)) #(5,)

model.summary()

#-------
#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mae'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

#4. 평가
loss, mae = model.evaluate(x,y, batch_size=1)
print(loss, mae)

aaa = model.evaluate(x_test,y_test,batch_size=1)
print(aaa)

x_prd = np.array([[230,240,250]])
x_prd = x_prd.reshape(1,3,1)

bbb = model.predict(x_prd,batch_size=1)
print(bbb)

y_predict = model.predict(x_test,batch_size=1)

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("R2: ",r2_y_predict)









