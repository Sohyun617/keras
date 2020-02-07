from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x= x.reshape(x.shape[0], x.shape[1], 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1),return_sequences=True )) 
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(BatchNormalization()) #흩어진 가중치들을 모아줌
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(5, activation='linear'))
model.add(Dense(1))

model.summary()

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=20, mode='auto') # max를 넘지않게 min 밑으로 떨어지지 않게 patience,
model.fit(x,y, epochs =100, batch_size=1, verbose= 1, callbacks=[early_stopping]) 
#patience값이 너무 크면 과적합이 발생할 수도 있음

#4. 평가
loss, mae = model.evaluate(x,y, batch_size=1)
print(loss, mae)

x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]])
x_input = x_input.reshape(4,3, 1)

y_predict = model.predict(x_input)
print(y_predict)

