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
model.add(LSTM(10, activation = 'relu', input_shape = (3,1))) #10= 출력노드수, input_shape = 열이 세개, 한개씩 잘라서 작업
model.add(Dense(8)) #(5,3) 
model.add(Dense(5))
model.add(Dense(1)) #(5,)

#model.summary()

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mae'])
model.fit(x,y, epochs =100, batch_size=1)

#4. 평가
loss, mae = model.evaluate(x,y, batch_size=1)
print(loss, mae)

x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]])
x_input = x_input.reshape(4,3, 1)

y_predict = model.predict(x_input)
print(y_predict)