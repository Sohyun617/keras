from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x= array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[7,8,9]])
y= array([4,5,6,7,8])

print(x.shape)
print(y.shape)

x= x.reshape(x.shape[0], x.shape[1], 1)
#x = x.reshape(5,3,1) # 뒤에서 (none, 3,1)로 받기위해 5,3,1로 reshape해준것

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1)))
 #10= 출력노드수, input_shape = 열이 세개, 한개씩 잘라서 작업
model.add(Dense(5)) #(5,3) 
model.add(Dense(1)) #(5,)

model.summary()


#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mae'])
model.fit(x,y, epochs =100, batch_size=1)

#4. 평가
loss, mae = model.evaluate(x,y, batch_size=1)
print(loss, mae)

x_input = array([6,7,8]) #(3,) -> (1,3)-> (1,3,1)
x_input = x_input.reshape(1,3,1 )

y_predict = model.predict(x_input)
print(y_predict)