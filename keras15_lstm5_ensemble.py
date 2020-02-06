from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
#1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],
           [2,3,4],[3,4,5],[4,5,6]])
y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7])

x1= x1.reshape(x1.shape[0], x1.shape[1], 1)
x2= x2.reshape(x1.shape[0], x2.shape[1], 1)

# #2. 모델구성
# model = Sequential()
# model.add(LSTM(10, activation = 'relu', input_shape = (3,1))) #10= 출력노드수, input_shape = 열이 세개, 한개씩 잘라서 작업
# model.add(Dense(10)) #(5,3) 
# model.add(Dense(8))
# model.add(Dense(1)) #(5,)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1= Input(shape=(3,1))
model1 = LSTM(10, activation='relu')(input1)
model1 = Dense(5)(model1)

input2= Input(shape=(3,1))
model2 = LSTM(11, activation='relu')(input2)
model2 = Dense(5)(model2)

from keras.layers.merge import concatenate, Add
#merge1 = concatenate([output1, output2]) #output1과 2를 합치다
merge1 = Add()([model1, model2])

output1 = Dense(30)(merge1)
output1 = Dense(30)(merge1)
output1 = Dense(1)(merge1)

output2 = Dense(20)(merge1)
output2 = Dense(20)(merge1)
output2 = Dense(1)(merge1)

# middle1 = Dense(4)(merge1)
# middle2 = Dense(7)(middle1)
# output3 = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = [output1,output2])

#model = Model(inputs = [input1, input2], outputs = output3)
model.summary()

#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=20, mode='auto') # max를 넘지않게 min 밑으로 떨어지지 않게 patience,
model.fit([x1,x2],[y1,y2], epochs =1000, batch_size=1, verbose= 1, callbacks=[early_stopping]) 
#patience값이 너무 크면 과적합이 발생할 수도 있음

#4. 평가
loss_child = model.evaluate([x1,x2],[y1,y2], batch_size=1)
print(loss_child)

x1_input = array([[6.5,7.5,8.5],[50,60,70],
                 [70,80,90],[100,110,120]])
x2_input = array([[6.5,7.5,8.5],[50,60,70],
                  [70,80,90],[100,110,120]])

x1_input = x1_input.reshape(4,3,1)
x2_input = x2_input.reshape(4,3,1)

y_predict = model.predict([x1_input, x2_input])
print(y_predict)

