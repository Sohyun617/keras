import numpy as np
from numpy import array
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Input

def split_sequence(sequence, n_steps):
    x,y = list(), list()
    for i in range(len(sequence)): #10
        end_ix = i + n_steps #0+4=4/// 6+4
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] #0,1,2,3/4
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)               

dataset = [10,20,30,40,50,60,70,80,90,100]
n_steps = 3

x, y = split_sequence(dataset, n_steps)
print(x)
print(y)

x= x.reshape(x.shape[0], x.shape[1], 1)
#----------------------------------

#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1))) #10= 출력노드수, input_shape = 열이 세개, 한개씩 잘라서 작업
model.add(Dense(8)) #(5,3) 
model.add(Dense(5))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=100,batch_size=1)

#4. 평가
loss= model.evaluate(x,y, batch_size=1)
print(loss)

x_prd = np.array([[90,100,110]])
x_prd = x_prd.reshape(1,3,1)

y_predict = model.predict(x_prd)
print(y_predict)






