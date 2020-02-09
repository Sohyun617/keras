import numpy as np
from numpy import array, hstack
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Input

def split_sequence(sequence, n_steps):
    x,y = list(), list()
    for i in range(len(sequence)): #10
        end_ix = i + n_steps 
        if end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1,-1] 
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)               

in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))]) #i=0, 10+15=25, out_seq=[25, 45, 65, 85 ...]

# print(in_seq1.shape) #(10,)
# print(in_seq2.shape) #(10,)

in_seq1 = in_seq1.reshape(len(in_seq1),1)
in_seq2 = in_seq2.reshape(len(in_seq2),1)
out_seq = out_seq.reshape(len(out_seq),1)

# print(in_seq1.shape) #(10,1)
# print(in_seq2.shape) #(10,1)
# print(out_seq.shape) #(10,1)

dataset = hstack((in_seq1, in_seq2, out_seq))
n_steps = 3
#print(dataset)

x, y = split_sequence(dataset, n_steps)

# for i in range(len(x)):
#     print(x[i], y[i])

# print(x.shape) # (8,3,2)
print(y.shape) #(8,)

x= x.reshape(x.shape[0],-1) #8,6 ##############

#실습
#1. 함수분석
#2. DNN 모델 만들기
#3. 지표는 loss
#4.[[90 95]
#  [100 105]
#  [110 115]]
#  x predict 값

# #모델구성

input1 = Input(shape=(6,)) ##########
dense1 = Dense(5,activation='relu')(input1)
dense2 = Dense(3)(dense1)
output = Dense(1)(dense2)

model = Model(inputs=input1, outputs=output)

# #훈련&평가
model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=100,batch_size=1)

loss= model.evaluate(x,y, batch_size=1)
print(loss)

x_prd = np.array([[90,95],[100,105],[110,115]])
x_prd = x_prd.reshape(1,6) ###################

bbb = model.predict(x_prd,batch_size=1)
print(bbb)




