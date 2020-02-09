
import numpy as np
from numpy import array, hstack
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Input


def split_sequence(sequence, n_steps):
    x,y = list(), list()
    for i in range(len(sequence)): #10
        end_ix = i + n_steps 
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix-1,] 
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

for i in range(len(x)):
    print(x[i], y[i])

print(x.shape) # (7,3,3)
print(y.shape) #(7,3)


# model = Sequential()
# model.add(LSTM(10, activation = 'relu', input_shape = (3,3))) #############
#  #10= 출력노드수, input_shape = 열이 세개, 한개씩 잘라서 작업
# model.add(Dense(5)) #(5,3) 
# model.add(Dense(3)) #(5,) ############

# # #훈련&평가
# model.compile(optimizer='adam',loss='mse')
# model.fit(x,y,epochs=100,batch_size=1)

# loss= model.evaluate(x,y, batch_size=1)
# print(loss)

# x_prd = np.array([[90,95,185],[100,105,205],[110,115,225]])
# x_prd = x_prd.reshape(1,3,3) ###############

# y_predict = model.predict(x_prd)
# print(y_predict)