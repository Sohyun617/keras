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

for i in range(len(x)):
    print(x[i],y[i])

#----------------------------------

input1 = Input(shape=(3,))
dense1 = Dense(5,activation='relu')(input1)
dense2 = Dense(3)(dense1)
output = Dense(1)(dense2)

model = Model(inputs=input1, outputs=output)

model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=100,batch_size=1)

loss= model.evaluate(x,y, batch_size=1)
print(loss)

x_prd = np.array([[90,100,110]])
# p_inpx = np.transpose(p_inpx)
bbb = model.predict(x_prd,batch_size=1)
print(bbb)







