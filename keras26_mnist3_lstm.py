from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, LSTM, Input
from keras.callbacks import EarlyStopping

import numpy as np 

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# print(x_train)
# print(y_train)
print(x_train.shape) 
print(y_train.shape)

# x_train = x_train.reshape(x_train.shape[0],28*28)
# x_test = x_test.reshape(x_test.shape[0],28*28)

print(x_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (28,28))) 
model.add(Dense(5)) #(5,3) 
model.add(Dense(1)) #(5,)
# model.summary()

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(x_train,y_train, validation_split=0.2,
          epochs=100, batch_size=8, verbose=1)

acc=model.evaluate(x_test, y_test)

print(acc)
