from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LSTM,Reshape
from keras.callbacks import EarlyStopping
import numpy as np 

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# print(x_train)
# print(y_train)
# print(x_train.shape) #(60000,28,28)
# print(y_train.shape) #(60000,)

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape) #(60000,28,28,1)
print(y_train.shape) #(60000,10)


# model.add(MaxPooling2D(2,2)) 
# model.add(Flatten()) # dense 층에 전해주기위해서 4*4*7값을 flatten이 계산
# model.add(Dense(10, activation='softmax'))


cnn = Sequential()
cnn.add(Conv2D(32,
                 (2,2),
                 padding='valid',
                 activation='relu',
                 strides=1,input_shape=(28,28,1)))
cnn.add(Conv2D(32,(2,2)))
cnn.add(MaxPooling2D(2,2))
cnn.add(Flatten()) # dense 층에 전해주기위해서 4*4*7값을 flatten이 계산

cnn.add(Reshape((-1,1)))

cnn.add(LSTM(10, activation = 'relu'))
cnn.add(Dense(5)) #(5,3) 
cnn.add(Dense(10))

cnn.summary()

cnn.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


cnn.fit(x_train, y_train,
          batch_size=8,
          verbose=1,
          epochs=100,
          validation_split=0.2)
score, acc = cnn.evaluate(x_test, y_test, batch_size=8)
print('Test score:', score)
print('Test accuracy:', acc)


