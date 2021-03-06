from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np 

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
#(32 ,32, 3)
# print(x_train)
# print(y_train)
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0],32,32,3).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],32,32,3).astype('float32')/255

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

model = Sequential() 
model.add(Conv2D(32,(2,2),strides=2,padding= 'valid', #7 장, 픽셀 2x2로 짜른다
                 input_shape=(32,32,3))) 
model.add(Conv2D(32,(2,2)))
model.add(MaxPooling2D(2,2)) 
model.add(Flatten()) # dense 층에 전해주기위해서 4*4*7값을 flatten이 계산
model.add(Dense(10, activation='softmax'))

#model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=20)

model.fit(x_train,y_train, validation_split=0.2,
          epochs=100, batch_size=8, verbose=1, 
          callbacks=[early_stopping])

acc=model.evaluate(x_test, y_test)

print(acc)

