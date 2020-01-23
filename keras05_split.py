#1. 데이터
import numpy as np

x= np.array(range(1,101))
y= np.array(range(1,101))

train_index = int(len(x)*0.6)
tv_index=int(len(x)*0.2)

x_train = x[0:train_index]
y_train = y[0:train_index]
x_test =x[train_index:train_index +tv_index]
y_test =y[train_index:train_index +tv_index]
x_val =x[train_index +tv_index: train_index + (tv_index *2)]
y_val =y[train_index +tv_index: train_index + (tv_index *2)]

# print(x.shape) 
# print(y.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim =1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs =100, batch_size=1, validation_data= (x_val, y_val))

#4. 평가
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse: ', mse)

x_prd = np.array([101,102,103])
aaa= model.predict(x_prd, batch_size=1)
print(aaa)

# bbb = model.predict(x, batch_size=1)
# print(bbb)


