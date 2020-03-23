import pandas as pd
import numpy as np

samsung = np.load('./samsung/data/samsung.npy')
kospi200 =np.load('./samsung/data/kospi200.npy')

# print(samsung)
# print(samsung.shape)
# print(kospi200)
# print(kospi200.shape)

def split_xy5(dataset, time_steps, y_column):
    x,y = list(),list()
    for i in range(len(dataset)):
        x_end_number = i+time_steps
        y_end_number = x_end_number+y_column
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] #0:5, : ->5x5
        tmp_y = dataset[x_end_number:y_end_number, 3] #5:6, 3
        x.append(tmp_x)       
        y.append(tmp_y)
    return np.array(x), np.array(y)        #(421, 5,5) ->(421,1)

x,y = split_xy5(samsung,5,1)
#print(x.shape)
#print(y.shape)
#print(x[0,:],'\n',y[0])
#print(x)
print(y)


#데이터셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=1, test_size=0.3, shuffle= False
)
print(x_train.shape) #(294,5,5)
print(x_test.shape) #(127,5,5)

#3차원 ->2차원
x_train= np.reshape(x_train,
        (x_train.shape[0], x_train.shape[1]* x_train.shape[2]))
x_test= np.reshape(x_test,
        (x_test.shape[0], x_test.shape[1]* x_test.shape[2]))
print(x_train.shape) 
print(x_test.shape)

#데이터 전처리
#standardScaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)


# print(x_train)
# print(x_train.shape) #(294,25)

#모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_shape =(25,))) ############
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) ############


# #훈련&평가
model.compile(optimizer='adam',loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train,y_train,epochs=10,validation_split=0.2,
          verbose=1,batch_size=1, callbacks=[early_stopping])

loss= model.evaluate(x,y, batch_size=1)
print(loss)

# y_pred= model.predict(x_test,y_test)

# for i in range(5):
#     print('종가 : ', y_test[i], '/예측가 :', y_pred[i])
# x_prd = np.array(x[-1])
# # x_prd = np.array([[57800, 58400, 56800, 19749457,
# #                   5880,1,1,1,
# #                   1,1,1,1,
# #                   1,1,1,1,
# #                   1,1,1,1]])
# x_prd = x_prd.reshape(1,25)

# test = model.predict(x_prd, batch_size=2)
# print(test)
