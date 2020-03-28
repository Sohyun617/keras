import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy import array, hstack
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Input


path = 'C:\\Users\\student\\Desktop\\'
#kospi = pd.read_csv(path+'kospi200.csv', encoding = 'euc-kr')
samtest = pd.read_csv(path+'samtest_data.csv', encoding = 'euc-kr')
# print(samsung)
kospi200 =np.load('./samsung/data/kospi200.npy')

samsung_siga = []
samsung_high = []
samsung_low = []
samsung_close = []
samsung_volume = []

for k in range(len(samtest)):
    samsung_siga.append(samtest[['시가']].iloc[k].values[0].replace(',', ''))
    samsung_siga[k] = int(samsung_siga[k])
    
for k in range(len(samtest)):
    samsung_high.append(samtest[['고가']].iloc[k].values[0].replace(',', ''))
    samsung_high[k] = int(samsung_high[k])
    
for k in range(len(samtest)):
    samsung_low.append(samtest[['저가']].iloc[k].values[0].replace(',', ''))
    samsung_low[k] = int(samsung_low[k])
    
for k in range(len(samtest)):
    samsung_close.append(samtest[['종가']].iloc[k].values[0].replace(',', ''))
    samsung_close[k] = int(samsung_close[k])
    
for k in range(len(samtest)):
    samsung_volume.append(samtest[['거래량']].iloc[k].values[0].replace(',', ''))
    samsung_volume[k] = int(samsung_volume[k])

#print(samsung_siga)
# print(samsung_high)
# print(samsung_low)
# print(samsung_close)
# print(samsung_volume)

samsung_siga2= np.array([samsung_siga])
samsung_high2= np.array([samsung_high])
samsung_low2= np.array([samsung_low])
samsung_close2= np.array([samsung_close])
samsung_volume2= np.array([samsung_volume])

# print(samsung_siga2.shape) #(1,426)
# print(samsung_high2.shape)#(1,426)
# print(samsung_low2.shape)#(1,426)
# print(samsung_close2.shape)#(1,426)
# print(samsung_volume2.shape)#(1,426)

samsung_siga2= samsung_siga2.reshape(231,1)
samsung_high2= samsung_high2.reshape(231,1)
samsung_low2= samsung_low2.reshape(231,1)
samsung_close2= samsung_close2.reshape(231,1)
samsung_volume2= samsung_volume2.reshape(231,1)

# print(samsung_siga2.shape) #(231,1)
# print(samsung_high2.shape)#(231,1)
# print(samsung_low2.shape)#(231,1)
# print(samsung_close2.shape)#(231,1)
# print(samsung_volume2.shape)#(231,1)
#print(samsung_volume2)

dataset = hstack((samsung_siga2,samsung_high2,samsung_low2,samsung_volume2,samsung_close2))
#print(dataset)
y= samsung_close2
x= dataset

#print(dataset.shape)
#print(samsung_close2.shape)


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


x,y = split_xy5(dataset,5,1)
# for i in range(len(x)):
#     print(x[i], y[i])
    
# print(x.shape) #(226,5,5)
# print(y.shape) #(226,)

#print(x)
#print(y)

#데이터셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=1, test_size=0.3, shuffle= False
)
# print(x_train.shape) #(158,5,5)
# print(x_test.shape) #(68,5,5)


#3차원 ->2차원
x_train= np.reshape(x_train,
        (x_train.shape[0], x_train.shape[1]* x_train.shape[2]))
x_test= np.reshape(x_test,
        (x_test.shape[0], x_test.shape[1]* x_test.shape[2]))
#print(x_train.shape) #158,25
#print(x_test.shape) #68,25

#데이터전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)


#모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_shape =(25,))) ############
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) ############

# model.summary()

#훈련&평가
model.compile(optimizer='adam',loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)

model.fit(x_train, y_train, epochs=10,validation_split=0.2,
          verbose=1,batch_size=1, callbacks=[early_stopping])

loss= model.evaluate(x_test,y_test, batch_size=1)
print(loss)

y_pred= model.predict(x_test,y_test)
for i in range(5):
    print('종가 : ', y[i], '/예측가 :', y_pred[i])
# x_prd = np.array(x[-1])
# # x_prd = np.array([[57800, 58400, 56800, 19749457,
# #                   5880,1,1,1,
# #                   1,1,1,1,
# #                   1,1,1,1,
# #                   1,1,1,1]])
# x_prd = x_prd.reshape(1,25)

# test = model.predict(x_prd, batch_size=2)
# print(test)




