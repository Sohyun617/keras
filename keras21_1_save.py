#2.모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()#——————————모델 방식

#model.add(Dense(5, input_shape = (2,)))

model.add(Dense(5, input_dim = 3)) #———Hidden Layer 1
model.add(Dense(2)) #——————————Hidden Layer 2
model.add(Dense(3)) #——————————Hidden Layer 3
model.add(Dense(1)) #——————————Output Layer #iuput_dim과 Output을 맞춰줘야 한다. 

#model.summary() #————————————레이어 형태&파라미터

model.save('./save/savetest01.h5')
print('저장 잘 됐다')