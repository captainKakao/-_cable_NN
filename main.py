import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input
from keras import Model
from numpy import genfromtxt
import os

file_src = "C:/Users/Professional/Documents/pycharm/Static_cable_data_creator/files/"
old = 0
d = 1000
k = 0
incr = 0
my_indic = 0

for x in os.listdir(file_src):
    num, name = x.split("_")
    if int(num) > old:
        old = int(num)

loaded_arr = []
in_list = [[0 for i in range(11)] for j in range(old+1)]
out_list = [[0 for i in range(33)] for j in range(old+1)]


for x in os.listdir(file_src):

    num, name = x.split("_")
    part, end = name.split(".")
    my_indic = int(num)

    my_data = genfromtxt(file_src + x)
    if part == 'in':
        for a in range(0,len(my_data)):
            in_list[my_indic][a] = my_data[a]/d

    if part == 'out':
        for a in range(0, len(my_data)):
            out_list[my_indic][a] = my_data[a]/d



model = keras .Sequential()
model.add(Dense(units=11, input_shape=(11,), activation='tanh'))
model.add(Dense(units=8, activation='tanh'))
model.add(Dense(units=8, activation='tanh'))
model.add(Dense(units=33, activation='tanh'))
print(model.summary())
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))



print(1)
history = model.fit(in_list, out_list, epochs=150, verbose=True)
print(2)
plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
model.save('cable_NN.keras')
# model.save_weights('cable_NN_w.keras')
# print(model.get_weights())

result = model.predict(in_list[:1])
print(result)

print(in_list[:1])

# print(model.predict([10, 100, 0.02, 1.2, 0.01, 1025, 1.5, 0, -100, -100, -100]))
