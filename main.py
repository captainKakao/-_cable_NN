import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input
from keras import Model
import tensorflow as tf
from numpy import genfromtxt
import os

# file_src = "C:/Users/Professional/Documents/pycharm/Static_cable_data_creator/files/"
file_src = "F:/files_NN_new/"
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
in_list1 = [[0 for i in range(10)] for j in range(old+1)]
out_list1 = [[0 for i in range(33)] for j in range(old+1)]
in_list = np.zeros((old+1, 10))
out_list= np.zeros((old+1, 33))


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
model.add(Dense(units=10, input_shape=(10,), activation='tanh'))
model.add(Dense(units=32, activation='tanh'))
model.add(Dense(units=32, activation='tanh'))
# model.add(Dense(units=32, activation='tanh'))
# model.add(Dense(units=32, activation='tanh'))
model.add(Dense(units=33, activation='tanh'))
print(model.summary())
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))



print(1)
history = model.fit(in_list, out_list, epochs=100, batch_size=16, validation_split=0.2, verbose=True)
print(2)

# model.evaluate()

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
