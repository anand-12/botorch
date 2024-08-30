import numpy as np

#read a npy file

data = np.load('./Ackley_2/MultiModel_likelihood.npy', allow_pickle=True)

print(type(data))
print(data.shape)
print(data[0].shape)
print(type(data[0]))
print(type(data[0][0]))
print(type(data[0][1]))
print(type(data[0][4]))
print(type(data[0][5]))
print(type(data[0][6]))
