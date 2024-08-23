import numpy as np

#read a npy file

data = np.load('/Users/anand/Desktop/SBU/botorch/MMMA/Beale/baseline_function_Beale2_kernel_Matern52_acquisition_LogEI_optimization_results.npy', allow_pickle=True)

print(type(data))
print(data.shape)
print(data[0].shape)
print(type(data[0]))
print(type(data[0][0]))
print(type(data[0][1]))
print(type(data[0][2]))
print(type(data[0][3]))
print(type(data[0][4]))
