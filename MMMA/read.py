import numpy as np

#read a npy file

data = np.load('./Hartmann/False_uniform_MMMA_function_Hartmann6_kernel_Matern52_acquisition_LogEI_optimization_results.npy', allow_pickle=True)

print(data)