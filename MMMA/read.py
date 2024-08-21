import numpy as np

#read a npy file

data = np.load('multithreaded_LogEI_Matern52_hartmann_optimization_results.npy', allow_pickle=True)

print(data)