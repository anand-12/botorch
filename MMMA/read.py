import numpy as np

#read a npy file

data = np.load('baseline_Hartmann_optimization_results.npy', allow_pickle=True)

print(data)