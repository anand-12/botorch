import numpy as np

#read a npy file

data = np.load('False_likelihood_MMMA_function_DropWave2_kernel_Matern52_RBF_acquisition_LogEI_LogPI_UCB_optimization_results.npy', allow_pickle=True)

print(data.shape)