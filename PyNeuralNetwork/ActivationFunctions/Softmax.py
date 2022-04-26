import numpy as np

def Softmax(z):
	'''
	Softmax function - convert the output of a neural network to an 
	array of probabilities.
	
	Inputs
	======
	z : float
		Output matrix, shape (m,k), where m is the number of samples and
		k is the number of output nodes.
		
	Returns
	=======
	s : float
		Softmax output - shape (m,k).
	'''
	return (np.exp(z).T/np.sum(np.exp(z),axis=1)).T
