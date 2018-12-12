import numpy as np

def Sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))


def SigmoidGradient(z):
	
	return Sigmoid(z)*(1.0-Sigmoid(z));
