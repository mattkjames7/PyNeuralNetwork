import numpy as np

def Sigmoid(z):
	#return 1.0/(1.0 + np.exp(-z))
	return np.exp(-np.logaddexp(0,-z))

def SigmoidGradient(z):
	
	return Sigmoid(z)*(1.0-Sigmoid(z));

def InverseSigmoid(a):
	z = - np.log(1/np.float64(a) - 1)
	return z

def InverseSigmoidGradient(a):
	z = InverseSigmoid(a)
	return SigmoidGradient(z)
