import numpy as np
from .ForwardPropagate import ForwardPropagate
from .Softmax import Softmax

def Predict(X,Theta,L,s,Threshold=0.5,y2d=False):

	m = X.shape[0]
	a = []
	z = []
	L = np.size(s)
	a.append(X)
	for j in range(1,L):
		if j == L-1:
			at = np.zeros((m,s[j]),dtype='float32')
		else:
			at = np.zeros((m,s[j]+1),dtype='float32')
		zt = np.copy(at)
		a.append(at)
		z.append(zt)
	ForwardPropagate(a,z,L,Theta)	
	h = a[-1]
	prob = Softmax(z[-1])
	if y2d:
		result = np.where(h >= Threshold)
		res = np.zeros(h.shape,dtype='int32')
		res[result[0],result[1]] = 1
		return  res,prob
	result = np.where(h == np.array([np.max(h,axis=1)]).T) 
	res = np.zeros(m,dtype='int32')
	res[result[0]] = result[1] + 1
	return res,prob
	
