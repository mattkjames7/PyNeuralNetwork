import numpy as np
from .Sigmoid import Sigmoid

def ForwardPropagate(a,z,L,Theta):
	for j in range(0,L-1):
		zt = np.dot(a[j],Theta[j].T)
		if j < L-2:
			at = np.copy(a[j+1])
			at[:,1:] = Sigmoid(zt)
			at[:,0] = 1.0
		else:
			at = Sigmoid(zt)
		a[j+1] = at
		z[j] = zt
