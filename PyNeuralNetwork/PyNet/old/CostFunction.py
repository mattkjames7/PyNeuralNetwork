import numpy as np
from .ForwardPropagate import ForwardPropagate
from .RollTheta import RollTheta

def CostFunction(Theta,X,y,s,Lambda):
	#now to populate a, including adding the ones for the bias unit
	a = []
	z = []
	m = np.size(y)
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


	#calculate J first (I will replace this with the vectorized version once I find my code)
	#simple way is to sum of K outputs (s[-1])
	J = 0.0
	h = a[-1]
	m = X.shape[0]
	for k in range(0,s[-1]):
		#select the training 
		yk = np.float32(y == k+1)
		hk = h[:,k]
		J +=  (1.0/m)*np.sum(-yk*np.log(hk.clip(min=1e-40)) - (1.0 - yk)*np.log((1.0 - hk).clip(min=1e-40)))

		
	#Regularization
	Reg = 0.0
	for i in range(0,L-1):
		T = Theta[i]
		Reg += Lambda/(2.0*m) * np.sum(T[:,1:]**2.0)
	J += Reg
	
	return J


def CostFunction_y2d(Theta,X,y,s,Lambda):
	#now to populate a, including adding the ones for the bias unit
	a = []
	z = []
	shape = y.shape
	m = shape[0]
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


	#calculate J first (I will replace this with the vectorized version once I find my code)
	#simple way is to sum of K outputs (s[-1])
	J = 0.0
	h = a[-1]
	m = X.shape[0]
	for k in range(0,s[-1]):
		#select the training 
		yk = np.float32(y[:,k])
		hk = h[:,k]
		J +=  (1.0/m)*np.sum(-yk*np.log(hk.clip(min=1e-40)) - (1.0 - yk)*np.log((1.0 - hk).clip(min=1e-40)))

		
	#Regularization
	Reg = 0.0
	for i in range(0,L-1):
		T = Theta[i]
		Reg += Lambda/(2.0*m) * np.sum(T[:,1:]**2.0)
	J += Reg
	
	return J
	
def ReturnUnrolledCostFunction(X,y,s,Lambda,y2d=False):
	
	
	def UnrolledCostFunction(Theta):
		
		RolledTheta = RollTheta(Theta,s)
		if y2d:
			J = CostFunction_y2d(RolledTheta,X,y,s,Lambda)
		else:
			J = CostFunction(RolledTheta,X,y,s,Lambda)
		return J
	
	return UnrolledCostFunction
		
def ReturnCostPrinter(X,y,s,Lambda,y2d=False):
	
	
	def PrintCost(Theta):
		RolledTheta = RollTheta(Theta,s)
		if y2d:
			J = CostFunction_y2d(RolledTheta,X,y,s,Lambda)
		else:
			J = CostFunction(RolledTheta,X,y,s,Lambda)
		print('Cost: {0}'.format(J))
		
	return PrintCost
		
