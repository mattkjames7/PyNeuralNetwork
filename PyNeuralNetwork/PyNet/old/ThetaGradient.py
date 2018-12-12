import numpy as np
from .ForwardPropagate import ForwardPropagate
from .RollTheta import RollTheta,UnrollTheta
from .CostFunction import ReturnUnrolledCostFunction

def BackPropagate(a,y,s,L,Theta,Lambda):
	shape = y.shape
	m = shape[0]
	delta = []
	Tgrad = []
	for i in range(0,L):
		delta.append(None)
		Tgrad.append(None)
#		#now calculate deltas		
	for j in range(L-1,-1,-1):
		if j == L-1:
			dl = np.zeros((m,s[-1]),dtype='float32')
			for k in range(0,s[-1]):
				yk = np.float32(y == k+1)
				dl[:,k] = a[-1][:,k] - yk	
			
		else:
			dl = np.dot(delta[j+1],Theta[j])*a[j]*(1.0-a[j])
			dl = dl[:,1:]
		delta[j] = dl
	
	#now to work out Theta gradients?
	for j in range(0,L-1):
		Tg = np.dot(delta[j+1].T,a[j])/m
		Tg[:,1:] += (np.float32(Lambda)/m)*(Theta[j][:,1:]**2)
		Tgrad[j] = Tg
		
	return Tgrad[:-1]

def BackPropagate_y2d(a,y,s,L,Theta,Lambda):
	shape = y.shape
	m = shape[0]
	delta = []
	Tgrad = []
	for i in range(0,L):
		delta.append(None)
		Tgrad.append(None)
#		#now calculate deltas		
	for j in range(L-1,-1,-1):
		if j == L-1:
			dl = np.zeros((m,s[-1]),dtype='float32')
			for k in range(0,s[-1]):
				yk = np.float32(y[:,k])
				dl[:,k] = a[-1][:,k] - yk	
			
		else:
			dl = np.dot(delta[j+1],Theta[j])*a[j]*(1.0-a[j])
			dl = dl[:,1:]
		delta[j] = dl
	
	#now to work out Theta gradients?
	for j in range(0,L-1):
		Tg = np.dot(delta[j+1].T,a[j])/m
		Tg[:,1:] += (np.float32(Lambda)/m)*(Theta[j][:,1:]**2)
		Tgrad[j] = Tg
		
	return Tgrad[:-1]



def ThetaGradient(Theta,X,y,s,Lambda):
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
	
	Tgrad = BackPropagate(a,y,s,L,Theta,Lambda)	
	return Tgrad
	
def ThetaGradient_y2d(Theta,X,y,s,Lambda):
	a = []
	z = []
	m = y.shape[0]
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
	
	Tgrad = BackPropagate_y2d(a,y,s,L,Theta,Lambda)	
	return Tgrad
	
def ReturnUnrolledThetaGradientFunction(X,y,s,Lambda,y2d=False):
	
	
	def UnrolledThetaGradient(Theta):
		
		RolledTheta = RollTheta(Theta,s)
		if y2d:
			dJ = ThetaGradient_y2d(RolledTheta,X,y,s,Lambda)
		else:
			dJ = ThetaGradient(RolledTheta,X,y,s,Lambda)
		udJ = UnrollTheta(dJ)
		return udJ
		
	
	return UnrolledThetaGradient


def ApproxGradient(Theta,X,y,s,Lambda,E=1e-4):
	
	CF = ReturnUnrolledCostFunction(X,y,s,Lambda)
	T = UnrollTheta(Theta)
	
	nT = np.size(T)
	P = np.zeros(nT,dtype='float32')
	Tgrad = np.zeros(nT,dtype='float32')
	for i in range(0,nT):
		p = P
		p[i] = E
		J0 = CF(T-p)
		J1 = CF(T+p)
		Tgrad[i] = (J1-J0)/(2.0*E)
		
	Tgrad = RollTheta(Tgrad,s)
	return Tgrad
		
		
