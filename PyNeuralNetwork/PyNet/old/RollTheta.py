import numpy as np

def RollTheta(UTheta,s):
	L = np.size(s)
	Theta = []
	p = 0
	for j in range(0,L-1):
		n = s[j+1]
		m = s[j]+1
		T = UTheta[p:p+n*m].reshape((n,m))
		Theta.append(T)
		p+=n*m
	return Theta

def UnrollTheta(Theta):
	
	nt = len(Theta)
	n = 0 
	for i in range(0,nt):
		n += np.size(Theta[i])
		
	UTheta = np.zeros(n,dtype='float32')
	p = 0
	for i in range(0,nt):
		n = np.size(Theta[i])	
		UTheta[p:p+n] = Theta[i].flatten()
		p+=n

	return UTheta
