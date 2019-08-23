import numpy as np

def Softplus(z):
	#ReLU == Rectified linear unit
	#This is an approximation called the softplus function - the derivitive of which is the sigmoid function
	sh = np.shape(z)
	if len(sh) == 0:
		z = np.array([z])
	else:
		z = np.array(z)
#	if np.size(z) > 1:
#		bad = np.where(z > 50)
#		good = np.where(z <=50)
#		out = np.copy(z)
#		out[bad] = z[bad]
#		out[good] = np.log(1.0 + np.exp(z[good]))
#		return out
#	else:
#		if z > 50:
#			return z
#		else:
#			return np.log(1.0 + np.exp(z))
	
	out = np.zeros(z.shape,dtype=z.dtype)
	neg = np.where(z <= 0)
	pos = np.where(z > 0)
	out[neg] = np.log(1.0 + np.exp(z[neg]))
	out[pos] = np.log(np.exp(-z[pos]) + 1) + z[pos]
	return out
	
	
	
def SoftplusGradient(z):
	#ReLU == Rectified linear unit
	#This is an approximation called the softplus function - the derivitive of which is the sigmoid function

	sh = np.shape(z)
	if len(sh) == 0:
		z = np.array([z])
	else:
		z = np.array(z)
#	if np.size(z) > 1:
#		bad = np.where(z > 50)
#		good = np.where(z <=50)
#		out = np.copy(z)
#		out[bad] = 1
#		out[good] = 1.0/(1.0 + np.exp(-z[good]))
#		return out
#	else:
#		if z > 50:
#			return 1
#		else:
#			return 1.0/(1.0 + np.exp(-z))

	out = np.zeros(z.shape,dtype=z.dtype)
	neg = np.where(z < 0)
	pos = np.where(z >= 0)
	out[neg] = np.exp(z[neg])/(np.exp(z[neg]) + 1)
	out[pos] = 1.0/(1.0 + np.exp(-z[pos]))

	return out

def InverseSoftplus(a):
	sh = np.shape(a)
	if len(sh) == 0:
		a = np.array([a])
	else:
		a = np.array(a)	
	#z = np.log(np.exp(a) - 1.0)
	#return z
#	return np.log(1 - np.exp(-a)) + a
	out = np.zeros(a.shape,dtype=a.dtype)
	neg = np.where(a < 0)
	pos = np.where(a >= 0)
	out[neg] = np.log(np.exp(a[neg]) - 1.0)
	out[pos] = np.log(1 - np.exp(-a[pos])) + a[pos]

	return out	

def InverseSoftplusGradient(a):
	z = InverseSoftplus(a)
	return SoftplusGradient(z)
