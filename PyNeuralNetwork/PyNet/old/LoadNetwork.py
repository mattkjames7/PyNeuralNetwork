import numpy as np
import os
from .NeuralNetwork import NeuralNetwork

def LoadNetwork(FileName):
	if os.path.isfile(FileName) == False:
		print('file not found')
		return None
	f = open(FileName,'rb')
	Trained = np.fromfile(f,dtype='bool8',count=1)[0]
	L = np.fromfile(f,dtype='int32',count=1)[0]
	s = np.fromfile(f,dtype='int32',count=L)
	Lambda = np.fromfile(f,dtype='float32',count=1)[0]
	Range = np.fromfile(d,dtype='float32',count=1)[0]
	mt = np.fromfile(f,dtype='int32',count=1)[0]
	mcv = np.fromfile(f,dtype='int32',count=1)[0]

	if mt > 0:
		Xt = np.fromfile(f,dtype='float32',count=mt*(s[0]+1)).reshape((mt,s[0]+1))
		yt = np.fromfile(f,dtype='float32',count=mt)
	else:
		Xt = np.array([],dtype='float32')
		yt = np.array([],dtype='float32')
	if mcv > 0:
		Xcv = np.fromfile(f,dtype='float32',count=mcv*(s[0]+1)).reshape((mcv,s[0]+1))
		ycv = np.fromfile(f,dtype='float32',count=mcv)
	else:
		Xcv = np.array([],dtype='float32')
		ycv = np.array([],dtype='float32')	
		
			
	Theta = []
	for i in range(0,L-1):
		dim = [s[i+1],s[i]+1]
		Theta.append(np.fromfile(f,dtype='float32',count=dim[0]*dim[1]).reshape((dim[0],dim[1])))
		
	nSteps = np.fromfile(f,dtype='int32',count=1)[0]
	nJ = np.fromfile(f,dtype='int32',count=1)[0]
	
	if nSteps > 0:
		Jt = np.fromfile(f,dtype='float32',count=nJ) 
		Jcv = np.fromfile(f,dtype='float32',count=nJ) 
		Acct = np.fromfile(f,dtype='float32',count=nJ) 
		Acccv = np.fromfile(f,dtype='float32',count=nJ) 
	else:
		Jt = np.array([],dtype='float32')
		Jcv	= np.array([],dtype='float32')
		Acct = np.array([],dtype='float32')
		Acccv= np.array([],dtype='float32')



	f.close()
	
	
	net = NeuralNetwork(s,Lambda)
	net.Trained = Trained
	net.mt = mt
	net.mcv = m
	net.Xt = Xt
	net.yt = yt
	net.Xcv = Xcv
	net.ycv = ycv
	net.Theta = Theta
	net.Jt = Jt
	net.Jcv = Jcv
	net.Acct = Acct
	net.Acccv = Acccv
	
	return net
