#include "propagate.h"

void Propagate(MatrixArray &a, MatrixArray &z, MatrixArray &ThetaW, MatrixArray &ThetaB, ActFunc *AF) {
	/*******************************************************************
	 * If it works as planned, then this should propagate the data 
	 * through the neural network.
	 * ****************************************************************/
	int i, L;
	L = a.n;
	for (i=0;i<L-1;i++) {
		MatrixDot(*a.matrix[i],*ThetaW.matrix[i],false,false,*z.matrix[i]); //a[i].shape = (m,s[i]),ThetaW[i].shape = (s[i],s[i+1]), z[i].shape = (m,s[i+1])
		AddBiasVectorToMatrix(*z.matrix[i],*ThetaB.matrix[i]); //z[i].shape = (m,s[i+1]), ThetaB[i].shape = (1,s[i+1])
		ApplyFunctionToMatrix(*z.matrix[i],AF[i],*a.matrix[i+1]); //z[i].shape = (m,s[i+1]), a[i+1].shape = (m,s[i+1])
	}
}
