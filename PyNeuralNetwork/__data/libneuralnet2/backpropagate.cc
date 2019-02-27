#include "backpropagate.h"

void BackPropagate(MatrixArray &ThetaW, MatrixArray &ThetaB, 
	MatrixArray &Deltas, MatrixArray &a, Matrix y, ActFunc *AFgrad, 
	CostFuncDelta *CFDelta, double L1, double L2, 
	MatrixArray &ThetaWGrad, MatrixArray &ThetaBGrad) {
	
	int m = a.matrix[0].shape[0];
	
	/*create a matrix of ones*/
	Matrix ones(1,m);
	ones.AddScalar(1.0);
	
	/* Find out the number of layers*/
	int L = ThetaW.n + 1;
	
	/*loop through each layer backwards, calculating the deltas, then gradients*/
	int i;
	for (i=L-2;i>=0;i--) {
		if (i == L-2) {
			/*This would be the final layer, use CF delta*/
			CFDelta(a[L-1],y,AFGrad[L-1],Deltas[i]);
		} else {
			/*calculate deltas using previous deltas */
			_BackPropDeltas(*Deltas.matrix[i+1],*ThetaW.matrix[i+1],AF[i+1],*a.matrix[i+1],*Deltas.matrix[i]);
		}
		/*calculate weight gradients*/
		MatrixDot(*a.matrix[i],*Deltas.matrix[i],true,false,*ThetaWGrad.matrix[i]);
		ThetaWGrad.matrix[i].DivideScalar((double) m);
		ApplyRegGradToMatrix(ThetaW,ThetaWGrad,L1,L2,m);
		
		/*calculate bias gradients*/
		MatrixDot(ones,*Deltas.matrix[i],false,false,*ThetaBGrad.matrix[i]);
		ThetaBGrad.matrix[i].DivideScalar((double) m);
	}
		
	
}

void _BackPropDeltas(Matrix &dlin, Matrix &ThetaW, ActFunc AFGrad, Matrix &a, Matrix &dlout) {
	MatrixDot(dlin,ThetaW,false,true,dlout);
	int i, j, ni, nj;
	ni = dlout.shape[0];
	nj = dlout.shape[1];
	
	for (i=0;i<ni;i++) {
		for (j=0;j<nj;j++) {
			dlout.data[i][j] = dlout.data[i][j]*AF(a.data[i][j]);
		}
	}
	
} 
