#include "regularization.h"

float L1Regularization(MatrixArray &Theta, float L1, int m) { 
	float J = 0.0;
	
	int i,j,k,nj,nk;
	for (i=0;i<Theta->n;i++) {
		/*get matrix dimensions*/
		nj = Theta->matrix[i]->shape[0];
		nk = Theta->matrix[i]->shape[1];
		for (j=0;j<nj;j++) {
			for (k=0;k<nk;k++) {
				J += fabsf(Theta->matrix[i]->data[j][k]);
			}
		}
	}
	
	return L1*J/((float) m*2.0);
}


float L2Regularization(MatrixArray &Theta, float L2, int m) { 
	float J = 0.0;
	
	int i,j,k,nj,nk;
	for (i=0;i<Theta->n;i++) {
		/*get matrix dimensions*/
		nj = Theta->matrix[i]->shape[0];
		nk = Theta->matrix[i]->shape[1];
		for (j=0;j<nj;j++) {
			for (k=0;k<nk;k++) {
				J += powf(Theta->matrix[i]->data[j][k],2.0);
			}
		}
	}
	
	return L2*J/((float) m*2.0);
}
