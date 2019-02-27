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


void ApplyRegGradToMatrix(MatrixArray &ThetaW, MatrixArray &ThetaWGrad,double L1, double L2, int m) {
	
	int nA = ThetaW.n;
	int i, j, ni, nj, k;
	double C1 = L1/(2.0*m), C2 = L2/(2.0*m), l1;
	
	for (k=0;k<nA;k++) {
		ni = Theta.matrix[k].shape[0]
		nj = Theta.matrix[k].shape[1]
		for (i=0;i<ni;i++) {
			for (j=0;j<nj;j++) {
				if (Theta.matrix[k].data[i][j] >= 0) {
					l1 = 1.0;
				} else {
					l1 = -1.0;
				}
				ThetaWGrad.matrix[k].data[i][j] += C1*l1 + C2*ThetaW.matrix[k].data[i][j];
			}
		}
	}
	
}
	
}
