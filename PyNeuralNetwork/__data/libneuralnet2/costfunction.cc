#include "costfunction.h"

float CrossEntropyCost(Matrix &h, Matrix &y, MatrixArray &Theta, float L1, float L2) {
	/*******************************************************************
	 * This function will calculate the cross-entropy cost function.
	 * 
	 * Inputs:
	 * 		h - Matrix containing the ouputs of the neural network, 
	 * 			shape (m,K), where m = number of samples, K = number of
	 * 			output nodes.
	 *		y - Matrix containing the one-hot target values for h,
	 * 			shape (m,K).
	 *		&Theta - MatrixArray object containing network weights.
	 * 		L1 - L1 regularization parameter.
	 * 		L2 - L2 regularization parameter.
	 * 
	 * Output:
	 * 		J - cost (set L1=0.0 and L2=0.0 for classification cost). 
	 ******************************************************************/
	
	float J = 0.0;
	int m, K;
	m = y->shape[0];
	K = y->shape[1];
	
	/* calculate classification cost first*/
	for (i=0;i<K;i++) {
		for (j=0;j<m;j++) {
			J += -y->data[j,i]*cliplog(h->data[j,i],1e-40) - (1.0-y->data[j,i])*cliplog((1.0-h->data[j,i]),1e-40);
		}
	}
	J/=((float) m);
	
	/*Calculate L1 cost*/
	if (L1 > 0.0) {
		J += L1Regularization(Theta,L1,m);
	}
	
	/*Calculate L2 cost*/
	if (L2 > 0.0) {
		J += L1Regularization(Theta,L2,m);
	}
	
	return J;
}

void CrossEntropyDelta(Matrix &h, Matrix &y, float *InvAFGrad, Matrix &Deltas) {
	/*******************************************************************
	 * Calculate delta for this cost function (easy because it is just
	 * the target subtracted from the result).
	 ******************************************************************/
	
	MatrixSubtract(h,y,false,false,Deltas);
}


float MeanSquaredCost(Matrix &h, Matrix &y, MatrixArray &Theta, float L1, float L2) {
	/*******************************************************************
	 * This function will calculate the cross-entropy cost function.
	 * 
	 * Inputs:
	 * 		h - Matrix containing the ouputs of the neural network, 
	 * 			shape (m,K), where m = number of samples, K = number of
	 * 			output nodes.
	 *		y - Matrix containing the one-hot target values for h,
	 * 			shape (m,K).
	 *		&Theta - MatrixArray object containing network weights.
	 * 		L1 - L1 regularization parameter.
	 * 		L2 - L2 regularization parameter.
	 * 
	 * Output:
	 * 		J - cost (set L1=0.0 and L2=0.0 for classification cost). 
	 ******************************************************************/
	
	float J = 0.0;
	int m, K;
	m = y->shape[0];
	K = y->shape[1];
	
	/* calculate classification cost first*/
	for (i=0;i<K;i++) {
		for (j=0;j<m;j++) {
			J += powf(h->data[j,i]-y->data[j,i],2.0);
		}
	}
	J/=((float) 2*m*K);
	
	/*Calculate L1 cost*/
	if (L1 > 0.0) {
		J += L1Regularization(Theta,L1,m);
	}
	
	/*Calculate L2 cost*/
	if (L2 > 0.0) {
		J += L1Regularization(Theta,L2,m);
	}
	
	return J;
}

void MeanSquaredDelta(Matrix &h, Matrix &y, float *InvAFGrad, Matrix &Deltas) {
	/*******************************************************************
	 * Calculate delta for this cost function.
	 ******************************************************************/
	MatrixSubtract(h,y,false,false,Deltas);
	/*This Delta calculation requires finding the gradient of the 
	 * inverse of the activation function*/
	int i,j;
	for (i=0;i<h.shape[0];i++) {
		for (j=0;j<h.shape[1];j++) {
			Deltas.data[i,j] = Deltas.data[i,j]*InvAFGrad(h.data[i,j]);
		}
	}
}

