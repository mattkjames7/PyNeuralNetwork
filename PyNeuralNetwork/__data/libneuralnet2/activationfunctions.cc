#include "activationfunctions.h"

double AF_LeakyReLU(double z) {
	return z*max((double) (z > 0),0.01);
}

double AF_LeakyReLUGradient(double z) {
	return max((double) (z > 0),0.01);
}

double AF_InverseLeakyReLU(double a) {
	return a/max((double) (a > 0),0.01);
}

double AF_InverseLeakyReLUGradient(double a) {
	return AF_LeakyReLUGradient(InverseLeakyReLU(a));
}

double AF_Linear(double z) {
	return z;
} 

double AF_LinearGradient(double z) {
	return 1.0;
} 

double AF_ReLU(double z) {
	return z*(double) (z > 0));
}

double AF_ReLUGradient(double z) {
	return ((double) (z > 0));
}

double AF_Sigmoid(double z) {
	return 1.0/(1.0 + exp(-z));
}

double AF_SigmoidGradient(double z) {
	return Sigmoid(z)*(1.0 - Sigmoid(z));
}

double AF_InverseSigmoid(double a) {
	return -log(1.0/a - 1.0);
}

double AF_InverseSigmoidGradient(double a) {
	return AF_SigmoidGradient(AF_InverseSigmoid(a));
}

double AF_Softplus(double z) { 
	if (z > 50.0) {
		return z;
	} else {
		return log(1.0 + exp(z));
	}
}

double AF_SoftplusGradient(double z) {
	if (z > 50.0) {
		return 1.0;
	} else {
		return 1.0/(1.0 + exp(-z));
	}
}	

double AF_InverseSoftplus(double a) {
	return log(exp(a) -1.0);
}

double AF_InverseSoftplusGradient(double a) {
	return AF_SoftplusGradient(AF_InverseSoftplus(a));
}

double AF_Tanh(double z) {
	return 2.0*AF_Sigmoid(z) - 1.0;
}

double AF_TanhGradient(double z) {
	return 1.0 - pow(AF_Tanh(z),2.0);
}

double AF_InverseTanh(double a) {
	return log(2.0/(a + 1.0) - 1.0)/(-2.0);
}

double AF_InverseTanhGradient(double a) {
	return AF_TanhGradient(AF_InverseTanh(a));
}
