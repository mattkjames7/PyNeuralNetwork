#ifndef __ACTIVATIONFUNCTIONS_H__
#define __ACTIVATIONFUNCTIONS_H__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

double AF_LeakyReLU(double z);
double AF_LeakyReLUGradient(double z);
double AF_InverseLeakyReLU(double a);
double AF_InverseLeakyReLUGradient(double a);
double AF_Linear(double z);
double AF_LinearGradient(double z);
double AF_ReLU(double z);
double AF_ReLUGradient(double z);
double AF_Sigmoid(double z);
double AF_SigmoidGradient(double z);
double AF_InverseSigmoid(double a);
double AF_InverseSigmoidGradient(double a);
double AF_Softplus(double z);
double AF_SoftplusGradient(double z);
double AF_InverseSoftplus(double a);
double AF_InverseSoftplusGradient(double a);
double AF_Tanh(double z);
double AF_TanhGradient(double z);
double AF_InverseTanh(double a);
double AF_InverseTanhGradient(double a);

typedef double (*ActFunc)(double);
#endif
