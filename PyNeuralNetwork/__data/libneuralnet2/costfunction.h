#ifndef __COSTFUNCTION_H__
#define __COSTFUNCTION_H__
#include <math.h>
#include <stdio.h>
#include "matrixarray.h"
#include "cliplog.h"
#include "propagate.h"
#include "regularization.h"
using namespace std;
float CrossEntropyCost(Matrix &h, Matrix &y, MatrixArray &Theta, double L1, double L2);
void CrossEntropyDelta(Matrix &h, Matrix &y, ActFunc InvAFGrad, Matrix &Deltas);
float MeanSquaredCost(Matrix &h, Matrix &y, MatrixArray &Theta, double L1, double L2);
void MeanSquaredDelta(Matrix &h, Matrix &y, ActFunc InvAFGrad, Matrix &Deltas);

typedef float (*CostFunc)(Matrix&,Matrix&,MatrixArray&,double, double);
typedef float (*CostFuncDelta)(Matrix&,Matrix&,ActFunc,Matrix&);
#endif
