#ifndef __REGULARIZATION_H__
#define __REGULARIZATION_H__
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "matrix.h"
#include "matrixarray.h"
#include <omp.h>
using namespace std;

float L1Regularization(MatrixArray &Theta, float L1, int m);
float L2Regularization(MatrixArray &Theta, float L2, int m);
#endif
