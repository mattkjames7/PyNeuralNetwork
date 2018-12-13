#ifndef __MATRIXMATH_H__
#define __MATRIXMATH_H__
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "matrix.h"
#include <omp.h>
using namespace std;

void MatrixMultiply(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out);
void MatrixDot(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out);
void MatrixAdd(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out);
void MatrixSubtract(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out);
#endif
