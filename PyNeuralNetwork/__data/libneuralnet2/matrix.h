#ifndef __MATRIX_H_INCLUDED__
#define __MATRIX_H_INCLUDED__
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

class Matrix {
	public:
		int shape[2];
		int size;
		Matrix(int*);
		Matrix(int,int);
		Matrix(int*,double**);
		Matrix(int,int,double**);
		Matrix(const Matrix &obj);
		~Matrix();
		void FillZeros();
		void TimesScalar(double);
		void DivideScalar(double);
		void AddScalar(double);
		void SubtractScalar(double);
		void SubtractFromScalar(double);
		void PrintMatrix();
		void PrintMatrix(const char *);
		void CopyMatrix(Matrix&);
		void FillMatrix(double**);
		void ReturnMatrix(double**);
		double **data = NULL;
	private:
		bool DeleteData;
		
};

#endif
