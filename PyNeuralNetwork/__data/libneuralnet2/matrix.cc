#include "matrix.h"

//initialize matrix with shape, fill with zeros
Matrix::Matrix(int *inshape) {
	int i;
	Matrix::shape[0] = inshape[0];
	Matrix::shape[1] = inshape[1];
	Matrix::size = inshape[0]*inshape[1];
	Matrix::data = new double*[Matrix::shape[0]];
	for (i=0;i<Matrix::shape[0];i++) {
		Matrix::data[i] = new double[Matrix::shape[1]];
	}
	Matrix::DeleteData = true;
	Matrix::FillZeros();
}

Matrix::Matrix(int x, int y) {
	int i;
	Matrix::shape[0] = x;
	Matrix::shape[1] = y;
	Matrix::size = x*y;
	Matrix::data = new double*[Matrix::shape[0]];
	for (i=0;i<Matrix::shape[0];i++) {
		Matrix::data[i] = new double[Matrix::shape[1]];
	}
	Matrix::DeleteData = true;
	Matrix::FillZeros();
}


Matrix::Matrix(int *inshape, double **matrix) {
	int i;
	Matrix::shape[0] = inshape[0];
	Matrix::shape[1] = inshape[1];
	Matrix::size = inshape[0]*inshape[1];
	Matrix::data = matrix;
	Matrix::DeleteData = false;
}

Matrix::Matrix(int x, int y, double **matrix) {
	int i;
	Matrix::shape[0] = x;
	Matrix::shape[1] = y;
	Matrix::size = x*y;
	Matrix::data = matrix;
	Matrix::DeleteData = false;
}

//copy constructor
Matrix::Matrix(const Matrix &obj) {
	printf("Matrix copy constructor called!");
	Matrix::shape[0] = obj.shape[0];
	Matrix::shape[1] = obj.shape[1];
	Matrix::size = obj.size;
	Matrix::data = new double*[Matrix::shape[0]];
	int i,j;
	for (i=0;i<Matrix::shape[0];i++) {
		Matrix::data[i] = new double[Matrix::shape[1]];
	}

	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) Matrix::shape[0])/maxthreads);
	#pragma omp parallel private(i,j)
	{
		#pragma omp for schedule(dynamic,chunk) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] = obj.data[i][j];
			}
		}
	}
}


//must deallocate the data array
Matrix::~Matrix() {
	if (Matrix::DeleteData) {
		int i;
		for (i=0;i<Matrix::shape[0];i++) {
			delete Matrix::data[i];
		}
		delete[] Matrix::data;
	}
}

void Matrix::FillZeros() {
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) Matrix::shape[0])/maxthreads);
	int i,j;
	#pragma omp parallel private(i,j)
	{
		
		#pragma omp for schedule(dynamic,chunk) 
		
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] = 0.0;
			}
		}
	}
}

void Matrix::TimesScalar(double x) {
	int i,j;
	#pragma omp parallel private(i,j)
	{
		
		#pragma omp for schedule(dynamic) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] *= x;
			}
		}
	}
}	

void Matrix::DivideScalar(double x) {
	int i,j;
	#pragma omp parallel private(i,j)
	{
		
		#pragma omp for schedule(dynamic) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] /= x;
			}
		}
	}
}	

void Matrix::AddScalar(double x) {
	int i,j;
	#pragma omp parallel private(i,j)
	{
		#pragma omp for schedule(dynamic) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] += x;
			}
		}
	}
}	

void Matrix::SubtractScalar(double x) {
	int i,j;
	#pragma omp parallel private(i,j)
	{
		#pragma omp for schedule(dynamic) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] -= x;
			}
		}
	}
}	

void Matrix::SubtractFromScalar(double x) {
	int i,j;
	#pragma omp parallel private(i,j)
	{
		#pragma omp for schedule(dynamic) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] = x - Matrix::data[i][j];
			}
		}
	}
}	

void Matrix::PrintMatrix() {
	int i,j;
	for (i=0;i<Matrix::shape[0];i++) {
		for (j=0;j<Matrix::shape[1];j++) {
			printf("%10.5f ",Matrix::data[i][j]);
		}
		printf("\n");
	}
}

void Matrix::PrintMatrix(const char *str) {
	int i,j;
	printf("%s\n",str);
	printf("shape = (%d,%d)\n",Matrix::shape[0],Matrix::shape[1]);
	for (i=0;i<Matrix::shape[0];i++) {
		for (j=0;j<Matrix::shape[1];j++) {
			printf("%10.5f ",Matrix::data[i][j]);
		}
		printf("\n");
	}
}

void Matrix::CopyMatrix(Matrix &m) {
	int i,j;
	#pragma omp parallel private(i,j)
	{
		#pragma omp for schedule(dynamic) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] = m.data[i][j];
			}
		}
	}
}

void Matrix::FillMatrix(double **filldata) {
	int i,j;
	#pragma omp parallel private(i,j)
	{
		#pragma omp for schedule(dynamic) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				Matrix::data[i][j] = filldata[i][j];
			}
		}
	}
}


void Matrix::ReturnMatrix(double **outdata) {
	int i,j;
	#pragma omp parallel private(i,j)
	{
		#pragma omp for schedule(dynamic) 
		for (i=0;i<Matrix::shape[0];i++) {
			for (j=0;j<Matrix::shape[1];j++) {
				outdata[i][j] = Matrix::data[i][j];
			}
		}
	}
}
