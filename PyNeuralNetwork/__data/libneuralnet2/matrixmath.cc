#include "matrixmath.h"


void _Multab(Matrix &a, Matrix&b, Matrix &out) {
	int *oshape = out.shape;
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) oshape[0])/maxthreads);
	#pragma omp parallel 
	{
		int i,j;
		#pragma omp for schedule(dynamic,chunk) 
		for (i=0;i<oshape[0];i++) {
			for (j=0;j<oshape[1];j++) {
				out.data[i][j] = a.data[i][j] * b.data[i][j];
			}
		}
	}
}	

void _MultaTb(Matrix &a, Matrix&b, Matrix &out) {
	int *oshape = out.shape;
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) oshape[0])/maxthreads);
	#pragma omp parallel 
	{
		int i,j;
		#pragma omp for schedule(dynamic,chunk) 
		for (i=0;i<oshape[0];i++) {
			for (j=0;j<oshape[1];j++) {
				out.data[i][j] = a.data[j][i] * b.data[i][j];
			}
		}
	}
}	

void _MultabT(Matrix &a, Matrix&b, Matrix &out) {
	int *oshape = out.shape;
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) oshape[0])/maxthreads);
	#pragma omp parallel 
	{
		int i,j;
		#pragma omp for schedule(dynamic,chunk) 
		for (i=0;i<oshape[0];i++) {
			for (j=0;j<oshape[1];j++) {
				out.data[i][j] = a.data[i][j] * b.data[j][i];
			}
		}
	}
}	

void _MultaTbT(Matrix &a, Matrix&b, Matrix &out) {
	int *oshape = out.shape;
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) oshape[0])/maxthreads);
	#pragma omp parallel 
	{
		int i,j;
		#pragma omp for schedule(dynamic,chunk) 
		for (i=0;i<oshape[0];i++) {
			for (j=0;j<oshape[1];j++) {
				out.data[i][j] = a.data[j][i] * b.data[j][i];
			}
		}
	}
}	



void MatrixMultiply(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out) {
	if (not aT && not bT) {
		//no transpose
		_Multab(a,b,out);
	} else if (aT && not bT) {
		//only aT
		_MultaTb(a,b,out);
	} else if (not aT && bT) {
		//only bT
		_MultabT(a,b,out);
	} else {
		//both transposed
		_MultaTbT(a,b,out);
	}
}



void _Dotab(Matrix &a, Matrix &b, Matrix &out) {
	int kdim;
	if (a.shape[1] == b.shape[0]) {
		kdim = a.shape[1];
	} else {
		printf("Warning! shape of input values should be a(i,j), b(j,k), this may produce strange results\n");
		if (a.shape[1] < b.shape[0]) {
			kdim = a.shape[1];
		} else {
			kdim = b.shape[0];
		}
	}
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) a.shape[0])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j,k;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<a.shape[0];i++) {
			for (j=0;j<b.shape[1];j++) {
				tmp = 0.0;
				for (k=0;k<kdim;k++) {
					tmp += a.data[i][k] * b.data[k][j];
				}
				out.data[i][j] = tmp;
			}
		}
	}
}		


void _DotaTb(Matrix &a, Matrix &b, Matrix &out) {
	int kdim;
	if (a.shape[0] == b.shape[0]) {
		kdim = a.shape[0];
	} else {
		printf("Warning! shape of input values should be a(i,j), b(j,k), this may produce strange results\n");
		if (a.shape[0] < b.shape[0]) {
			kdim = a.shape[0];
		} else {
			kdim = b.shape[0];
		}
	}
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) a.shape[1])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j,k;
		double tmp;
		#pragma omp for schedule(dynamic,chunk) 
		for (i=0;i<a.shape[1];i++) {
			for (j=0;j<b.shape[1];j++) {
				tmp = 0.0;
				for (k=0;k<kdim;k++) {
					tmp += a.data[k][i] * b.data[k][j];
				}
				out.data[i][j] = tmp;
			}
		}
	}
}		


void _DotabT(Matrix &a, Matrix &b, Matrix &out) {
	int kdim;
	if (a.shape[1] == b.shape[1]) {
		kdim = a.shape[1];
	} else {
		printf("Warning! shape of input values should be a(i,j), b(j,k), this may produce strange results\n");
		if (a.shape[1] < b.shape[1]) {
			kdim = a.shape[1];
		} else {
			kdim = b.shape[1];
		}
	}
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) a.shape[0])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j,k;
		double tmp;
		#pragma omp for schedule(dynamic,chunk) 
		for (i=0;i<a.shape[0];i++) {
			for (j=0;j<b.shape[0];j++) {
				tmp = 0.0;
				for (k=0;k<kdim;k++) {
					tmp += a.data[i][k] * b.data[j][k];
				}
				out.data[i][j] = tmp;
			}
		}
	}
}		

void _DotaTbT(Matrix &a, Matrix &b, Matrix &out) {
	int kdim;
	if (a.shape[0] == b.shape[1]) {
		kdim = a.shape[0];
	} else {
		printf("Warning! shape of input values should be a(i,j), b(j,k), this may produce strange results\n");
		if (a.shape[0] < b.shape[1]) {
			kdim = a.shape[0];
		} else {
			kdim = b.shape[1];
		}
	}
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) a.shape[1])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j,k;
		double tmp;
		#pragma omp for schedule(dynamic,chunk) 
		for (i=0;i<a.shape[1];i++) {
			for (j=0;j<b.shape[0];j++) {
				tmp = 0.0;
				for (k=0;k<kdim;k++) {
					tmp += a.data[k][i] * b.data[j][k];
				}
				out.data[i][j] = tmp;
			}
		}
	}
}		

void MatrixDot(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out) {
	if (not aT && not bT) {
		//no transpose
		_Dotab(a,b,out);
	} else if (aT && not bT) {
		//only aT
		_DotaTb(a,b,out);
	} else if (not aT && bT) {
		//only bT
		_DotabT(a,b,out);
	} else {
		//both transposed
		_DotaTbT(a,b,out);
	}
}



void _Subab(Matrix &a, Matrix &b, Matrix &out) {
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[0])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] = a.data[i][j] - b.data[i][j];
			}
		}
	}
}	

void _SubaTb(Matrix &a, Matrix &b, Matrix &out) {
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[1])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] = a.data[j][i] - b.data[i][j];
			}
		}
	}
}	

void _SubabT(Matrix &a, Matrix &b, Matrix &out) {
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[0])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] = a.data[i][j] - b.data[j][i];
			}
		}
	}
}	

void _SubaTbT(Matrix &a, Matrix &b, Matrix &out) {
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[0])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] = a.data[j][i] - b.data[j][i];
			}
		}
	}
}	

void MatrixSubtract(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out) {
	if (not aT && not bT) {
		//no transpose
		_Subab(a,b,out);
	} else if (aT && not bT) {
		//only aT
		_SubaTb(a,b,out);
	} else if (not aT && bT) {
		//only bT
		_SubabT(a,b,out);
	} else {
		//both transposed
		_SubaTbT(a,b,out);
	}
}

void _Addab(Matrix &a, Matrix &b, Matrix &out) {
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[0])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] = a.data[i][j] + b.data[i][j];
			}
		}
	}
}	

void _AddaTb(Matrix &a, Matrix &b, Matrix &out) {
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[1])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] = a.data[j][i] + b.data[i][j];
			}
		}
	}
}	

void _AddabT(Matrix &a, Matrix &b, Matrix &out) {
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[0])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] = a.data[i][j] + b.data[j][i];
			}
		}
	}
}	

void _AddaTbT(Matrix &a, Matrix &b, Matrix &out) {
    omp_set_num_threads(8);
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[0])/(maxthreads*10));
	#pragma omp parallel 
	{
		int i,j;
		double tmp;
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] = a.data[j][i] + b.data[j][i];
			}
		}
	}
}	

void MatrixAdd(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out) {
	if (not aT && not bT) {
		//no transpose
		_Addab(a,b,out);
	} else if (aT && not bT) {
		//only aT
		_AddaTb(a,b,out);
	} else if (not aT && bT) {
		//only bT
		_AddabT(a,b,out);
	} else {
		//both transposed
		_AddaTbT(a,b,out);
	}
}


void ApplyFunctionToMatrix(Matrix &a, ActFunc AF, Matrix &o) {
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) out.shape[0])/(maxthreads*10));
	int i,j;
	#pragma omp parallel private(i,j)
	{

		#pragma omp for schedule(static,chunk) 
		for (i=0;i<out.shape[0];i++) {
			for (j=0;j<out.shape[1];j++) {
				out.data[i][j] =AF(a.data[j][i]);
			}
		}
	}	
}

void ApplyFunctionToMatrix(Matrix &a, ActFunc AF) {
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) a.shape[0])/(maxthreads*10));
	int i,j;
	#pragma omp parallel private(i,j)
	{

		#pragma omp for schedule(static,chunk) 
		for (i=0;i<a.shape[0];i++) {
			for (j=0;j<a.shape[1];j++) {
				a.data[i][j] =AF(a.data[j][i]);
			}
		}
	}	
}

void AddBiasVectorToMatrix(Matrix &a, Matrix &b) {
	/*******************************************************************
	 * Adds a bias vector of shape (1,n) to a matrix of shape (m,n).
	 * ****************************************************************/
	
	int maxthreads = omp_get_max_threads();
	int chunk = ceil(((double) a.shape[0])/(maxthreads*10));
	int i,j;
	#pragma omp parallel private(i,j)
	{
		#pragma omp for schedule(static,chunk) 
		for (i=0;i<a.shape[0];i++) {
			for (j=0;j<a.shape[1];j++) {
				a.data[i][j] = a.data[i][j] + b.data[0][j];
			}
		}
	}		
}
