#ifndef __ARRAYTOOLS_H__
#define __ARRAYTOOLS_H__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

template <class T> T* CreateArray(T *a, int n) {
	T *tmp = new T[n];
	return tmp;
}

template <class T> void DestroyArray(T *a) {
	delete[] a;
}


template <class T> T** Create2DArray(T **a, int n0, int n1) {
	T **tmp = new T*[n0];
	int i;
	for (i=0;i<n0;i++) {
		tmp[i] = new T[n1];
	}
	return tmp;
}

template <class T> void Destroy2DArray(T **a, int n0) {
	int i;
	for (i=0;i<n0;i++) {
		delete [] a[i];
	}
	delete[] a;
}

template <class T> T* AppendToArray(T *a, int na, T *b, int nb, int *n) {
	n[0] = na + nb; //size of new array
	printf("%d + %d -> %d\n",na,nb,n[0]);
	int *tmp = new T[n[0]]; //temporary array
	if (na > 0) {
		memcpy(tmp,a,na*sizeof(T)); //copy contents of a to tmp
		delete[] a; // delete original array
	}
	memcpy(&tmp[na],b,nb*sizeof(T)); //copy contents of b to tmp
	a = tmp; //transfer pointer to new array
	// must return pointer
	return a;
}


#endif
