#ifndef COO_CSR_READER_HPP
#define COO_CSR_READER_HPP
#include <stdio.h>
#include <stdlib.h>

template <typename T> 
struct coo { 
  T* vals;
  int * cols;
  int * rows;
  int nnz;
  int nr;
  int nc;
}; 

template <typename T> 
struct csr { 
  T* vals;
  int * cols;
  int * rptr;
  int nnz;
  int nr;
  int nc;
}; 

template <typename T>
void read_mm_coo (char* filename, coo<T>&);

void scanline(FILE *f, int *r, int *c, float *v);
void scanline(FILE *f, int *r, int *c, double *v);

template <typename T>
void coocsr(coo<T>& coo_data, csr<T>& csr_data);

#endif
