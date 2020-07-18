#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "coo_csr_reader.hpp"
#include "blockReduction.hpp"
using namespace std;

#ifdef _DOUBLE
typedef double real;
#else
typedef float real;
#endif

void spmv(csr<real> csr_data, real* res, real* x) {
    #pragma omp parallel for
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            res[i] += csr_data.vals[k]*x[csr_data.cols[k]];
        }
    }
}

void spmvt_omp(csr<real> csr_data, real* res, real* x) {
    #pragma omp parallel for reduction(+:x[0:csr_data.nc])
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

void spmvt_atomic(csr<real> csr_data, real* res, real* x) {
    #pragma omp parallel for
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            #pragma omp atomic
            x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

void spmvt_blocks(csr<real> csr_data, real* res, real* x) {
    BlockArray<real> x_p(csr_data.nc, x);
    #pragma omp parallel for reduction(+:x_p)
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

int main (int argc,char **argv){
    int count = atoi(argv[2]);
    double time;

    // read matrix from matrix market file, convert to CSR format    
    coo<real> coo_data;
    csr<real> csr_data;
    read_mm_coo(argv[1],coo_data);
    coocsr(coo_data, csr_data);

    // Initialize test vectors
    real* res = new real[coo_data.nr];
    for (int i = 0; i < coo_data.nr; i++) {
        res[i] = 0;
    }
    real* x = new real[coo_data.nc];
    for (int i = 0; i < coo_data.nc; i++) {
        x[i] = sin(i);
    }

    // Sparse Matrix Vector Product
    for (int c = 0; c <count; c++){
        spmv(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmv(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("spmv time on %d threads: %f\n",omp_get_max_threads(),time);
    
    // Sparse Transpose-Matrix Vector Product using standard OpenMP Reduction
    for (int c = 0; c <count; c++){
        spmvt_omp(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_omp(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_ompreduce time on %d threads: %f\n",omp_get_max_threads(),time);
    
    // Sparse Transpose-Matrix Vector Product using Atomics
    for (int c = 0; c <count; c++){
        spmvt_atomic(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_atomic(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_atomic time on %d threads: %f\n",omp_get_max_threads(),time);
    
    // Sparse Transpose-Matrix Vector Product using Block Sparse Reductions
    for (int c = 0; c <count; c++){
        spmvt_blocks(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_blocks(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_blockreduce time on %d threads: %f\n",omp_get_max_threads(),time);
    
    delete(csr_data.cols);
    delete(csr_data.vals);
    delete(csr_data.rptr);
    delete(coo_data.cols);
    delete(coo_data.vals);
    delete(coo_data.rows);
    delete(res);
    delete(x);
}
