#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "coo_csr_reader.hpp"
#include "spray.hpp"
#ifdef __INTEL_COMPILER
#include <mkl_spblas.h>
#endif
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

void spmvt_serial(csr<real> csr_data, real* res, real* x) {
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
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

void spmvt_blocks(csr<real> csr_data, real* res, real* x, bool useLocks = false) {
    spray::BlockReduction<real> x_p(csr_data.nc, x, useLocks);
    #pragma omp parallel for reduction(+:x_p)
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

void spmvt_containeratomic(csr<real> csr_data, real* res, real* x) {
    spray::AtomicReduction<real> x_p(x);
    #pragma omp parallel for reduction(+:x_p)
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

void spmvt_containerdense(csr<real> csr_data, real* res, real* x) {
    spray::DenseReduction<real> x_p(csr_data.nc, x);
    #pragma omp parallel for reduction(+:x_p)
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

void spmvt_map(csr<real> csr_data, real* res, real* x) {
    spray::STLMapReduction<real> x_p(x);
    #pragma omp parallel for reduction(+:x_p)
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

void spmvt_btree(csr<real> csr_data, real* res, real* x) {
    spray::BtreeReduction<real> x_p(x);
    #pragma omp parallel for reduction(+:x_p)
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

void spmvt_keeper(csr<real> csr_data, real* res, real* x) {
    spray::KeeperReduction<real> x_p(csr_data.nc, x);
    #pragma omp parallel for reduction(+:x_p)
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }
}

#ifdef __INTEL_COMPILER
void spmvt_mkl(csr<double> csr_data, double* res, double* x) {
    char transpose = 't';
    mkl_cspblas_dcsrgemv(&transpose, &csr_data.nr, csr_data.vals, csr_data.rptr, csr_data.cols, x, res);
}

void spmvt_mkl(csr<float> csr_data, float* res, float* x) {
    char transpose = 't';
    mkl_cspblas_scsrgemv(&transpose, &csr_data.nr, csr_data.vals, csr_data.rptr, csr_data.cols, x, res);
}

void spmvt_mkl_ie(csr<double> csr_data, double* res, double* x) {
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    if(SPARSE_STATUS_SUCCESS != mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, csr_data.nr, csr_data.nc, csr_data.rptr, csr_data.rptr+1, csr_data.cols, csr_data.vals) ||
       SPARSE_STATUS_SUCCESS != mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_TRANSPOSE, descr, 100000) ||
       SPARSE_STATUS_SUCCESS != mkl_sparse_optimize(A) ||
       SPARSE_STATUS_SUCCESS != mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, A, descr, x, 1.0, res) ||
       SPARSE_STATUS_SUCCESS != mkl_sparse_destroy(A)) {
        printf("Error in MKL I/E\n");
        exit(-1);
    }
}

void spmvt_mkl_ie(csr<float> csr_data, float* res, float* x) {
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    if(SPARSE_STATUS_SUCCESS != mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, csr_data.nr, csr_data.nc, csr_data.rptr, csr_data.rptr+1, csr_data.cols, csr_data.vals) ||
       SPARSE_STATUS_SUCCESS != mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_TRANSPOSE, descr, 100000) ||
       SPARSE_STATUS_SUCCESS != mkl_sparse_optimize(A) ||
       SPARSE_STATUS_SUCCESS != mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, A, descr, x, 1.0, res) ||
       SPARSE_STATUS_SUCCESS != mkl_sparse_destroy(A)) {
        printf("Error in MKL I/E\n");
        exit(-1);
    }
}

#endif

int main (int argc,char **argv){
    int count = atoi(argv[2]);
    double time;

    if(argc < 3) {
      printf("Usage: ./exec <matrix_file_name> <num_iter>\n");
      exit(1);
    }

    // read matrix from matrix market file, convert to CSR format    
    coo<real> coo_data;
    csr<real> csr_data;
    read_mm_coo(argv[1],coo_data);
    coocsr(coo_data, csr_data);

    // Initialize test vectors
    real* res = (real*) malloc(coo_data.nr * sizeof(real));
    for (int i = 0; i < coo_data.nr; i++) {
        res[i] = 0;
    }
    real* x = (real*) malloc(coo_data.nc * sizeof(real));
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

    // Sparse Transpose-Matrix Vector Product, sequential code
    for (int c = 0; c <count; c++){
        spmvt_serial(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_serial(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_serial time on %d threads: %f\n",omp_get_max_threads(),time);
    
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
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_blocks(csr_data, res, x, true);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_blocklockreduce time on %d threads: %f\n",omp_get_max_threads(),time);

    // Sparse Transpose-Matrix Vector Product using Container Atomics
    for (int c = 0; c <count; c++){
        spmvt_containeratomic(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_containeratomic(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_containeratomic time on %d threads: %f\n",omp_get_max_threads(),time);

    // Sparse Transpose-Matrix Vector Product using Container Dense
    for (int c = 0; c <count; c++){
        spmvt_containerdense(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_containerdense(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_containerdense time on %d threads: %f\n",omp_get_max_threads(),time);

    // Sparse Transpose-Matrix Vector Product using STL Maps
    for (int c = 0; c <count; c++){
        spmvt_map(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_map(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_map time on %d threads: %f\n",omp_get_max_threads(),time);

    // Sparse Transpose-Matrix Vector Product using Btree
    for (int c = 0; c <count; c++){
        spmvt_btree(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_btree(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_btree time on %d threads: %f\n",omp_get_max_threads(),time);

    // Sparse Transpose-Matrix Vector Product using Keeper Reduction
    for (int c = 0; c <count; c++){
        spmvt_keeper(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_keeper(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_keeper time on %d threads: %f\n",omp_get_max_threads(),time);

    
#ifdef __INTEL_COMPILER
    // Sparse Transpose-Matrix Vector Product using Intel MKL
    for (int c = 0; c <count; c++){
        spmvt_mkl(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_mkl(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_mkl time on %d threads: %f\n",omp_get_max_threads(),time);

    // Sparse Transpose-Matrix Vector Product using Intel MKL
    for (int c = 0; c <count; c++){
        spmvt_mkl_ie(csr_data, res, x);
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        spmvt_mkl_ie(csr_data, res, x);
    }
    time = omp_get_wtime() - time;
    printf("sptmv_mkl_ie time on %d threads: %f\n",omp_get_max_threads(),time);
#endif

    free(csr_data.cols);
    free(csr_data.vals);
    free(csr_data.rptr);
    free(coo_data.cols);
    free(coo_data.vals);
    free(coo_data.rows);
    free(res);
    free(x);
}
