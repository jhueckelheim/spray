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

int main (int argc,char **argv){
    int count = atoi(argv[2]);
    double time;           
    coo<real> coo_data;
    csr<real> csr_data;
    read_mm_coo(argv[1],coo_data);
    coocsr(coo_data, csr_data);

    real* res = new real[coo_data.nr];
    for (int i = 0; i < coo_data.nr; i++) {
        res[i] = 0;
    }
    real* x = new real[coo_data.nc];
    for (int i = 0; i < coo_data.nc; i++) {
        x[i] = sin(i);
    }
    for (int c = 0; c <count; c++){
        #pragma omp parallel for
        for (int i = 0; i < coo_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                res[i] += csr_data.vals[k]*x[csr_data.cols[k]];
            }
        }
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        #pragma omp parallel for
        for (int i = 0; i < coo_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                res[i] += csr_data.vals[k]*x[csr_data.cols[k]];
            }
        }
    }
    time = omp_get_wtime() - time;
    printf("spmv time on %d threads: %f\n",omp_get_max_threads(),time);
    
    for (int c = 0; c <count; c++){
        #pragma omp parallel for reduction(+:x[0:coo_data.nc])
        for (int i = 0; i < coo_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        #pragma omp parallel for reduction(+:x[0:coo_data.nc])
        for (int i = 0; i < coo_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    time = omp_get_wtime() - time;
    printf("sptmv_ompreduce time on %d threads: %f\n",omp_get_max_threads(),time);
    
    for (int c = 0; c <count; c++){
        #pragma omp parallel for
        for (int i = 0; i < coo_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                #pragma omp atomic
                x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    time = omp_get_wtime();
    for (int c = 0; c <count; c++){
        #pragma omp parallel for
        for (int i = 0; i < coo_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                #pragma omp atomic
                x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    time = omp_get_wtime() - time;
    printf("sptmv_atomic time on %d threads: %f\n",omp_get_max_threads(),time);
    
    {
        BlockArray<real> x_p(coo_data.nc, x);
        for (int c = 0; c <count; c++){
            #pragma omp parallel for reduction(+:x_p)
            for (int i = 0; i < coo_data.nr; i++) {
                for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                    x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
                }
            }
        }
    }
    time = omp_get_wtime();
    {
        BlockArray<real> x_p(coo_data.nc, x);
        for (int c = 0; c <count; c++){
            #pragma omp parallel for reduction(+:x_p)
            for (int i = 0; i < coo_data.nr; i++) {
                for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                    x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
                }
            }
        }
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
