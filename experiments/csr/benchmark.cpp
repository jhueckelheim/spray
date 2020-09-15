#include <benchmark/benchmark.h>
#include <omp.h>
#include <math.h>
#include "coo_csr_reader.hpp"
#include "spray.hpp"
#ifdef __INTEL_COMPILER
#include <mkl_spblas.h>
#endif

#ifdef _DOUBLE
typedef double real;
#else
typedef float real;
#endif

static void init_test(int numthreads, char* mfilename, csr<real>& csr_data, real*& res, real*& x) {
  coo<real> coo_data;
  read_mm_coo(mfilename,coo_data);
  coocsr(coo_data, csr_data);

  // Initialize test vectors
  res = (real*) malloc(coo_data.nr * sizeof(real));
  for (int i = 0; i < coo_data.nr; i++) {
    res[i] = 0;
  }
  x = (real*) malloc(coo_data.nc * sizeof(real));
  for (int i = 0; i < coo_data.nc; i++) {
    x[i] = sin(i);
  }
  omp_set_dynamic(0);
  omp_set_num_threads(numthreads);
}

static void destroy_test(real* res, real* x) {
  free(res);
  free(x);
}

static void BM_spmv(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        #pragma omp parallel for
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                res[i] += csr_data.vals[k]*x[csr_data.cols[k]];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_serial(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_omp(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        #pragma omp parallel for reduction(+:x[0:csr_data.nc])
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_atomic(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        #pragma omp parallel for
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                #pragma omp atomic
                x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_blocks(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        spray::BlockReduction<real> x_p(csr_data.nc, x, false);
        #pragma omp parallel for reduction(+:x_p)
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_locks(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        spray::BlockReduction<real> x_p(csr_data.nc, x, true);
        #pragma omp parallel for reduction(+:x_p)
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_catomic(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        spray::AtomicReduction<real> x_p(x);
        #pragma omp parallel for reduction(+:x_p)
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_cdense(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        spray::DenseReduction<real> x_p(csr_data.nc, x);
        #pragma omp parallel for reduction(+:x_p)
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_map(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        spray::STLMapReduction<real> x_p(x);
        #pragma omp parallel for reduction(+:x_p)
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_btree(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        spray::BtreeReduction<real> x_p(x);
        #pragma omp parallel for reduction(+:x_p)
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

static void BM_spmvt_keeper(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    for (auto _ : state) {
        spray::KeeperReduction<real> x_p(csr_data.nc, x);
        #pragma omp parallel for reduction(+:x_p)
        for (int i = 0; i < csr_data.nr; i++) {
            for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
                x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
            }
        }
    }
    destroy_test(res, x);
}

#ifdef __INTEL_COMPILER
static void BM_spmvt_mkl(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    char transpose = 't';
    for (auto _ : state) {
#ifdef _DOUBLE
        mkl_cspblas_dcsrgemv(&transpose, &csr_data.nr, csr_data.vals, csr_data.rptr, csr_data.cols, x, res);
#else
        mkl_cspblas_scsrgemv(&transpose, &csr_data.nr, csr_data.vals, csr_data.rptr, csr_data.cols, x, res);
#endif
    }
    destroy_test(res, x);
}

static void BM_spmvt_mkl_ie(benchmark::State& state, char* mfilename) {
    csr<real> csr_data;
    real *res, *x;
    init_test(state.range(0), mfilename, csr_data, res, x);
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
#ifdef _DOUBLE
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, csr_data.nr, csr_data.nc, csr_data.rptr, csr_data.rptr+1, csr_data.cols, csr_data.vals);
#else
    mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, csr_data.nr, csr_data.nc, csr_data.rptr, csr_data.rptr+1, csr_data.cols, csr_data.vals);
#endif
    mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_TRANSPOSE, descr, 100000);
    mkl_sparse_optimize(A);
    for (auto _ : state) {
#ifdef _DOUBLE
        mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, A, descr, x, 1.0, res);
#else
        mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, A, descr, x, 1.0, res);
#endif
    }
    mkl_sparse_destroy(A);
    destroy_test(res, x);
}
#endif

#define BM_MAT(matname, matfile) \
    BENCHMARK_CAPTURE(BM_spmvt_serial, matname, matfile)->Arg(1)->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_omp, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmv, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_serial, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_omp, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_atomic, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_blocks, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_locks, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_catomic, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_cdense, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_map, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_btree, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_keeper, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_mkl, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime(); \
    BENCHMARK_CAPTURE(BM_spmvt_mkl_ie, matname, matfile)->ArgsProduct({threadcounts})->UseRealTime();
#define threadcounts {1,2,4,8,16,28,56}

BM_MAT(s3dkt3m2, "s3dkt3m2.mtx")
BM_MAT(circuit5M, "circuit5M.mtx")
BM_MAT(debr, "debr.mtx")

BENCHMARK_MAIN();
