#include <benchmark/benchmark.h>
#include <omp.h>
#include <math.h>
#include "spray.hpp"

static void init_test(int numthreads, int stencilsize, int domainsize, real*& inb, real*& outb, real*& weightsl, real*& weightsr, real& weightc) {
  omp_set_dynamic(0);
  omp_set_num_threads(numthreads);
  inb = (real*) aligned_alloc(512,domainsize * sizeof(real));
  outb = (real*) aligned_alloc(512,domainsize * sizeof(real));
  weightsl = (real*) malloc(stencilsize * sizeof(real));
  weightsr = (real*) malloc(stencilsize * sizeof(real));
  for(int i=0; i<domainsize; i++) {
    outb[i] = sin(0.1*i);
    inb[i] = 0.0;
  }
  for(int i=0; i<stencilsize; i++) {
    weightsl[i] = sin(0.1*i);
    weightsr[i] = cos(0.1*i);
  }
  weightc = -2.0;
}

static void destroy_test(real* inb, real* outb, real* weightsl, real* weightsr) {
  free(inb);
  free(outb);
  free(weightsl);
  free(weightsr);
}

static void BM_serial(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb[i - j - 1] += wl[j]*outb[i];
                inb[i + j + 1] += wr[j]*outb[i];
            }
            inb[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_omp(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        #pragma omp parallel for reduction(+:inb[0:N])
        for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb[i - j - 1] += wl[j]*outb[i];
                inb[i + j + 1] += wr[j]*outb[i];
            }
            inb[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_atomic(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        #pragma omp parallel for
        for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                #pragma omp atomic update
                inb[i - j - 1] += wl[j]*outb[i];
                #pragma omp atomic update
                inb[i + j + 1] += wr[j]*outb[i];
            }
            #pragma omp atomic update
            inb[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_blocks(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BlockReduction<real> inb_b(N,inb,false);
        #pragma omp parallel for reduction(+:inb_b)
        for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_locks(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BlockReduction<real> inb_b(N,inb,true);
        #pragma omp parallel for reduction(+:inb_b)
        for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_catomic(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::AtomicReduction<real> inb_b(inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_cdense(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::DenseReduction<real> inb_b(N,inb);
        #pragma omp parallel for reduction(+:inb_b)
        for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_map(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::STLMapReduction<real> inb_b(inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_btree(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BtreeReduction<real> inb_b(inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_keeper(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::KeeperReduction<real> inb_b(N,inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

static void BM_awblck16(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BlockReduction16<real> inb_b(N,inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}
static void BM_awblck64(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BlockReduction64<real> inb_b(N,inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}
static void BM_awblck256(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BlockReduction256<real> inb_b(N,inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}
static void BM_awblck1024(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BlockReduction1024<real> inb_b(N,inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}
static void BM_awblck4096(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BlockReduction4096<real> inb_b(N,inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}
static void BM_awblck16384(benchmark::State& state) {
    real *inb, *outb, *wl, *wr, wc;
    int S = state.range(1);
    int N = state.range(2);
    init_test(state.range(0), S, N, inb, outb, wl, wr, wc);
    for (auto _ : state) {
        spray::BlockReduction4096<real> inb_b(N,inb);
        #pragma omp parallel for reduction(+:inb_b)
        for(int i=S; i<N-S; i++) {
        //for (int i = N-S-1; i > S-1; --i) {
            for (int j = S-1; j > -1; --j) {
                inb_b[i - j - 1] += wl[j]*outb[i];
                inb_b[i + j + 1] += wr[j]*outb[i];
            }
            inb_b[i] += wc*outb[i];
        }
    }
    destroy_test(inb, outb, wl, wr);
}

BENCHMARK(BM_serial )->ArgsProduct({{1               },{1},{10000000}})->UseRealTime();
BENCHMARK(BM_omp    )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_atomic )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_blocks )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_locks  )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_catomic)->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_cdense )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_keeper )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_awblck16 )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_awblck256 )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_awblck1024 )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_awblck4096 )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
BENCHMARK(BM_awblck16384 )->ArgsProduct({{1,2,4,8,16,28,56},{1},{10000000}})->UseRealTime();
//BENCHMARK(BM_map    )->ArgsProduct({{1,2,4,8,16,28,56},{1},{1000000}})->UseRealTime();
//BENCHMARK(BM_btree  )->ArgsProduct({{1,2,4,8,16,28,56},{1},{1000000}})->UseRealTime();

BENCHMARK_MAIN();
