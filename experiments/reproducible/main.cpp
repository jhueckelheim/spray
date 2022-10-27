#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
using namespace std;
typedef double real;

real reduce(int n_iter, int n_th) {
    real res = 0;
    #pragma omp parallel for reduction(+:res) num_threads(n_th)
    for(int i=0; i<n_iter; i++) {
        res += sin(real(i));
    }
    return res;
}

real reduce_repro(int n_iter, int n_th) {
    real res[n_iter];
    #pragma omp parallel for num_threads(n_th)
    for(int i=0; i<n_iter; i++) {
        res[i] = sin(real(i));
    }
    for(int gr=2; gr/2 < n_iter; gr *= 2) {
        #pragma omp parallel for num_threads(n_th)
        for(int i=0; i<n_iter-gr/2; i+=gr) {
            res[i] += res[i+gr/2];
	}
    }
    return res[0];
}

int main (int argc,char **argv){
    if(argc < 2) {
      printf("Usage: ./exec <num_iter> <num_reps>\n");
      exit(-1);
    }

    int n_iter = atoi(argv[1]);
    int n_reps = atoi(argv[2]);
    real res_ref = reduce_repro(n_iter, 1);
    real res_simple = reduce(n_iter, 1);
    printf("expect approx. %.17g, got %.17g from reproducible reducer.\n", res_simple, res_ref);

    real time = omp_get_wtime();
    for (int c = 0; c < n_reps; c++){
        int n_th = c % omp_get_max_threads();
        real res = reduce_repro(n_iter, n_th);
        if(res != res_ref) {
            printf("error in iter %d. expect %.17g, got %.17g with %d threads\n", c, res_ref, res, n_th);
        }
    }
    time = omp_get_wtime() - time;
    printf("time on %d threads: %f\n",omp_get_max_threads(), time);

    return 0;
}

