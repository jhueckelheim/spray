#include "denseReduction.h"
#include <stdio.h>
#include <omp.h>

#define N 10

int main(int argc, char** argv) {
  double arr[N];
  spray_dense_double sp_arr;
  spray_dense_init(&sp_arr, &(arr[0]), N);
  for(int i=0;i<N;i++) {
    arr[i] = 0;
  }
  #pragma omp parallel for reduction(+:sp_arr)
  for(int i=1;i<N-1;i++) {
    double tid = 0.0*omp_get_thread_num();
    spray_dense_increment(&sp_arr, i-1, 1.0+tid);
    spray_dense_increment(&sp_arr, i,   2.0+tid);
    spray_dense_increment(&sp_arr, i+1, 4.0+tid);
  }
  for(int i=0;i<N;i++) {
    printf("%lf ",arr[i]);
  }
  printf("\n");
  return 0;
}
