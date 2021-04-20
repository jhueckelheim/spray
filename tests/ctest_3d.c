#include "ndDenseReduction.h"
#include <stdio.h>
#include <omp.h>

#define NI 10
#define NJ 10
#define NK 10

int main(int argc, char** argv) {
  double arr[NI][NJ][NK];
  for(int i=0;i<NI;i++) {
    for(int j=0;j<NJ;j++) {
      for(int k=0;k<NK;k++) {
        arr[i][j][k] = 0;
      }
    }
  }

  spray_nddense_double sp_arr;
  spray_nddense_init(&sp_arr, &(arr[0][0][0]), NI, NJ, NK);
  #pragma omp parallel for reduction(+:sp_arr)
  for(int i=1;i<NI-1;i++) {
    for(int j=1;j<NJ-1;j++) {
      for(int k=1;k<NK-1;k++) {
        double tid = 100000.0*(omp_get_thread_num()+1);
        spray_nddense_increment(&sp_arr, i,   j,   k,     1.0+tid);
        spray_nddense_increment(&sp_arr, i-1, j,   k,     2.0+tid);
        spray_nddense_increment(&sp_arr, i+1, j,   k,     4.0+tid);
        spray_nddense_increment(&sp_arr, i,   j-1, k,     8.0+tid);
        spray_nddense_increment(&sp_arr, i,   j+1, k,    16.0+tid);
        spray_nddense_increment(&sp_arr, i,   j,   k-1,  32.0+tid);
        spray_nddense_increment(&sp_arr, i,   j,   k+1,  64.0+tid);
      }
    }
  }

  for(int i=0;i<NI;i++) {
    for(int j=0;j<NJ;j++) {
      for(int k=0;k<NK;k++) {
        printf("%lf ",arr[i][j][k]);
      }
      printf("\n");
    }
    printf("\n\n");
  }
  printf("\n");
  return 0;
}
