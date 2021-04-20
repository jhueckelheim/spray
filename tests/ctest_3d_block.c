#include "ndblockReduction.h"
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NI 64
#define NJ 64
#define NK 64

int main(int argc, char** argv) {
  double arr[NI][NJ][NK];
  double arr_ref[NI][NJ][NK];
  for(int i=0;i<NI;i++) {
    for(int j=0;j<NJ;j++) {
      for(int k=0;k<NK;k++) {
        arr[i][j][k] = 0;
        arr_ref[i][j][k] = 0;
      }
    }
  }

  spray_ndblock_double sp_arr;
  spray_ndblock_init(&sp_arr, &(arr[0][0][0]), NI, NJ, NK);
  #pragma omp parallel for reduction(+:sp_arr)
  for(int i=1;i<NI-1;i++) {
    for(int j=1;j<NJ-1;j++) {
      for(int k=1;k<NK-1;k++) {
        spray_ndblock_increment(&sp_arr, i,   j,   k,     1.0);
        spray_ndblock_increment(&sp_arr, i-1, j,   k,     2.0);
        spray_ndblock_increment(&sp_arr, i+1, j,   k,     4.0);
        spray_ndblock_increment(&sp_arr, i,   j-1, k,     8.0);
        spray_ndblock_increment(&sp_arr, i,   j+1, k,    16.0);
        spray_ndblock_increment(&sp_arr, i,   j,   k-1,  32.0);
        spray_ndblock_increment(&sp_arr, i,   j,   k+1,  64.0);
      }
    }
  }

  for(int i=1;i<NI-1;i++) {
    for(int j=1;j<NJ-1;j++) {
      for(int k=1;k<NK-1;k++) {
        double tid = 100000.0*(omp_get_thread_num()+1);
        arr_ref[i  ][j  ][k  ]+=  1.0;
        arr_ref[i-1][j  ][k  ]+=  2.0;
        arr_ref[i+1][j  ][k  ]+=  4.0;
        arr_ref[i  ][j-1][k  ]+=  8.0;
        arr_ref[i  ][j+1][k  ]+= 16.0;
        arr_ref[i  ][j  ][k-1]+= 32.0;
        arr_ref[i  ][j  ][k+1]+= 64.0;
      }
    }
  }
  double err = 0.0;
  for(int i=0;i<NI;i++) {
    for(int j=0;j<NJ;j++) {
      for(int k=0;k<NK;k++) {
        err += fabs(arr[i][j][k]-arr_ref[i][j][k]);
      }
    }
  }
  printf("Error: %lf\n",err);
  return 0;
}
