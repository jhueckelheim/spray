#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

long testReduction(double* out, int n);

int main(int argc, char** argv) {
  double out[NSIZE];
  double time;
  int i;
  long memsize;
  int checksum = 0;
  printf("running %s on %d threads\n",argv[0],omp_get_max_threads());

  for(i=0; i<NSIZE; i++) {
    out[i] = 1.0;
  }
  time = omp_get_wtime();
  memsize = testReduction(&(out[0]), NSIZE)/1024/1024;
  time = omp_get_wtime() - time;
  for(i=0; i<NSIZE; i++) {
    checksum = (checksum + ((int)out[i])*(i%15485863))%179424673;
  }
  printf("memsize %ld time %f checksum %d\n", memsize,time,checksum);

  return 0;
}
