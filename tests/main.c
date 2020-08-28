#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

void testReduction(double* out, int n);

int main(int argc, char** argv) {
  double out[NSIZE];
  double time;
  int i;
  int checksum = 0;
  printf("running %s on %d threads\n",argv[0],omp_get_max_threads());

  for(i=0; i<NSIZE; i++) {
    out[i] = 1.0;
  }
  time = omp_get_wtime();
  testReduction(&(out[0]), NSIZE);
  time = omp_get_wtime() - time;
  for(i=0; i<NSIZE; i++) {
    checksum = (checksum + ((int)out[i])*(i%15485863))%179424673;
  }
  printf("time %f checksum %d\n", time,checksum);

  return 0;
}
