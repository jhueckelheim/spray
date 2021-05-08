#include <stdio.h>
#include "dynamicKeeperReduction.h"
#define N 20

int main(int arc, char* argv[]) {
  TEMPLATE(spray_keeper,float) keeperf;
  TEMPLATE(spray_keeper,double) keeperd;
  ownership_sequence* ownerseqs = (ownership_sequence*)malloc(omp_get_max_threads()*sizeof(ownership_sequence));
  ownerseqs[0].owner_start = (int*)malloc(2*sizeof(int));
  ownerseqs[0].owner = (int*)malloc(1*sizeof(int));
  ownerseqs[0].owner_start[0] = 0;
  ownerseqs[0].owner_start[1] = 3 * N;
  ownerseqs[0].owner[0] = 0; 
  for(int i=1;i<omp_get_max_threads();i++) {
    ownerseqs[i].owner_start = (int*)malloc(3*sizeof(int));
    ownerseqs[i].owner = (int*)malloc(2*sizeof(int));
    ownerseqs[i].owner_start[0] = 0;
    ownerseqs[i].owner_start[1] = 1;
    ownerseqs[i].owner_start[2] = 3 * N;
    ownerseqs[i].owner[0] = i-1;
    ownerseqs[i].owner[1] = i; 
  }
  float contentf[N];
  double contentd[N];
  for(int i=0; i<N; i++) {
    contentf[i] = 0;
    contentd[i] = 0;
  }
  spray_keeper_init_float(&keeperf, contentf, ownerseqs);
  spray_keeper_init_double(&keeperd, contentd, ownerseqs);
  #pragma omp parallel for reduction(+:keeperf,keeperd)
  for(int i=1; i<N-1; i++) {
    spray_keeper_increment_float(&keeperf, i-1, 1.0);
    spray_keeper_increment_float(&keeperf, i,   2.0);
    spray_keeper_increment_float(&keeperf, i+1, 4.0);
    spray_keeper_increment_double(&keeperd, i-1, 1.0);
    spray_keeper_increment_double(&keeperd, i,   2.0);
    spray_keeper_increment_double(&keeperd, i+1, 4.0);
  }
  spray_keeper_finalize_float(&keeperf);
  spray_keeper_finalize_double(&keeperd);

  for(int i=0; i<N; i++) {
    printf("%d %f %lf\n",i,contentf[i],contentd[i]);
  }
  return 0;
}

