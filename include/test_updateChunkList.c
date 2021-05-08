#include <stdio.h>
#include "updateChunkList.h"
#define N 20

int main(int arc, char* argv[]) {
  TEMPLATE(_spray_updlist,float) updf;
  TEMPLATE(_spray_updlist,double) updd;
  float contentf[N];
  double contentd[N];
  float contentf_ref[N];
  double contentd_ref[N];
  for(int i=0; i<N; i++) {
    contentf[i] = 0;
    contentd[i] = 0;
    contentf_ref[i] = 0;
    contentd_ref[i] = 0;
  }
  _spray_updlist_init_float(&updf);
  _spray_updlist_init_double(&updd);
  for(int i=0; i<2*N; i++) {
    _spray_updlist_append_float(&updf, i%N, (float)i);
    _spray_updlist_append_double(&updd, i%N, (double)i);
    contentf_ref[i%N] += (float)i;
    contentd_ref[i%N] += (double)i;
  }

  _spray_updlist_commit_float(&updf, &contentf[0]);
  _spray_updlist_commit_double(&updd, &contentd[0]);
  for(int i=0; i<N; i++) {
    printf("%f %f %lf %lf\n",contentf[i],contentf_ref[i],contentd[i],contentd_ref[i]);
  }
  return 0;
}

