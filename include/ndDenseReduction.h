#ifndef NDDENSEREDUCTION_H
#define NDDENSEREDUCTION_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#define __ALIGNMENT__ 64

typedef struct {
  int sizex, sizey, sizez, sizexyz;
  bool allocated;
  double *content;
} spray_nddense_double;

void spray_nddense_init(spray_nddense_double *__restrict__ init,
                      double *orig, int sizex, int sizey, int sizez) {
  init->sizex = sizex;
  init->sizey = sizey;
  init->sizez = sizez;
  init->sizexyz = sizex*sizey*sizez;
  init->content = orig;
  init->allocated = false;
}

void spray_nddense_increment(spray_nddense_double *obj, int ix, int iy, int iz, double val) {
  double (*contentptr)[obj->sizey][obj->sizez] = (double (*)[obj->sizey][obj->sizez])obj->content;
  contentptr[ix][iy][iz] += val;
}

void _spray_nddense_ompinit(spray_nddense_double *__restrict__ init,
                         spray_nddense_double *__restrict__ orig) {
  init->sizex = orig->sizex;
  init->sizey = orig->sizey;
  init->sizez = orig->sizez;
  init->sizexyz = orig->sizexyz;
  init->allocated = true;
  init->content = (double*)(aligned_alloc(__ALIGNMENT__, orig->sizexyz * sizeof(double)));
  for(int i=0;i<orig->sizexyz;i++) {
    init->content[i] = 0.0;
  }
}

void _spray_nddense_ompreduce(spray_nddense_double *__restrict__ out,
                              spray_nddense_double *__restrict__ in) {
  double* outc = out->content;
  double* inc = in->content;
  if(!out->allocated) {
#pragma omp simd aligned(inc : __ALIGNMENT__)
    for (int i = 0; i < out->sizexyz; i++)
      outc[i] += inc[i];
  }
  else {
#pragma omp simd aligned(outc, inc : __ALIGNMENT__)
    for (int i = 0; i < out->sizexyz; i++)
      outc[i] += inc[i];
  }
  free(inc);
}

#pragma omp declare reduction(+ : spray_nddense_double :      \
    _spray_nddense_ompreduce(&omp_out, &omp_in))              \
    initializer (_spray_nddense_ompinit(&omp_priv, &omp_orig))
#endif
