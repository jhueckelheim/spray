#ifndef DENSEREDUCTION_H
#define DENSEREDUCTION_H

#include <stdbool.h>
#include <stdlib.h>

#define __ALIGNMENT__ 64

typedef struct {
  int size;
  bool allocated;
  double *content;
} spray_dense_double;

void spray_dense_init(spray_dense_double *__restrict__ init,
                      double *orig, int size) {
  init->size = size;
  init->content = orig;
  init->allocated = false;
}

void spray_dense_increment(spray_dense_double *obj, int idx, double val) {
  obj->content[idx] += val;
}

void _spray_dense_ompinit(spray_dense_double *__restrict__ init,
                         spray_dense_double *__restrict__ orig) {
  init->size = orig->size;
  init->content = (double*)(aligned_alloc(__ALIGNMENT__, orig->size * sizeof(double)));
  for(int i=0; i<orig->size; i++) init->content[i] = 0;
  init->allocated = true;
}

void _spray_dense_ompreduce(spray_dense_double *__restrict__ out,
                           spray_dense_double *__restrict__ in) {
  double* outc = out->content;
  double* inc = in->content;
  if(!out->allocated) {
#pragma omp simd aligned(inc : __ALIGNMENT__)
    for (int i = 0; i < out->size; i++)
      outc[i] += inc[i];
  }
  else {
#pragma omp simd aligned(outc, inc : __ALIGNMENT__)
    for (int i = 0; i < out->size; i++)
      outc[i] += inc[i];
  }
  free(inc);
}

#pragma omp declare reduction(+ : spray_dense_double :      \
    _spray_dense_ompreduce(&omp_out, &omp_in))              \
    initializer (_spray_dense_ompinit(&omp_priv, &omp_orig))
#endif
