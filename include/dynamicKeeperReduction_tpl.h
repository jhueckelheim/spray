#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "templates.h"
#include "updateChunkList_tpl.h"

extern int Hits, Misses;

#define likely(x) __builtin_expect(!!(x),1)
#define unlikely(x) __builtin_expect(!!(x),0)

typedef struct {
  T *content;
  Bitarray atomic_array;
  int idx;
  T val;
} TEMPLATE(spray_keeper,T);

void TEMPLATE(spray_keeper_init,T)(TEMPLATE(spray_keeper,T)* init, T *orig,  Bitarray atomic_array) {
  init->content = orig;
  init->atomic_array = atomic_array;
  init->idx = 0;
  init->val = 0;
}

void TEMPLATE(_spray_keeper_ompinit,T)(TEMPLATE(spray_keeper,T) *__restrict__ init,
                            TEMPLATE(spray_keeper,T) *__restrict__ orig) {
  *init = *orig;
}

void TEMPLATE(spray_keeper_finalize,T)(TEMPLATE(spray_keeper,T) *__restrict__ obj) {
  #pragma omp atomic update
  obj->content[obj->idx] += obj->val;
}

void TEMPLATE(spray_keeper_increment,T)(TEMPLATE(spray_keeper,T) *obj, int x, int idx, T val) {
  __builtin_prefetch(&obj->content[obj->idx], 1, 3);
  __builtin_prefetch(&obj->content[idx], 1, 3);
  if (bitarray_get(obj->atomic_array, x)) {
    if (likely(obj->val)) {
      #pragma omp atomic update
      obj->content[obj->idx] += obj->val;
    }
    obj->idx = idx;
    obj->val = val;
    //#pragma omp atomic
    //++Misses;
  } else {
    obj->content[idx] += val;
    //#pragma omp atomic
    //++Hits;
  }
}

void TEMPLATE(_spray_keeper_ompreduce,T)(TEMPLATE(spray_keeper,T) *__restrict__ out,
                              TEMPLATE(spray_keeper,T) *__restrict__ in) {
}

#pragma omp declare reduction(+ : TEMPLATE(spray_keeper,T) :      \
    TEMPLATE(_spray_keeper_ompreduce,T)(&omp_out, &omp_in))              \
    initializer (TEMPLATE(_spray_keeper_ompinit,T)(&omp_priv, &omp_orig))
