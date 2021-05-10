#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "templates.h"
#include "updateChunkList_tpl.h"

extern int Hits, Misses;

typedef struct {
  T *content;
  Bitarray atomic_array;
} TEMPLATE(spray_keeper,T);

void TEMPLATE(spray_keeper_init,T)(TEMPLATE(spray_keeper,T)* init, T *orig,  Bitarray atomic_array) {
  init->content = orig;
  init->atomic_array = atomic_array;
}

void TEMPLATE(_spray_keeper_ompinit,T)(TEMPLATE(spray_keeper,T) *__restrict__ init,
                            TEMPLATE(spray_keeper,T) *__restrict__ orig) {
  *init = *orig;
}

void TEMPLATE(spray_keeper_finalize,T)(TEMPLATE(spray_keeper,T) *__restrict__ obj) {
}

void TEMPLATE(spray_keeper_increment,T)(TEMPLATE(spray_keeper,T) *obj, int x, int idx, T val) {
  __builtin_prefetch(&obj->content[idx], 1, 3);
  if (bitarray_get(obj->atomic_array, x)) {
    #pragma omp atomic update
    obj->content[idx] += val;
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
