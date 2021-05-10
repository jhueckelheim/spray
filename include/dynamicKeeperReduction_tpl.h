#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "templates.h"
#include "updateChunkList_tpl.h"

extern int Hits, Misses;

typedef struct {
  TEMPLATE(_spray_updlist,T) *updateLists;
  T *content;
  int* owner_array;
  int numthreads;
  int mythreadnum;
} TEMPLATE(spray_keeper,T);

void TEMPLATE(spray_keeper_init,T)(TEMPLATE(spray_keeper,T)* init, T *orig,  int *owner_array) {
  init->updateLists = (TEMPLATE(_spray_updlist,T)*)calloc(init->numthreads*init->numthreads,sizeof(TEMPLATE(_spray_updlist,T)));
  init->content = orig;
  init->owner_array = owner_array;
  init->numthreads = omp_get_max_threads();
  init->mythreadnum = omp_get_thread_num();
}

void TEMPLATE(_spray_keeper_ompinit,T)(TEMPLATE(spray_keeper,T) *__restrict__ init,
                            TEMPLATE(spray_keeper,T) *__restrict__ orig) {
  *init = *orig;
  init->mythreadnum = omp_get_thread_num();
}

void TEMPLATE(spray_keeper_finalize,T)(TEMPLATE(spray_keeper,T) *__restrict__ obj) {
  #pragma omp parallel
  {
    int tgt = omp_get_thread_num();
    for(int src=0; src<obj->numthreads; src++) {
      if(tgt == src) continue;
      TEMPLATE(_spray_updlist,T) *list = &(obj->updateLists[tgt*obj->numthreads+src]);
      if(list->lastchunk) {
        TEMPLATE(_spray_updlist_commit,T)(list,obj->content);
      }
    }
  }
  free(obj->updateLists);
}

void TEMPLATE(spray_keeper_increment,T)(TEMPLATE(spray_keeper,T) *obj, int x, int idx, T val) {
  int tgt = obj->owner_array[x];
  int src = obj->mythreadnum;
  if(tgt == src) {
    //#pragma omp atomic
    //++Hits;
    obj->content[idx] += val;
    //printf("in-place %d on tid %d\n",idx,src);
  }
  else {
    //#pragma omp atomic
    //++Misses;
    //printf("enqueue %d from tid %d to tid %d\n",idx,src,tgt);
    TEMPLATE(_spray_updlist,T) *list = &(obj->updateLists[tgt*obj->numthreads+src]);
    if(! list->lastchunk) {
      TEMPLATE(_spray_updlist_init,T)(list);
    }
    TEMPLATE(_spray_updlist_append,T)(list, idx, val);
  }
}

void TEMPLATE(_spray_keeper_ompreduce,T)(TEMPLATE(spray_keeper,T) *__restrict__ out,
                              TEMPLATE(spray_keeper,T) *__restrict__ in) {
}

#pragma omp declare reduction(+ : TEMPLATE(spray_keeper,T) :      \
    TEMPLATE(_spray_keeper_ompreduce,T)(&omp_out, &omp_in))              \
    initializer (TEMPLATE(_spray_keeper_ompinit,T)(&omp_priv, &omp_orig))
