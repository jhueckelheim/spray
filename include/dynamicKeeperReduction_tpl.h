#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "templates.h"
#include "updateChunkList_tpl.h"

typedef struct {
  TEMPLATE(_spray_updlist,T) *updateLists;
  T *content;
  ownership_sequence *ownerseqs;
  int* owner_start;
  int* owners;
  int accesscounter;
  int accesschunkcounter;
  int mythreadnum;
  int numthreads;
} TEMPLATE(spray_keeper,T);

void TEMPLATE(spray_keeper_init,T)(TEMPLATE(spray_keeper,T)* init, T *orig,  ownership_sequence *ownerseqs) {
  init->content = orig;
  init->ownerseqs = ownerseqs;
  init->numthreads = omp_get_max_threads();
  init->updateLists = (TEMPLATE(_spray_updlist,T)*)malloc(init->numthreads*init->numthreads*sizeof(TEMPLATE(_spray_updlist,T)));
  for(int i=0; i<init->numthreads; i++) {
    for(int j=0; j<init->numthreads; j++) {
      init->updateLists[i*init->numthreads+j].lastchunk = NULL;
    }
  }
}

void TEMPLATE(_spray_keeper_ompinit,T)(TEMPLATE(spray_keeper,T) *__restrict__ init,
                            TEMPLATE(spray_keeper,T) *__restrict__ orig) {
  init->content = orig->content;
  init->owner_start = orig->owner_start;
  init->owners = orig->owners;
  init->numthreads = orig->numthreads;
  init->updateLists = orig->updateLists;
  init->accesscounter = 0;
  init->accesschunkcounter = 0;
  init->mythreadnum = omp_get_thread_num();
  init->ownerseqs = &(orig->ownerseqs[init->mythreadnum]);
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

void TEMPLATE(spray_keeper_increment,T)(TEMPLATE(spray_keeper,T) *obj, int idx, T val) {
  int tgt = obj->ownerseqs->owner[obj->accesschunkcounter];
  int src = obj->mythreadnum;
  if(tgt == src) {
    obj->content[idx] += val;
    //printf("in-place %d on tid %d\n",idx,src);
  }
  else {
    //printf("enqueue %d from tid %d to tid %d\n",idx,src,tgt);
    TEMPLATE(_spray_updlist,T) *list = &(obj->updateLists[tgt*obj->numthreads+src]);
    if(! list->lastchunk) {
      TEMPLATE(_spray_updlist_init,T)(list);
    }
    TEMPLATE(_spray_updlist_append,T)(list, idx, val);
  }
  obj->accesscounter++;
  if(obj->ownerseqs->owner_start[obj->accesschunkcounter+1] == obj->accesscounter) {
    obj->accesschunkcounter++;
  }
}

void TEMPLATE(_spray_keeper_ompreduce,T)(TEMPLATE(spray_keeper,T) *__restrict__ out,
                              TEMPLATE(spray_keeper,T) *__restrict__ in) {
}

#pragma omp declare reduction(+ : TEMPLATE(spray_keeper,T) :      \
    TEMPLATE(_spray_keeper_ompreduce,T)(&omp_out, &omp_in))              \
    initializer (TEMPLATE(_spray_keeper_ompinit,T)(&omp_priv, &omp_orig))
