#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "templates.h"

#define ALIGNMENT 64
#define BSIZE 4
#define BSIZEND BSIZE*BSIZE*BSIZE

typedef struct {
  int sizex, sizey, sizez, sizexyz;
  int nblkx, nblky, nblkz, nblkxyz;
  bool allocated;
  T *content;
  T **blockcontent;
#ifdef USELOCKS
  omp_lock_t *writelocks;
  T *signal_lock;
#endif
} TEMPLATE(spray_ndblock,T);

void TEMPLATE(spray_ndblock_init,T)(TEMPLATE(spray_ndblock,T) *__restrict__ init,
                        T *orig, int sizex, int sizey, int sizez) {
  init->sizex = sizex;
  init->sizey = sizey;
  init->sizez = sizez;
  init->sizexyz = sizex*sizey*sizez;
  init->nblkx = (sizex - 1)/BSIZE + 1;
  init->nblky = (sizey - 1)/BSIZE + 1;
  init->nblkz = (sizez - 1)/BSIZE + 1;
  init->nblkxyz = init->nblkx * init->nblky * init->nblkz;
  init->content = orig;
  init->allocated = false;
#ifdef USELOCKS
  init->signal_lock = orig;
  init->writelocks = (omp_lock_t*)malloc(init->nblkxyz * sizeof(omp_lock_t));
  if (init->writelocks == NULL) {
    printf("Failed to alloc %d writelocks.\n", init->nblkxyz);
    exit(1);
  }
  for (int i = 0; i < init->nblkxyz; i++) {
    omp_init_lock(&(init->writelocks[i]));
  }
#endif
}

void TEMPLATE(spray_ndblock_destroy,T)(TEMPLATE(spray_ndblock,T) *__restrict__ obj) {
#ifdef USELOCKS
  for (int i = 0; i < obj->nblkxyz; i++) {
    omp_destroy_lock(&(obj->writelocks[i]));
  }
  free(obj->writelocks);
#endif
}

void TEMPLATE(spray_ndblock_increment,T)(TEMPLATE(spray_ndblock,T) *obj, int ix, int iy, int iz, T val) {
  int blocki = ix / BSIZE;
  int i_in_blk = ix % BSIZE;
  int blockj = iy / BSIZE;
  int j_in_blk = iy % BSIZE;
  int blockk = iz / BSIZE;
  int k_in_blk = iz % BSIZE;
  T* (*blockgrid)[obj->nblky][obj->nblkz] = (T* (*)[obj->nblky][obj->nblkz])obj->blockcontent;
  /*
  Check if we are about to write in a location that this thread already owns,
  which could be a privatized block, or a lock for that block in the original
  array. If we enter this branch, it means we have neither.
  */
  if(!blockgrid[blocki][blockj][blockk]) {
    /*
    First, we try to acquire a lock to the block in the original array. If this
    succeeds, we will set the block pointer to `signal_lock`.  In this case, the
    pointer will not be dereferenced later, but only serves as a signal that we
    may write into the original array.
    */
#ifdef USELOCKS
    omp_lock_t (*writelockgrid)[obj->nblky][obj->nblkz] = (omp_lock_t (*)[obj->nblky][obj->nblkz])obj->writelocks;
    if(omp_test_lock(&writelockgrid[blocki][blockj][blockk])) {
      blockgrid[blocki][blockj][blockk] = obj->signal_lock;
    }
    /*
    If we didn't get the lock (perhaps because another thread already owns it),
    we instead allocate a privatized block and initialize it to zero.
    */
    else {
#endif
      blockgrid[blocki][blockj][blockk] = (T*)aligned_alloc(ALIGNMENT, BSIZEND*sizeof(T));
      if (blockgrid[blocki][blockj][blockk] == NULL) {
        printf("Failed to alloc block %d %d %d.\n",blocki,blockj,blockk);
        exit(1);
      }
      for(int i=0; i<BSIZEND; i++) {
        blockgrid[blocki][blockj][blockk][i] = 0.0;
      }
#ifdef USELOCKS
    }
#endif
  }
  /*
  At this point, we have memory available to write (either from a previous call
  to this function, or because we just now initialized or acquired something).
  We can go ahead and commit our update, either by writing to the original
  array (if the pointer signals to us that we acquired the lock) or by writing
  to the privatized block.
  */
#ifdef USELOCKS
  if(blockgrid[blocki][blockj][blockk] == obj->signal_lock) {
    T (*outfield)[obj->sizey][obj->sizez] = (T (*)[obj->sizey][obj->sizez])obj->content;
    outfield[ix][iy][iz] += val;
  }
  else {
#endif
    T (*curblock)[BSIZE][BSIZE] = (T (*)[obj->sizey][obj->sizez])blockgrid[blocki][blockj][blockk];
    curblock[i_in_blk][j_in_blk][k_in_blk] += val;
#ifdef USELOCKS
  }
#endif
}

void TEMPLATE(_spray_ndblock_ompinit,T)(TEMPLATE(spray_ndblock,T) *__restrict__ init,
                            TEMPLATE(spray_ndblock,T) *__restrict__ orig) {
  init->sizex = orig->sizex;
  init->sizey = orig->sizey;
  init->sizez = orig->sizez;
  init->sizexyz = orig->sizexyz;
  init->nblkx = orig->nblkx;
  init->nblky = orig->nblky;
  init->nblkz = orig->nblkz;
  init->nblkxyz = orig->nblkxyz;
  init->allocated = true;
  init->content = orig->content;
#ifdef USELOCKS
  init->signal_lock = orig->signal_lock;
  init->writelocks = orig->writelocks;
#endif
  init->blockcontent = (T**)(aligned_alloc(ALIGNMENT, orig->nblkxyz * sizeof(T*)));
  if (init->blockcontent == NULL) {
    printf("Failed to alloc blockcontent.\n");
    exit(1);
  }
  for(int i=0;i<init->nblkxyz;i++) {
    init->blockcontent[i] = NULL;
  }
}

void TEMPLATE(_merge_block_into_orig,T)(int blocki, int blockj, int blockk, T* rawblock, TEMPLATE(spray_ndblock,T)* out) {
  /*
  This function takes a privatized block and adds it to the original output
  array. This looks a little complicated simply because the shapes do not
  match: The block is a contiguous chunk of BSIZE*BSIZE*BSIZE, which maps to
  BSIZE*BSIZE contiguous stripes of length BSIZE in the output array (each
  stripe is contiguous, but they are not near each other in memory).
  */
  T (*outfield)[out->sizey][out->sizez] = (T (*)[out->sizey][out->sizez])out->content;
  T (*block)[BSIZE][BSIZE] = (T (*)[BSIZE][BSIZE])rawblock;
  int blockstart_i = blocki*BSIZE;
  int blockstart_j = blockj*BSIZE;
  int blockstart_k = blockk*BSIZE;
  #pragma omp simd aligned(block : ALIGNMENT) collapse(3)
  for(int i_in_blk=0; i_in_blk<BSIZE; i_in_blk++) {
    for(int j_in_blk=0; j_in_blk<BSIZE; j_in_blk++) {
      for(int k_in_blk=0; k_in_blk<BSIZE; k_in_blk++) {
        outfield[blockstart_i+i_in_blk][blockstart_j+j_in_blk][blockstart_k+k_in_blk] += block[i_in_blk][j_in_blk][k_in_blk];
      }
    }
  }
  free(rawblock);
}

void TEMPLATE(_spray_ndblock_ompreduce,T)(TEMPLATE(spray_ndblock,T) *__restrict__ out,
                              TEMPLATE(spray_ndblock,T) *__restrict__ in) {
  /*
  We distinguish two basic cases: If `out` is just another reducer object that
  was created on a thread, we use the first branch. Else, `out` is the reducer
  object that holds a reference to the original array and does not hold any
  privatized blocks.
  */
  if(out->allocated) {
    /*
    We merge two reducer objects that did work on a thread and may hold any
    number of privatized blocks or locks. We iterate over the entire grid of
    block pointers, and do work depending on whether they point to a
    privatized block or signal an acquired lock.
    */
    T (*outfield)[out->sizey][out->sizez] = (T (*)[out->sizey][out->sizez])out->content;
    T* (*blockgrid_in)[in->nblky][in->nblkz] = (T* (*)[in->nblky][in->nblkz])in->blockcontent;
    T* (*blockgrid_out)[in->nblky][in->nblkz] = (T* (*)[in->nblky][in->nblkz])out->blockcontent;
    for(int blocki = 0; blocki<in->nblkx; blocki++) {
      for(int blockj = 0; blockj<in->nblky; blockj++) {
        for(int blockk = 0; blockk<in->nblkz; blockk++) {
          T* rawblk_in = blockgrid_in[blocki][blockj][blockk];
          T* rawblk_out = blockgrid_out[blocki][blockj][blockk];
          if(!rawblk_out) {
            /*
            If the `out` object does not have this block, we simply hand over the
            block from `in` to `out`. It does not matter if `in` has this block,
            because if it doesn't, this will simply copy over a NULL pointer.
            */
            blockgrid_out[blocki][blockj][blockk] = rawblk_in;
          }
          else if (rawblk_in) {
            /*
            If we get here, it means the `in` and `out` object both have data
            for this block. We now must distinguish between privatized blocks
            or locks into the original data array.
            */
#ifdef USELOCKS
            if(rawblk_in != out->signal_lock && rawblk_out != out->signal_lock) {
#endif
              /*
              In this case, both `in` and `out` hold a privatized block. We
              add the incoming block to the outgoing block and free the former.
              */
              #pragma omp simd aligned(rawblk_out, rawblk_in : ALIGNMENT)
              for(int i=0;i<BSIZEND;i++) {
                rawblk_out[i] += rawblk_in[i];
              }
              free(rawblk_in);
#ifdef USELOCKS
            }
            else {
              /*
              In this case, exactly one of the objects has a lock into the
              original array (we explicitly tested that at least one has it,
              but they can't both have it, since it is a lock!). The other
              object must hold a privatized block.
              */
              if(rawblk_in == out->signal_lock) {
                /*
                We want the outgoing object to have the lock. If the incoming
                object had it, we swap the data pointers of `in` and `out` so
                that `out` will hold the lock and `in` will hold the privatized
                block.
                */
                blockgrid_in[blocki][blockj][blockk] = rawblk_out;
                blockgrid_out[blocki][blockj][blockk] = out->signal_lock;
              }
              /*
              When we get here, we know that `out` holds a lock to this block
              in the original output array, and `in` has a pointer to a private
              block. We add the values from the private block into the original
              array, then free the private block.
              */
              TEMPLATE(_merge_block_into_orig,T)(blocki,blockj,blockk,rawblk_in,out);
            }
#endif
          }
        }
      }
    }
  }
  else {
    /*
    This is the case where `out` is just a thin shell around the original
    output array. We go through all privatized blocks in `in` and merge them
    into the output array.
    */
    T* (*blockgrid_in)[in->nblky][in->nblkz] = (T* (*)[in->nblky][in->nblkz])in->blockcontent;
    for(int blocki = 0; blocki<in->nblkx; blocki++) {
      for(int blockj = 0; blockj<in->nblky; blockj++) {
        for(int blockk = 0; blockk<in->nblkz; blockk++) {
          T* rawblk_in = blockgrid_in[blocki][blockj][blockk];
          if(rawblk_in
#ifdef USELOCKS
             && rawblk_in != out->signal_lock
#endif
            ) {
              TEMPLATE(_merge_block_into_orig,T)(blocki,blockj,blockk,rawblk_in,out);
          }
        }
      }
    }
  }
  /*
  While adding data from `in` into `out`, we have also freed all privatized
  blocks that were owned by `in`. Now is a good time to free() the array of
  pointers to `in`'s privatized blocks.
  */
  free(in->blockcontent);
}

#pragma omp declare reduction(+ : TEMPLATE(spray_ndblock,T) :      \
    TEMPLATE(_spray_ndblock_ompreduce,T)(&omp_out, &omp_in))              \
    initializer (TEMPLATE(_spray_ndblock_ompinit,T)(&omp_priv, &omp_orig))
