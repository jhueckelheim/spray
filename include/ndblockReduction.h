#ifndef NDBLOCKREDUCTION_H
#define NDBLOCKREDUCTION_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define ALIGNMENT 64
#define BSIZE 4
#define BSIZEND BSIZE*BSIZE*BSIZE

typedef struct {
  int sizex, sizey, sizez, sizexyz;
  int nblkx, nblky, nblkz, nblkxyz;
  bool allocated;
  double *content;
  double **blockcontent;
  omp_lock_t *writelocks;
  double *signal_lock;
} spray_ndblock_double;

void spray_ndblock_init(spray_ndblock_double *__restrict__ init,
                        double *orig, int sizex, int sizey, int sizez) {
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
  init->signal_lock = orig;
  init->writelocks = (omp_lock_t*)malloc(init->nblkxyz * sizeof(omp_lock_t));
  if (init->writelocks == NULL) {
    printf("Failed to alloc %d writelocks.\n", init->nblkxyz);
    exit(1);
  }
  for (int i = 0; i < init->nblkxyz; i++) {
    omp_init_lock(&(init->writelocks[i]));
  }
}

void spray_ndblock_destroy(spray_ndblock_double *__restrict__ obj) {
  for (int i = 0; i < obj->nblkxyz; i++) {
    omp_destroy_lock(&(obj->writelocks[i]));
  }
  free(obj->writelocks);
}

void spray_ndblock_increment(spray_ndblock_double *obj, int ix, int iy, int iz, double val) {
  int blocki = ix / BSIZE;
  int i_in_blk = ix % BSIZE;
  int blockj = iy / BSIZE;
  int j_in_blk = iy % BSIZE;
  int blockk = iz / BSIZE;
  int k_in_blk = iz % BSIZE;
  double* (*blockgrid)[obj->nblky][obj->nblkz] = (double* (*)[obj->nblky][obj->nblkz])obj->blockcontent;
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
    omp_lock_t (*writelockgrid)[obj->nblky][obj->nblkz] = (omp_lock_t (*)[obj->nblky][obj->nblkz])obj->writelocks;
    if(omp_test_lock(&writelockgrid[blocki][blockj][blockk])) {
      blockgrid[blocki][blockj][blockk] = obj->signal_lock;
    }
    /*
    If we didn't get the lock (perhaps because another thread already owns it),
    we instead allocate a privatized block and initialize it to zero.
    */
    else {
      blockgrid[blocki][blockj][blockk] = (double*)aligned_alloc(ALIGNMENT, BSIZEND*sizeof(double));
      if (blockgrid[blocki][blockj][blockk] == NULL) {
        printf("Failed to alloc block %d %d %d.\n",blocki,blockj,blockk);
        exit(1);
      }
      for(int i=0; i<BSIZEND; i++) {
        blockgrid[blocki][blockj][blockk][i] = 0.0;
      }
    }
  }
  /*
  At this point, we have memory available to write (either from a previous call
  to this function, or because we just now initialized or acquired something).
  We can go ahead and commit our update, either by writing to the original
  array (if the pointer signals to us that we acquired the lock) or by writing
  to the privatized block.
  */
  if(blockgrid[blocki][blockj][blockk] == obj->signal_lock) {
    double (*outfield)[obj->sizey][obj->sizez] = (double (*)[obj->sizey][obj->sizez])obj->content;
    outfield[ix][iy][iz] += val;
  }
  else {
    double (*curblock)[BSIZE][BSIZE] = (double (*)[obj->sizey][obj->sizez])blockgrid[blocki][blockj][blockk];
    curblock[i_in_blk][j_in_blk][k_in_blk] += val;
  }
}

void _spray_ndblock_ompinit(spray_ndblock_double *__restrict__ init,
                         spray_ndblock_double *__restrict__ orig) {
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
  init->signal_lock = orig->signal_lock;
  init->blockcontent = (double**)(aligned_alloc(ALIGNMENT, orig->nblkxyz * sizeof(double*)));
  if (init->blockcontent == NULL) {
    printf("Failed to alloc blockcontent.\n");
    exit(1);
  }
  for(int i=0;i<init->nblkxyz;i++) {
    init->blockcontent[i] = NULL;
  }
  init->writelocks = orig->writelocks;
}

void _spray_ndblock_ompreduce(spray_ndblock_double *__restrict__ out,
                              spray_ndblock_double *__restrict__ in) {
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
    privatized block or signal an acquired lock, on either `in` or `out`.
    */
    double (*outfield)[out->sizey][out->sizez] = (double (*)[out->sizey][out->sizez])out->content;
    double* (*blockgrid_in)[in->nblky][in->nblkz] = (double* (*)[in->nblky][in->nblkz])in->blockcontent;
    double* (*blockgrid_out)[in->nblky][in->nblkz] = (double* (*)[in->nblky][in->nblkz])out->blockcontent;
    for(int blocki = 0; blocki<in->nblkx; blocki++) {
      for(int blockj = 0; blockj<in->nblky; blockj++) {
        for(int blockk = 0; blockk<in->nblkz; blockk++) {
          double* rawblk_in = blockgrid_in[blocki][blockj][blockk];
          double* rawblk_out = blockgrid_out[blocki][blockj][blockk];
          if(!rawblk_out) {
            /*
            If the `out` object does not have this block, we simply hand over the
            block from `in` to `out`. It does not matter if `in` has this block,
            because if it doesn't, this will simply copy over a NULL pointer.
            */
            blockgrid_out[blocki][blockj][blockk] = rawblk_in;
          }
          else if (rawblk_in) {
            if(rawblk_in != out->signal_lock && rawblk_out != out->signal_lock) {
              #pragma omp simd aligned(rawblk_out, rawblk_in : ALIGNMENT)
              for(int i=0;i<in->nblkxyz;i++) {
                rawblk_out[i] += rawblk_in[i];
              }
              free(rawblk_in);
            }
            else {
              if(rawblk_in == out->signal_lock) {
                blockgrid_in[blocki][blockj][blockk] = rawblk_out;
                blockgrid_out[blocki][blockj][blockk] = out->signal_lock;
              }
              double (*outfield)[out->sizey][out->sizez] = (double (*)[out->sizey][out->sizez])out->content;
              double (*block)[BSIZE][BSIZE] = (double (*)[BSIZE][BSIZE])rawblk_in;
              int blockstart_i = blocki*BSIZE;
              int blockstart_j = blockj*BSIZE;
              int blockstart_k = blockk*BSIZE;
              #pragma omp simd aligned(rawblk_in : ALIGNMENT) collapse(3)
              for(int i_in_blk=0; i_in_blk<BSIZE; i_in_blk++) {
                for(int j_in_blk=0; j_in_blk<BSIZE; j_in_blk++) {
                  for(int k_in_blk=0; k_in_blk<BSIZE; k_in_blk++) {
                    outfield[blockstart_i+i_in_blk][blockstart_j+j_in_blk][blockstart_k+k_in_blk] += block[i_in_blk][j_in_blk][k_in_blk];
                  }
                }
              }
              free(blockgrid_in[blocki][blockj][blockk]);
            }
          }
        }
      }
    }
  }
  else {
    double (*outfield)[out->sizey][out->sizez] = (double (*)[out->sizey][out->sizez])out->content;
    double* (*blockgrid_in)[in->nblky][in->nblkz] = (double* (*)[in->nblky][in->nblkz])in->blockcontent;
    for(int blocki = 0; blocki<in->nblkx; blocki++) {
      for(int blockj = 0; blockj<in->nblky; blockj++) {
        for(int blockk = 0; blockk<in->nblkz; blockk++) {
          double* rawblk_in = blockgrid_in[blocki][blockj][blockk];
          if(rawblk_in && rawblk_in != out->signal_lock) {
            double (*block)[BSIZE][BSIZE] = (double (*)[BSIZE][BSIZE])rawblk_in;
            int blockstart_i = blocki*BSIZE;
            int blockstart_j = blockj*BSIZE;
            int blockstart_k = blockk*BSIZE;
            #pragma omp simd aligned(rawblk_in : ALIGNMENT) collapse(3)
            for(int i_in_blk=0; i_in_blk<BSIZE; i_in_blk++) {
              for(int j_in_blk=0; j_in_blk<BSIZE; j_in_blk++) {
                for(int k_in_blk=0; k_in_blk<BSIZE; k_in_blk++) {
                  outfield[blockstart_i+i_in_blk][blockstart_j+j_in_blk][blockstart_k+k_in_blk] += block[i_in_blk][j_in_blk][k_in_blk];
                }
              }
            }
            free(rawblk_in);
          }
        }
      }
    }
  }
  free(in->blockcontent);
}

#pragma omp declare reduction(+ : spray_ndblock_double :      \
    _spray_ndblock_ompreduce(&omp_out, &omp_in))              \
    initializer (_spray_ndblock_ompinit(&omp_priv, &omp_orig))
#endif
