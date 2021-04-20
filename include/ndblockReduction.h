#ifndef NDDENSEREDUCTION_H
#define NDDENSEREDUCTION_H

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
}

void spray_ndblock_increment(spray_ndblock_double *obj, int ix, int iy, int iz, double val) {
  int blocki = ix / BSIZE;
  int i_in_blk = ix % BSIZE;
  int blockj = iy / BSIZE;
  int j_in_blk = iy % BSIZE;
  int blockk = iz / BSIZE;
  int k_in_blk = iz % BSIZE;
  double* (*blockgrid)[obj->nblky][obj->nblkz] = (double* (*)[obj->nblky][obj->nblkz])obj->blockcontent;
  double* rawblk = blockgrid[blocki][blockj][blockk];
  if(!rawblk) {
    //printf("allocating new block %d %d %d on %p\n",blocki,blockj,blockk,obj->blockcontent);
    blockgrid[blocki][blockj][blockk] = (double*)aligned_alloc(ALIGNMENT, BSIZEND*sizeof(double));
    rawblk = blockgrid[blocki][blockj][blockk];
    for(int i=0; i<BSIZEND; i++) rawblk[i] = 0.0;
  }
  double (*curblock)[BSIZE][BSIZE] = (double (*)[obj->sizey][obj->sizez])rawblk;
  curblock[i_in_blk][j_in_blk][k_in_blk] += val;
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
  init->blockcontent = (double**)(aligned_alloc(ALIGNMENT, orig->nblkxyz * sizeof(double*)));
  for(int i=0;i<init->nblkxyz;i++) {
    init->blockcontent[i] = NULL;
  }
  //printf("init block %p on tid %d\n",init->blockcontent, omp_get_thread_num());
}

void _spray_ndblock_ompreduce(spray_ndblock_double *__restrict__ out,
                              spray_ndblock_double *__restrict__ in) {
  //printf("combine block %p and %p\n",in->blockcontent, out->blockcontent);
  if(out->allocated) {
    for(int block = 0; block<in->nblkxyz; block++) {
      double* rawblk_in = in->blockcontent[block];
      double* rawblk_out = out->blockcontent[block];
      if(!rawblk_out) {
        out->blockcontent[block] = rawblk_in;
        //printf("  %d hand over\n",block);
      }
      else if(rawblk_in) {
        #pragma omp simd aligned(rawblk_out, rawblk_in : ALIGNMENT)
        for(int i=0;i<in->nblkxyz;i++) {
          rawblk_out[i] += rawblk_in[i];
        }
        free(rawblk_in);
        //printf("  %d copy over\n",block);
      }
    }
  }
  else {
    //printf("  reshape\n");
    double (*outfield)[out->sizey][out->sizez] = (double (*)[out->sizey][out->sizez])out->content;
    double* (*blockgrid_in)[in->nblky][in->nblkz] = (double* (*)[in->nblky][in->nblkz])in->blockcontent;
    for(int blocki = 0; blocki<in->nblkx; blocki++) {
      for(int blockj = 0; blockj<in->nblky; blockj++) {
        for(int blockk = 0; blockk<in->nblkz; blockk++) {
          double* rawblk_in = blockgrid_in[blocki][blockj][blockk];
          //printf("    reshaping blk %d %d %d\n",blocki,blockj,blockk);
          if(rawblk_in) {
            //printf("    actually\n");
            double (*block)[BSIZE][BSIZE] = (double (*)[in->sizey][in->sizez])rawblk_in;
            for(int i_in_blk=0; i_in_blk<BSIZE; i_in_blk++) {
              int blockstart_i = blocki*BSIZE;
              for(int j_in_blk=0; j_in_blk<BSIZE; j_in_blk++) {
                int blockstart_j = blockj*BSIZE;
                #pragma omp simd aligned(rawblk_in : ALIGNMENT)
                for(int k_in_blk=0; k_in_blk<BSIZE; k_in_blk++) {
                  int blockstart_k = blockk*BSIZE;
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
}

#pragma omp declare reduction(+ : spray_ndblock_double :      \
    _spray_ndblock_ompreduce(&omp_out, &omp_in))              \
    initializer (_spray_ndblock_ompinit(&omp_priv, &omp_orig))
#endif
