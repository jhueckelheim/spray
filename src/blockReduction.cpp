#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "blockReduction.hpp"

template <typename contentType>
BlockArray<contentType>::BlockArray() {
  //printf("called constructor() on %p t %d\n",this,omp_get_thread_num());
  // TODO: dangerous uninitialized constructor
}

template <typename contentType>
BlockArray<contentType>::BlockArray(int totalsize) {
  //printf("called constructor(%d, %d) on %p t %d\n",blocksize, totalsize, this,omp_get_thread_num());
  this->init(totalsize);
}

template <typename contentType> void
BlockArray<contentType>::init(int totalsize) {
  this->nblocks = (totalsize-1)/BSIZE + 1;
  //printf("called init(%d, %d) on %p creating %d blocks t %d\n",blocksize, totalsize, this, this->nblocks,omp_get_thread_num());
  this->blocks = (contentType**) malloc(this->nblocks*sizeof(contentType*));
  for(int i=0; i<this->nblocks; i++) {
    this->blocks[i] = NULL;
  }
  this->memsize = this->nblocks*sizeof(contentType*);
  this->isOrig = 0;
  this->totalsize = totalsize;
}

template <typename contentType>
BlockArray<contentType>::BlockArray(int totalsize, contentType* orig) {
  int nblocks = (totalsize-1)/BSIZE + 1;
  //printf("called constructor(%d, %d, %p) on %p t %d\n",blocksize, totalsize, this, orig, omp_get_thread_num());
  this->blocks = (contentType**) malloc(sizeof(contentType*));
  this->nblocks = 1;
  this->blocks[0] = orig;
  this->isOrig = 1;
  this->totalsize = totalsize;
}

template <typename contentType>
BlockArray<contentType>::~BlockArray() {
  //printf("called destructor on %p t %d\n",this, omp_get_thread_num());
  free(this->blocks);
}

template <typename contentType> void
BlockArray<contentType>::createIfNeeded(int block) {
  if(this->blocks[block] == NULL) {
    this->blocks[block] = (contentType*) aligned_alloc(64,BSIZE * sizeof(contentType));
    //printf("alloc block %p while incrementing at %d, which is pos %d in block %d t %d\n",this->blocks[block],idx,idxInBlock,block,omp_get_thread_num());
    this->memsize+=BSIZE*sizeof(contentType);
    contentType *curOut = this->blocks[block];
    #pragma omp simd aligned(curOut:64)
    for(int j=0; j<BSIZE; j++) {
      curOut[j] = 0;
    }
  }
}

template <typename contentType> contentType&
BlockArray<contentType>::operator[](int idx) {
  int idxInBlock = idx % BSIZE;
  int block = idx / BSIZE;
  this->createIfNeeded(block);
  return this->blocks[block][idxInBlock];
}

template <typename contentType> void
BlockArray<contentType>::increment(int idx, contentType val) {
  int idxInBlock = idx % BSIZE;
  int block = idx / BSIZE;
  this->createIfNeeded(block);
  this->blocks[block][idxInBlock] += val;
}

template <typename contentType> long
BlockArray<contentType>::getMemSize() {
  return this->memsize;
}

template <typename contentType> void
BlockArray<contentType>::ompInit(BlockArray<contentType> *init, BlockArray<contentType> *orig) {
  init->init(orig->totalsize);
}

template <typename contentType> void
BlockArray<contentType>::ompReduce(BlockArray<contentType> *out, BlockArray<contentType> *in) {
  //printf("called reduce on %p %p t %d\n",out,in,omp_get_thread_num());
  int i,j;
  if(out->nblocks == 1 && out->blocks[0] != NULL) {
    //printf("called addToArray(%p) on %p t %d\n",out, in, omp_get_thread_num());
    for(i=0; i<in->nblocks; i++) {
      if(in->blocks[i] != NULL) {
        int startIdx = i*BSIZE;
        if(startIdx + BSIZE > out->totalsize) {
          for(j=0; j<out->totalsize - startIdx; j++) {
            out->blocks[0][startIdx+j] += in->blocks[i][j];
          }
        }
        else {
          contentType *curOut = &(out->blocks[0][startIdx]);
          contentType *curIn = in->blocks[i];
          #pragma omp simd aligned(curOut,curIn:64)
          for(j=0; j<BSIZE; j++) {
            curOut[j] += curIn[j];
          }
        }
        free(in->blocks[i]);
      }
    }
  }
  else {
    for(i=0; i<in->nblocks; i++) {
      if(in->blocks[i] != NULL) {
        if(out->blocks[i] != NULL) {
          for(j=0; j<BSIZE; j++) {
            out->blocks[i][j] += in->blocks[i][j];
          }
          free(in->blocks[i]);
        }
        else {
          out->blocks[i] = in->blocks[i];
        }
      }
    }
  }
  out->memsize += in->memsize;
}
