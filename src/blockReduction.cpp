#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <assert.h>
#include "blockReduction.hpp"

// This is the only constructor that is actually useful for end users: It
// creates a "special" BlockArray that actually just keeps a pointer to the
// original data array, and behaves like a BlockArray with just one block
// (containing the original data). In addition, it keeps an array of OpenMP
// locks that are used by the threads to reserve write-access into blocks of the
// data array, but this functionality is only used when useLocks is true.
template <typename contentType>
BlockArray<contentType>::BlockArray(int totalsize, contentType* orig, bool useLocks) {
  int nblocks = (totalsize-1)/BSIZE + 1;
  this->blocks = (contentType**) malloc(sizeof(contentType*));
  if(this->blocks == NULL) {
    printf("Failed to alloc blocks.\n");
    exit(1);
  }
  if(useLocks) {
    this->writelocks = (omp_lock_t*) malloc(nblocks*sizeof(omp_lock_t));
    if(this->writelocks == NULL) {
      printf("Failed to alloc %d writelocks.\n", nblocks);
      exit(1);
    }
    for(int i=0; i<nblocks; i++) {
      omp_init_lock(&(this->writelocks[i]));
    }
  }
  this->nblocks = 1;
  this->blocks[0] = orig;
  this->orig = NULL;
  this->totalsize = totalsize;
  this->initialized = true;
  this->useLocks = useLocks;
}

// This constructor must be used together with a call to init(), as it does not
// perform any of the necessary initialization tasks. It is only here so that
// OpenMP can create objects in the local scope on each thread that will be
// automatically destroyed when they go out of scope during the reduction.
template <typename contentType>
BlockArray<contentType>::BlockArray() {
  this->initialized = false;
}

// This initialization sets up the `init` object and connects it to the "special"
// BlockArray `orig` that wraps the original data. The `init` object allocates a
// list of pointers to blocks, but leaves the individual blocks un-allocated.
template <typename contentType> void
BlockArray<contentType>::ompInit(BlockArray<contentType> *init, BlockArray<contentType> *orig) {
  int totalsize = orig->totalsize;
  init->nblocks = (totalsize-1)/BSIZE + 1;
  init->blocks = (contentType**) malloc(init->nblocks*sizeof(contentType*));
  if(init->blocks == NULL) {
    printf("Failed to alloc blocks.\n");
    exit(1);
  }
  for(int i=0; i<init->nblocks; i++) {
    init->blocks[i] = NULL;
  }
  init->memsize = init->nblocks*sizeof(contentType*);
  init->orig = orig;
  init->totalsize = totalsize;
  init->initialized = true;
}

// Destructor, this is automatically called at the end of the parallel region
// for all normal BlockArrays that live on threads, and for the "special"
// BlockArray, it is called at the end of the scope where this is created by the
// user.
template <typename contentType>
BlockArray<contentType>::~BlockArray() {
  assert(this->initialized);
  free(this->blocks);
  if(orig == NULL && this->useLocks) {
    for(int i=0; i<nblocks; i++) {
      omp_destroy_lock(&(this->writelocks[i]));
    }
    free(this->writelocks);
  }
}

// One of the most important functions: When an array index is accessed, this
// function checks if the index lives inside a block that is already allocated
// on the current thread. If yes, nothing is done. If no, there are two
// possibilities: First, we attempt to get a lock for this block in the orignal
// array held by the "special" BlockArray. If successful, this thread will
// directly write into the output array. If unsuccessful, we allocate a
// privatized block of data now. A small quirk: we only take a lock and write
// into the original array if there is enough space for an entire block, to keep
// the functions handling block operations simple and avoid complicated bounds
// checks. If the problem size is not a multiple of the block size, the final
// slice of memory at the end is always handled as a privatized block.
template <typename contentType> void
BlockArray<contentType>::createBlockIfNeeded(int block) {
  assert(this->initialized);
  if(this->blocks[block] == NULL) {
    if(this->orig->useLocks && (block+1)*BSIZE<=this->totalsize &&
       omp_test_lock(&(this->orig->writelocks[block]))) {
      // we got the lock and this is a complete block, set us up to write
      // directly into the output array
      this->blocks[block] = &(this->orig->blocks[0][block*BSIZE]);
    }
    else {
      // we did not get the lock or the block is not complete, allocate a
      // private block here and initialize it to zero.
      this->blocks[block] = (contentType*) aligned_alloc(64,BSIZE * sizeof(contentType));
      this->memsize+=BSIZE*sizeof(contentType);
      contentType *curOut = this->blocks[block];
      #pragma omp simd aligned(curOut:64)
      for(int j=0; j<BSIZE; j++) {
        curOut[j] = 0;
      }
    }
  }
}

// Array index operator. If the array index lives in a block that does not exist
// yet, we call createBlockIfNeeded() to create this block.
template <typename contentType> contentType&
BlockArray<contentType>::operator[](int idx) {
  assert(this->initialized);
  int idxInBlock = idx % BSIZE;
  int block = idx / BSIZE;
  this->createBlockIfNeeded(block);
  return this->blocks[block][idxInBlock];
}

// Function to obtain the peak memory consumption of this BlockArray. If this is
// called after the reduction is completed, it will instead report the overall
// memory consumption on all threads combined.
template <typename contentType> long
BlockArray<contentType>::getMemSize() {
  assert(this->initialized);
  return this->memsize;
}

// Another very important function: The actual reduction work. It iterates over
// the blocks in the input BlockArray, and merges them into the output
// BlockArray. Several things can happen:
//  - The output may be the "special" BlockArray that holds the original array.
//    If this is the case, for each block in the input array, there are three
//    possibilities:
//     + the block is actually just a pointer into the original array, because
//       the current thread managed to get a lock on it. No work needs to be
//       done.
//     + the block must be added to the output, and there is enough room for
//       this block in the output array. Perform this work.
//     + the block must be added, but the original array does not have space
//       for the entire block. This may happen only for the last block, if the
//       problem size is not a multiple of the block size. In this case, copy
//       only the relevant part of the block.
//  - The output may be just another BlockArray. There are three cases:
//     + A block exists only in the output. No work needed.
//     + A block exists only in the input. Simply pass ownership of that block
//       to the output.
//     + A block exists in input and output. Actually do work here: Increment
//       the output by the values in the input.
template <typename contentType> void
BlockArray<contentType>::ompReduce(BlockArray<contentType> *out, BlockArray<contentType> *in) {
  assert(out->initialized && in->initialized);
  int i,j;
  if(out->nblocks == 1 && out->blocks[0] != NULL) {
    // An output with only one block: That is probably our "special" output
    // array (or the problem size is so small that we only have one block, which
    // would defeat the purpose of all this, but should also work).
    for(i=0; i<in->nblocks; i++) {
      if(in->blocks[i] != NULL) {
        // This block exists in the input, we may have to do work.
        int startIdx = i*BSIZE;
        if(in->orig->useLocks && in->blocks[i] == &(out->blocks[0][startIdx])) {
          // Actually, the block is just a reference into the original array, so
          // the data is already where it needs to be. Release the lock now, we
          // are done with this block.
          omp_unset_lock(&(out->writelocks[i]));
        }
        else {
          // The block is a privatized block on this thread. We need to do work.
          if(startIdx + BSIZE > out->totalsize) {
            // This is the last block and we only use the part of it that
            // actually corresponds to a part of the original array.
            for(j=0; j<out->totalsize - startIdx; j++) {
              out->blocks[0][startIdx+j] += in->blocks[i][j];
            }
          }
          else {
            // This is a normal block that needs to be taken completely, and
            // added to the original array.
            contentType *curOut = &(out->blocks[0][startIdx]);
            contentType *curIn = in->blocks[i];
            // We do not know if the original array was aligned, so curOut is
            // not in the aligned clause.
            #pragma omp simd aligned(curIn:64)
            for(j=0; j<BSIZE; j++) {
              curOut[j] += curIn[j];
            }
          }
          // At this point we are done with this block, free it.
          free(in->blocks[i]);
        }
      }
    }
  }
  else {
    assert(out->nblocks == in->nblocks);
    // We are merging two BlockArrays, neither of which is the "special" one
    // with the original array.
    for(i=0; i<in->nblocks; i++) {
      if(in->blocks[i] != NULL) {
        // We only need to do anything for blocks that are allocated in the
        // input.
        if(out->blocks[i] != NULL) {
          // The block exists in input and output, add the input to the output
          // and afterwards, free the block in the input as it is no longer
          // needed.
          contentType *curOut = out->blocks[i];
          contentType *curIn = in->blocks[i];
          if(in->orig->useLocks && in->blocks[i] == &(in->orig->blocks[0][i*BSIZE])) {
            // We are using the locking functionality, and the output block
            // lives in the original array and then might be unaligned. Also, no
            // need to free this block, as it is part of the original array.
            #pragma omp simd aligned(curIn:64)
            for(j=0; j<BSIZE; j++) {
              curOut[j] += curIn[j];
            }
          }
          else {
            #pragma omp simd aligned(curIn,curOut:64)
            for(j=0; j<BSIZE; j++) {
              curOut[j] += curIn[j];
            }
            free(in->blocks[i]);
          }
        }
        else {
          // We have a block that only exists in the input, not in the output.
          // Hand it over to the output.
          out->blocks[i] = in->blocks[i];
        }
      }
    }
  }
  out->memsize += in->memsize;
}
