#ifndef BLOCKREDUCTION_HPP
#define BLOCKREDUCTION_HPP
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

namespace spray {
template <typename contentType> class BlockReduction {
public:
  // This is the only constructor that is useful for end users:
  // It creates a "special" BlockReduction that actually just keeps a pointer to
  // the original data array, and behaves like a BlockReduction with just one
  // block (containing the original data). In addition, it keeps an array of
  // OpenMP locks that are used by the threads to reserve write-access into
  // blocks of the data array, but this functionality is only used when useLocks
  // is true.
  BlockReduction(int totalsize, contentType *orig, bool useLocks = false) {
    int nblocks = (totalsize - 1) / BSIZE + 1;
    this->blocks = (contentType **)malloc(sizeof(contentType *));
    this->memsize = sizeof(contentType *);
    if (this->blocks == NULL) {
      printf("Failed to alloc blocks.\n");
      exit(1);
    }
    if (useLocks) {
      this->writelocks = (omp_lock_t *)malloc(nblocks * sizeof(omp_lock_t));
      this->memsize += nblocks * sizeof(omp_lock_t);
      if (this->writelocks == NULL) {
        printf("Failed to alloc %d writelocks.\n", nblocks);
        exit(1);
      }
      for (int i = 0; i < nblocks; i++) {
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

  // This constructor must be used together with a call to init(), as it does
  // not perform any of the necessary initialization tasks. It is only here so
  // that OpenMP can create objects in the local scope on each thread that will
  // be automatically destroyed when they go out of scope during the reduction.
  BlockReduction() { this->initialized = false; }

  // Destructor, this is automatically called at the end of the parallel region
  // for all normal BlockReductions that live on threads, and for the "special"
  // BlockReduction, it is called at the end of the scope where this is created
  // by the user.
  ~BlockReduction() {
    assert(this->initialized);
    free(this->blocks);
    if (orig == NULL && this->useLocks) {
      for (int i = 0; i < nblocks; i++) {
        omp_destroy_lock(&(this->writelocks[i]));
      }
      free(this->writelocks);
    }
  }

  // Function to obtain the peak memory consumption of this BlockReduction. If
  // this is called after the reduction is completed, it will instead report the
  // overall memory consumption on all threads combined.
  long getMemSize() {
    assert(this->initialized);
    return this->memsize;
  }

  // This initialization sets up the `init` object and connects it to the
  // "special" BlockReduction `orig` that wraps the original data. The `init`
  // object allocates a list of pointers to blocks, but leaves the individual
  // blocks un-allocated.
  static void ompInit(BlockReduction<contentType> *init,
                      BlockReduction<contentType> *orig) {
    int totalsize = orig->totalsize;
    init->nblocks = (totalsize - 1) / BSIZE + 1;
    init->blocks =
        (contentType **)malloc(init->nblocks * sizeof(contentType *));
    init->memsize = init->nblocks * sizeof(contentType *);
    if (init->blocks == NULL) {
      printf("Failed to alloc blocks.\n");
      exit(1);
    }
    for (int i = 0; i < init->nblocks; i++) {
      init->blocks[i] = NULL;
    }
    init->orig = orig;
    init->totalsize = totalsize;
    init->initialized = true;
  }

  // Another very important function: Initiate the reductions. It iterates over
  // the blocks in the input BlockReduction, and merges them into the output
  // BlockReduction. Several things can happen:
  //  - The output may be the "special" BlockReduction that holds the original
  //  array.
  //    If this is the case, for each block in the input array, there are two
  //    possibilities:
  //     + the block is actually just a pointer into the original array, because
  //       the current thread managed to get a lock on it. No work needs to be
  //       done.
  //     + the block must be added to the output, perform this work.
  //  - The output may be just another BlockReduction. There are three cases:
  //     + A block exists only in the output. No work needed.
  //     + A block exists only in the input. Simply pass ownership of that block
  //       to the output.
  //     + A block exists in input and output. Actually do work here: Increment
  //       the output by the values in the input. A small wrinkle here: Either
  //       the input or the output block might actually be a block in the
  //       original array as a result of a thread having gotten the lock to it,
  //       in which case we must not attempt to free() this block, can not
  //       assume that the block is aligned, and we make sure that the output
  //       BlockReduction will obtain the ownership of this original block, even
  //       if it belonged to the input BlockReduction (which will fall out of
  //       scope after this and will no longer need the lock to that block
  //       anyways).
  static void ompReduce(BlockReduction<contentType> *out,
                        BlockReduction<contentType> *in) {
    assert(out->initialized && in->initialized);
    int i, j;
    if (out->nblocks == 1 && out->blocks[0] != NULL) {
      // An output with only one block: That is probably our "special" output
      // array (or the problem size is so small that we only have one block,
      // which would defeat the purpose of all this, but should also work).
      for (i = 0; i < in->nblocks; i++) {
        if (in->blocks[i] != NULL) {
          // This block exists in the input, we may have to do work.
          int startIdx = i * BSIZE;
          if (in->orig->useLocks &&
              in->blocks[i] == &(out->blocks[0][startIdx])) {
            // Actually, the block is just a reference into the original array,
            // so the data is already where it needs to be. Release the lock
            // now, we are done with this block.
            omp_unset_lock(&(out->writelocks[i]));
          } else {
            // The block is a privatized block on this thread. We need to do
            // work.
            addBlock(out->blocks[0] + startIdx, in->blocks[i], i,
                     out->totalsize, false);
            // At this point we are done with this block, free it.
            free(in->blocks[i]);
          }
        }
      }
    } else {
      assert(out->nblocks == in->nblocks);
      // We are merging two BlockReductions, neither of which is the "special"
      // one
      for (i = 0; i < in->nblocks; i++) {
        if (in->blocks[i] != NULL) {
          // We only need to do anything for blocks that are allocated in the
          // input.
          if (out->blocks[i] != NULL) {
            // The block exists in input and output. Add them together, then
            // free the input block as it is no longer needed.
            contentType *curOut = out->blocks[i];
            contentType *curIn = in->blocks[i];
            contentType *origBlock = &(in->orig->blocks[0][i * BSIZE]);
            if (in->orig->useLocks && curIn == origBlock) {
              // We are using the locking functionality, and the input block
              // lives in the original array. In this case, we swap ownership of
              // the blocks, since it is beneficial to keep the references and
              // locks to the original array around so that we can do more
              // useful work during the tree-shaped reduction phase, rather than
              // deferring all work to the final merge on the master thread.
              curOut = in->blocks[i];
              curIn = out->blocks[i];
              out->blocks[i] = in->blocks[i];
            }
            if (in->orig->useLocks && curOut == origBlock) {
              // We are using the locking functionality, and the output block
              // lives in the original array. The output block might be
              // unaligned.
              addBlock(curOut, curIn, i, out->totalsize, false);
            } else {
              // We are merging two privatized blocks, neither of them is part
              // of the original array.
              addBlock(curOut, curIn, i, out->totalsize, true);
            }
            free(curIn);
          } else {
            // We have a block that only exists in the input, not in the output.
            // Hand it over to the output.
            out->blocks[i] = in->blocks[i];
          }
        }
      }
    }
    out->memsize += in->memsize;
  }

  // Array index operator. If the array index lives in a block that does not
  // exist yet, we call createBlockIfNeeded() to create this block.
  contentType &operator[](int idx) {
    assert(this->initialized);
    int idxInBlock = idx % BSIZE;
    int block = idx / BSIZE;
    this->createBlockIfNeeded(block);
    return this->blocks[block][idxInBlock];
  }

private:
  contentType **blocks;
  omp_lock_t *writelocks;
  int nblocks;
  int totalsize;
  long memsize;
  bool initialized;
  bool useLocks;
  BlockReduction<contentType> *orig;
  // Perform the actual adding work. In the best case, this copies a whole block
  // and uses only aligned SIMD instructions. There is a special case for the
  // final block, which may be smaller than BSIZE, and for blocks that are
  // actually just pointers to the original array, wich may be unaligned.
  static void addBlock(contentType *curOut, contentType *curIn, int block,
                       int totalsize, bool outAligned) {
    int startIdx = block * BSIZE;
    int j;
    if (startIdx + BSIZE > totalsize) {
      // This is the last block and we only use the part of it that
      // actually corresponds to a part of the original array.
      if (outAligned) {
#pragma omp simd aligned(curIn, curOut : 64)
        for (j = 0; j < totalsize - startIdx; j++) {
          curOut[j] += curIn[j];
        }
      } else {
#pragma omp simd aligned(curIn : 64)
        for (j = 0; j < totalsize - startIdx; j++) {
          curOut[j] += curIn[j];
        }
      }
    } else {
      // This is a normal block, we copy all of it.
      if (outAligned) {
#pragma omp simd aligned(curIn, curOut : 64)
        for (j = 0; j < BSIZE; j++) {
          curOut[j] += curIn[j];
        }
      } else {
#pragma omp simd aligned(curIn : 64)
        for (j = 0; j < BSIZE; j++) {
          curOut[j] += curIn[j];
        }
      }
    }
  }

  // One of the most important functions: When an array index is accessed, this
  // function checks if the index lives inside a block that is already allocated
  // on the current thread. If yes, nothing is done. If no, there are two
  // possibilities: First, we attempt to get a lock for this block in the
  // orignal array held by the "special" BlockReduction. If successful, this
  // thread will directly write into the output array. If unsuccessful, we
  // allocate a privatized block of data now.
  void createBlockIfNeeded(int block) {
    assert(this->initialized);
    if (this->blocks[block] == NULL) {
      if (this->orig->useLocks &&
          omp_test_lock(&(this->orig->writelocks[block]))) {
        // we got the lock, set us up to write directly into the output array
        this->blocks[block] = &(this->orig->blocks[0][block * BSIZE]);
      } else {
        // we did not get the lock, allocate a private block here and initialize
        // it to zero.
        this->blocks[block] =
            (contentType *)aligned_alloc(64, BSIZE * sizeof(contentType));
        this->memsize += BSIZE * sizeof(contentType);
        contentType *curOut = this->blocks[block];
#pragma omp simd aligned(curOut : 64)
        for (int j = 0; j < BSIZE; j++) {
          curOut[j] = 0;
        }
      }
    }
  }
};

template class BlockReduction<double>;
template class BlockReduction<float>;
} // namespace spray

#pragma omp declare reduction(+ : spray::BlockReduction<double> : \
    spray::BlockReduction<double>::ompReduce(&omp_out, &omp_in))                         \
    initializer (spray::BlockReduction<double>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : spray::BlockReduction<float> : \
    spray::BlockReduction<float>::ompReduce(&omp_out, &omp_in))                         \
    initializer (spray::BlockReduction<float>::ompInit(&omp_priv, &omp_orig))
#endif
