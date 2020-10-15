#ifndef AWBLOCKREDUCTION_HPP
#define AWBLOCKREDUCTION_HPP

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define CACHE_LINE_SIZE 64

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cstdio>
#include <new>
#include <omp.h>

namespace spray {
template <typename contentType, unsigned BlockSize>
struct alignas(CACHE_LINE_SIZE) AlignedBlockReduction {
  static constexpr std::size_t Alignment{CACHE_LINE_SIZE};

  struct alignas(CACHE_LINE_SIZE) signalType {
    bool set() {
      int zero = 0;
      return flag.compare_exchange_strong(zero, 1, std::memory_order_release,
                                          std::memory_order_relaxed);
    }

  private:
    std::atomic<int> flag{0};
  };

  struct blockType {
    blockType &operator=(contentType *p) {
      data = p;
      return *this;
    }
    operator contentType *&() { return data; }
    contentType *data = nullptr;
  };
  //using blockType = contentType *;

  // This is the only constructor that is useful for end users:
  // It creates a "special" AlignedBlockReduction that actually just keeps a
  // pointer to the original data array, and behaves like a
  // AlignedBlockReduction with just one block (containing the original data).
  // In addition, it keeps an array of
  AlignedBlockReduction(int totalsize, contentType *data)
      : data(data), orig(this) {
    assert((std::intptr_t(data) % Alignment) == 0 &&
           "AlignedBlockReduction cannot be used with underaligned pointers!");
    assert((totalsize % BlockSize) == 0 && "");

    nblocks = (totalsize - 1) / BlockSize + 1;
    //signals = new (AlignmentVal) signalType[nblocks];
    signals = (signalType*) aligned_alloc(Alignment, nblocks * sizeof(signalType));
    //blocks = new (AlignmentVal) blockType[nblocks];
    //std::memset(blocks, 0, sizeof(blockType) * nblocks);
  }

  // This constructor must be used together with a call to init(), as it does
  // not perform any of the necessary initialization tasks. It is only here so
  // that OpenMP can create objects in the local scope on each thread that will
  // be automatically destroyed when they go out of scope during the reduction.
  AlignedBlockReduction() {
  }

  // Destructor, this is automatically called at the end of the parallel region
  // for all normal BlockReductions that live on threads, and for the "special"
  // AlignedBlockReduction, it is called at the end of the scope where this is
  // created by the user.
  ~AlignedBlockReduction() {
    //delete[] signals;
    free(signals);
  }

  static void ompInit(AlignedBlockReduction *init,
                      AlignedBlockReduction *orig) {
    init->orig = orig->orig;
    init->data = orig->data;
    init->nblocks = orig->nblocks;
    init->signals = nullptr;
    //init->blocks = new (AlignmentVal) blockType[init->nblocks];
    init->blocks = (blockType*) aligned_alloc(Alignment, init->nblocks * sizeof(blockType));
    std::memset(init->blocks, 0, sizeof(blockType) * init->nblocks);
  }

  static void ompReduce(AlignedBlockReduction *__restrict__ out,
                        AlignedBlockReduction *__restrict__ in) {
    assert(out->nblocks == in->nblocks);
    for (int i = 0; i < in->nblocks; i++) {
      contentType *dataOut = &out->data[BlockSize * i];
      contentType *&curOut = out->blocks ? out->blocks[i] : dataOut;
      contentType *&curIn = in->blocks[i];

      if (!curIn)
        continue;
      if (curIn == &in->data[BlockSize * i])
        continue;

      if (!curOut) {
        std::swap(curIn, curOut);
        continue;
      }

#pragma omp simd aligned(curIn, curOut : Alignment)
      for (int j = 0; j < BlockSize; j++)
        curOut[j] += curIn[j];

      //delete[] curIn;
      free(curIn);
    }

    //delete[] in->blocks;
    free(in->blocks);
  }

  // Array index operator. We call createBlockIfNeeded() to create this block
  // if the array index lives in a block that does not yet exist.
  contentType &operator[](int idx) {
    int block = idx / BlockSize;
    int idxInBlock = idx % BlockSize;
    this->createBlockIfNeeded(block);
    return this->blocks[block][idxInBlock];
  }

private:
  AlignedBlockReduction *orig = nullptr;
  contentType *data = nullptr;
  int64_t nblocks = 0;
  blockType *blocks = nullptr;
  signalType *signals = nullptr;

  // One of the most important functions: When an array index is accessed, this
  // function checks if the index lives inside a block that is already allocated
  // on the current thread. If yes, nothing is done. If no, there are two
  // possibilities: First, we attempt to get a lock for this block in the
  // orignal array held by the "special" AlignedBlockReduction. If successful,
  // this thread will directly write into the output array. If unsuccessful, we
  // allocate a privatized block of data now.
  void createBlockIfNeeded(int blockIdx) {
    if (this->blocks[blockIdx])
      return;

    if (this->orig->signals[blockIdx].set()) {
      this->blocks[blockIdx] = &this->data[blockIdx * BlockSize];
    } else {
      //this->blocks[blockIdx] = new (AlignmentVal) contentType[BlockSize];
      this->blocks[blockIdx] = (contentType*) aligned_alloc(Alignment, BlockSize * sizeof(contentType));
      std::memset(this->blocks[blockIdx], 0, sizeof(contentType) * BlockSize);
    }
  }
};

template <typename contentType>
using BlockReduction128 = AlignedBlockReduction<contentType, 128>;

template <typename contentType>
using BlockReduction256 = AlignedBlockReduction<contentType, 256>;

template <typename contentType>
using BlockReduction4096 = AlignedBlockReduction<contentType, 4096>;
} // namespace spray

#pragma omp declare reduction(+ : spray::BlockReduction128<double> : \
    spray::BlockReduction128<double>::ompReduce(&omp_out, &omp_in))                         \
    initializer (spray::BlockReduction128<double>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : spray::BlockReduction128<float> : \
    spray::BlockReduction128<float>::ompReduce(&omp_out, &omp_in))                         \
    initializer (spray::BlockReduction128<float>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : spray::BlockReduction4096<float> : \
    spray::BlockReduction4096<float>::ompReduce(&omp_out, &omp_in))                         \
    initializer (spray::BlockReduction4096<float>::ompInit(&omp_priv, &omp_orig))
#pragma omp declare reduction(+ : spray::BlockReduction4096<double> : \
    spray::BlockReduction4096<double>::ompReduce(&omp_out, &omp_in))                         \
    initializer (spray::BlockReduction4096<double>::ompInit(&omp_priv, &omp_orig))
#endif
