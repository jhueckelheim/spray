#ifndef BLOCKREDUCTION_HPP
#define BLOCKREDUCTION_HPP
#include <omp.h>

template <typename contentType> class BlockArray {
  public:
    BlockArray();
    BlockArray(int totalsize, contentType* orig, bool useLocks=false);
    ~BlockArray();
    long getMemSize();
    static void ompInit(BlockArray<contentType> *init, BlockArray<contentType> *orig);
    static void ompReduce(BlockArray<contentType> *out, BlockArray<contentType> *in);
    contentType& operator[](int idx);

  private:
    contentType** blocks;
    omp_lock_t* writelocks;
    int nblocks;
    int totalsize;
    long memsize;
    bool initialized;
    bool useLocks;
    BlockArray<contentType>* orig; 
    void createBlockIfNeeded(int block);
};

template class BlockArray<double>;
template class BlockArray<float>;

#pragma omp declare reduction(+ : BlockArray<double> : \
  BlockArray<double>::ompReduce(&omp_out, &omp_in))                         \
  initializer (BlockArray<double>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : BlockArray<float> : \
  BlockArray<float>::ompReduce(&omp_out, &omp_in))                         \
  initializer (BlockArray<float>::ompInit(&omp_priv, &omp_orig))

#endif
