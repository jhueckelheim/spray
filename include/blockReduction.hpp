#ifndef BLOCKREDUCTION_HPP
#define BLOCKREDUCTION_HPP

template <typename contentType> class BlockArray {
  public:
    BlockArray();
    BlockArray(int totalsize);
    BlockArray(int totalsize, contentType* orig);
    ~BlockArray();
    void init(int totalsize);
    void increment(int idx, contentType val);
    long getMemSize();
    static void ompInit(BlockArray<contentType> *init, BlockArray<contentType> *orig);
    static void ompReduce(BlockArray<contentType> *out, BlockArray<contentType> *in);
    contentType& operator[](int idx);

  private:
    contentType** blocks;
    int nblocks;
    int totalsize;
    long memsize;
    int isOrig;
    void createIfNeeded(int block);
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
