#ifndef ATOMICREDUCTION_HPP
#define ATOMICREDUCTION_HPP
#include <assert.h>

namespace spray {
template <typename contentType> class AtomicScalar {
public:
  AtomicScalar(contentType *val) { this->val = val; }

  void operator+=(contentType rhs) {
#pragma omp atomic update
    *(this->val) += rhs;
  }

private:
  contentType *val;
};

template <typename contentType> class AtomicReduction {
public:
  AtomicReduction() { this->initialized = false; }

  AtomicReduction(contentType *orig) {
    this->orig = orig;
    this->initialized = true;
  }

  long getMemSize() { return 0; }

  static void ompInit(AtomicReduction<contentType> *init,
                      AtomicReduction<contentType> *orig) {
    assert(orig->initialized);
    init->orig = orig->orig;
    init->initialized = true;
  }

  static void ompReduce(AtomicReduction<contentType> *out,
                        AtomicReduction<contentType> *in) {
    assert(out->initialized && in->initialized);
  }

  AtomicScalar<contentType> operator[](int idx) {
    assert(this->initialized);
    return AtomicScalar<contentType>(this->orig + idx);
  }

private:
  contentType *orig;
  bool initialized;
};

#pragma omp declare reduction(+ : AtomicReduction<double> : \
    AtomicReduction<double>::ompReduce(&omp_out, &omp_in))                         \
    initializer (AtomicReduction<double>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : AtomicReduction<float> : \
    AtomicReduction<float>::ompReduce(&omp_out, &omp_in))                         \
    initializer (AtomicReduction<float>::ompInit(&omp_priv, &omp_orig))
} // namespace spray
#endif
