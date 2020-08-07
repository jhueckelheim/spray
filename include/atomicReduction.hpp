#ifndef ATOMICREDUCTION_HPP
#define ATOMICREDUCTION_HPP
#include <assert.h>

template <typename contentType> class AtomicScalar {
  public:
    AtomicScalar(contentType* val) {
      this->val = val;
    }

    void operator+=(contentType rhs) {
      #pragma omp atomic update
      *(this->val) += rhs;
    }
  private:
    contentType* val;
};

template <typename contentType> class AtomicArray {
  public:
    AtomicArray() {
      this->initialized = false;
    }

    AtomicArray(contentType* orig) {
      this->orig = orig;
      this->initialized = true;
    }

    long getMemSize() {
      return 0;
    }

    static void ompInit(AtomicArray<contentType> *init, AtomicArray<contentType> *orig) {
      assert(orig->initialized);
      init->orig = orig->orig;
      init->initialized = true;
    }

    static void ompReduce(AtomicArray<contentType> *out, AtomicArray<contentType> *in) {
      assert(out->initialized && in->initialized);
    }

    AtomicScalar<contentType> operator[](int idx) {
      assert(this->initialized);
      return AtomicScalar<contentType>(this->orig + idx);
    }

  private:
    contentType* orig;
    bool initialized;
};

#pragma omp declare reduction(+ : AtomicArray<double> : \
  AtomicArray<double>::ompReduce(&omp_out, &omp_in))                         \
  initializer (AtomicArray<double>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : AtomicArray<float> : \
  AtomicArray<float>::ompReduce(&omp_out, &omp_in))                         \
  initializer (AtomicArray<float>::ompInit(&omp_priv, &omp_orig))

#endif
