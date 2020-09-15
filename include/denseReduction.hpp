#ifndef DENSEREDUCTION_HPP
#define DENSEREDUCTION_HPP
#include <assert.h>
#include <stdlib.h>

namespace spray {
template <typename contentType, unsigned alignment = 256> class DenseReduction {
public:
  DenseReduction() {}

  DenseReduction(unsigned size, contentType *orig)
      : content(orig), size(size), original(true) {}

  ~DenseReduction() {
    if (!original)
      free(this->content);
  }

  bool isInitialized() const { return content; }

  static void ompInit(DenseReduction<contentType> *__restrict__ init,
                      DenseReduction<contentType> *__restrict__ orig) {
    assert(orig->isInitialized());
    init->size = orig->size;
    init->content = reinterpret_cast<contentType *>(
        aligned_alloc(alignment, orig->size * sizeof(contentType)));
  }

  static void ompReduce(DenseReduction<contentType> *__restrict__ out,
                        DenseReduction<contentType> *__restrict__ in) {
    assert(out->isInitialized() && in->isInitialized());
    assert(out->size == in->size);
    contentType* outc = out->content;
    contentType* inc = in->content;
    if(out->original) {
      for (int i = 0; i < out->size; i++)
        outc[i] += inc[i];
    }
    else {
#pragma omp simd aligned(outc, inc : alignment)
      for (int i = 0; i < out->size; i++)
        outc[i] += inc[i];
    }
  }

  contentType &operator[](int idx) {
    assert(this->isInitialized());
    return this->content[idx];
  }

private:
  contentType *content = nullptr;

  /// Size of the content.
  unsigned size = 0;

  /// A flag marking the object pointing to the original (user-provided) data.
  bool original = false;
};
} // namespace spray

#pragma omp declare reduction(+ : spray::DenseReduction<double> : \
    spray::DenseReduction<double>::ompReduce(&omp_out, &omp_in))                         \
    initializer (spray::DenseReduction<double>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : spray::DenseReduction<float> : \
    spray::DenseReduction<float>::ompReduce(&omp_out, &omp_in))                         \
    initializer (spray::DenseReduction<float>::ompInit(&omp_priv, &omp_orig))
#endif
