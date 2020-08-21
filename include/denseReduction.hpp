#ifndef DENSEREDUCTION_HPP
#define DENSEREDUCTION_HPP
#include <assert.h>
#include <stdlib.h>

namespace spray {
template <typename contentType, unsigned alignment = 256> class DenseReduction {
public:
  DenseReduction() {}

  DenseReduction(unsigned size, contentType *orig)
      : content(orig), size(size) {}

  ~DenseReduction() {
    if (!isOriginal())
      free(this->content);
  }

  long getMemSize() { return this->memsize; }

  bool isOriginal() const { return isInitialized() && memsize == 0; }
  bool isInitialized() const { return content; }

  static void ompInit(DenseReduction<contentType> *__restrict__ init,
                      DenseReduction<contentType> *__restrict__ orig) {
    if (!init->isOriginal())
      free(init->content);

    assert(orig->isInitialized());
    init->size = orig->size;
    init->memsize = orig->size * sizeof(contentType);
    init->content = reinterpret_cast<contentType *>(
        aligned_alloc(alignment, init->memsize));
  }

  static void ompReduce(DenseReduction<contentType> *__restrict__ out,
                        DenseReduction<contentType> *__restrict__ in) {
    assert(out->isInitialized() && in->isInitialized());
    assert(out->size == in->size);
    out->memsize += in->memsize;

#pragma omp simd aligned(out->content, in->content : alignment)
    for (int i = 0; i < out->size; i++)
      out->content[i] += in->content[i];
  }

  contentType &operator[](int idx) {
    assert(this->isInitialized());
    return this->content[idx];
  }

private:
  contentType *content = nullptr;

  /// Size of the content.
  unsigned size = 0;

  /// Total memory size used by this reduction (tree).
  unsigned memsize = 0;
};

#pragma omp declare reduction(+ : DenseReduction<double> : \
    DenseReduction<double>::ompReduce(&omp_out, &omp_in))                         \
    initializer (DenseReduction<double>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : DenseReduction<float> : \
    DenseReduction<float>::ompReduce(&omp_out, &omp_in))                         \
    initializer (DenseReduction<float>::ompInit(&omp_priv, &omp_orig))
} // namespace spray
#endif
