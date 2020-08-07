#ifndef DENSEREDUCTION_HPP
#define DENSEREDUCTION_HPP
#include <assert.h>
#include <stdlib.h>

namespace spray {
  template <typename contentType> class DenseReduction {
    public:
      DenseReduction() {
        this->initialized = false;
      }
  
      DenseReduction(int size, contentType* orig) {
        this->content = orig;
        this->initialized = true;
        this->size = size;
        this->memsize = 0;
        this->isOrig = true;
      }
  
      ~DenseReduction() {
        if(!this->isOrig) {
          free(this->content);
        }
      }
  
      long getMemSize() {
        return this->memsize;
      }
  
      static void ompInit(DenseReduction<contentType> *init, DenseReduction<contentType> *orig) {
        assert(orig->initialized);
        init->initialized = true;
        init->size = orig->size;
        init->memsize = orig->size * sizeof(contentType);
        init->content = (contentType*) malloc(init->memsize);
        init->isOrig = false;
      }
  
      static void ompReduce(DenseReduction<contentType> *out, DenseReduction<contentType> *in) {
        assert(out->initialized && in->initialized);
        assert(out->size == in->size);
        for(int i=0; i<out->size; i++) {
          out->content[i] += in->content[i];
        }
        out->memsize += in->memsize;
      }
  
      contentType& operator[](int idx) {
        assert(this->initialized);
        return this->content[idx];
      }
  
    private:
      contentType* content;
      bool initialized;
      int size;
      long memsize;
      bool isOrig;
  };

  #pragma omp declare reduction(+ : DenseReduction<double> : \
    DenseReduction<double>::ompReduce(&omp_out, &omp_in))                         \
    initializer (DenseReduction<double>::ompInit(&omp_priv, &omp_orig))
  
  #pragma omp declare reduction(+ : DenseReduction<float> : \
    DenseReduction<float>::ompReduce(&omp_out, &omp_in))                         \
    initializer (DenseReduction<float>::ompInit(&omp_priv, &omp_orig))
}
#endif
