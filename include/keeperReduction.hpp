#ifndef KEEPERREDUCTION_HPP
#define KEEPERREDUCTION_HPP
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

namespace spray {
  template <typename contentType, unsigned blocksize> class UpdateChunk {
    public:
      UpdateChunk() {
        this->top = 0;
        this->next = nullptr;
      }
      contentType* nextRef(int idx) {
        if(this->top < blocksize) {
          contentType* ret = &this->content[this->top];
          this->indices[this->top] = idx;
          this->top++;
          return ret;
        }
        else return nullptr;
      }
      contentType content[blocksize];
      int indices[blocksize];
      int top = 0;
      UpdateChunk<contentType, blocksize>* next = nullptr;
  };

  template <typename contentType> class UpdateOnceScalar {
  public:
    UpdateOnceScalar(contentType *val) { this->val = val; }
  
    void operator+=(contentType rhs) {
      *(this->val) = rhs;
    }
  
  private:
    contentType *val;
  };

  template <typename contentType, unsigned blocksize = 256> class KeeperReduction {
    public:
      KeeperReduction() {
        this->initialized = false;
        this->threadID = omp_get_thread_num();
      }
  
      KeeperReduction(int size, contentType* orig, int numThreads = -1) {
        if(numThreads == -1) {
          this->numThreads = omp_get_max_threads();
        }
        this->content = orig;
        this->initialized = true;
        this->size = size;
        this->memsize = 0;
        this->isOrig = true;
        this->allIncomingUpdates = new UpdateChunk<contentType, blocksize>**[this->numThreads];
        auto upd = new UpdateChunk<contentType, blocksize>*[this->numThreads*this->numThreads];
        for(int i=0; i<this->numThreads; i++) {
          this->allIncomingUpdates[i] = &upd[i*this->numThreads];
          #pragma omp simd
          for(int j=0; j<this->numThreads; j++) {
            this->allIncomingUpdates[i][j] = nullptr;
          }
        }
      }
  
      static void ompInit(KeeperReduction<contentType,blocksize> *init, KeeperReduction<contentType,blocksize> *orig) {
        assert(orig->initialized);
        init->initialized = true;
        init->size = orig->size;
        init->memsize = 0;
        init->isOrig = false;
        init->content = orig->content;
        init->numThreads = orig->numThreads;
        init->allIncomingUpdates = orig->allIncomingUpdates;
        init->myOutgoingUpdates = new UpdateChunk<contentType, blocksize>*[orig->numThreads];
        #pragma omp simd
        for(int i=0; i<orig->numThreads; i++) {
          init->myOutgoingUpdates[i] = nullptr;
        }
      }
  
      ~KeeperReduction() {
        if(this->isOrig) {
          delete[] this->allIncomingUpdates[0];
          delete[] this->allIncomingUpdates;
        }
        else {
          #pragma omp barrier
          for(int i=0; i<this->numThreads; i++) {
            auto myIncomingUpdates = this->allIncomingUpdates[this->threadID][i];
            while(myIncomingUpdates) {
              for(int i=0; i<myIncomingUpdates->top; i++) {
                this->content[myIncomingUpdates->indices[i]] += myIncomingUpdates->content[i];
              }
              auto oldIncomingUpdates = myIncomingUpdates;
              myIncomingUpdates = myIncomingUpdates->next;
              delete oldIncomingUpdates;
            }
          }
          delete[] myOutgoingUpdates;
        }
      }
  
      UpdateOnceScalar<contentType> operator[](int idx) {
        int owner = idx * this->numThreads / this->size;
        if(owner == this->threadID) {
          return UpdateOnceScalar<contentType>(this->content+idx);
        }
        else {
          if(!this->myOutgoingUpdates[owner]) {
            auto newChunk = new UpdateChunk<contentType, blocksize>;
            this->myOutgoingUpdates[owner] = newChunk;
            this->allIncomingUpdates[owner][this->threadID] = newChunk;
          }
          contentType* ret = this->myOutgoingUpdates[owner]->nextRef(idx);
          if(!ret) {
            auto newChunk = new UpdateChunk<contentType, blocksize>;
            this->myOutgoingUpdates[owner]->next = newChunk;
            this->myOutgoingUpdates[owner] = newChunk;
            ret = this->myOutgoingUpdates[owner]->nextRef(idx);
          }
          return UpdateOnceScalar<contentType>(ret);
        }
      }
  
      long getMemSize() {
        return this->memsize;
      }
  
      static void ompReduce(KeeperReduction<contentType,blocksize> *out, KeeperReduction<contentType,blocksize> *in) {
        out->memsize += in->memsize;
      }
  
    private:
      contentType* content;
      bool initialized;
      int size;
      long memsize;
      bool isOrig;
      UpdateChunk<contentType,blocksize>** myOutgoingUpdates;
      UpdateChunk<contentType,blocksize>*** allIncomingUpdates;
      int numThreads;
      int threadID;
  };
}
#pragma omp declare reduction(+ : spray::KeeperReduction<double> : \
  spray::KeeperReduction<double>::ompReduce(&omp_out, &omp_in))                         \
  initializer (spray::KeeperReduction<double>::ompInit(&omp_priv, &omp_orig))

#pragma omp declare reduction(+ : spray::KeeperReduction<float> : \
  spray::KeeperReduction<float>::ompReduce(&omp_out, &omp_in))                         \
  initializer (spray::KeeperReduction<float>::ompInit(&omp_priv, &omp_orig))
#endif
