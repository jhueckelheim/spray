#ifndef KEEPERREDUCTION_HPP
#define KEEPERREDUCTION_HPP
#include <assert.h>
#include <stdlib.h>
#include <utility>
#include <deque>
#include <map>
#include <omp.h>

namespace spray {
  template <typename contentType, unsigned blocksize> class UpdateChunk {
    public:
      contentType* nextRef(int idx) {
        if(top < blocksize) {
          contentType* ret = &content[top];
          content[top] = 0.0;
          indices[top] = idx;
          top++;
          return ret;
        }
        else return nullptr;
      }
      contentType content[blocksize];
      int indices[blocksize];
      int top = 0;
      UpdateChunk<contentType, blocksize>* next = nullptr;
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
        for(int i=0; i<this->numThreads; i++) {
          this->allIncomingUpdates[i] = new UpdateChunk<contentType, blocksize>*[this->numThreads];
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
      }
  
      ~KeeperReduction() {
        if(this->isOrig) {
          for(int i=0; i<this->numThreads; i++) {
            delete[] this->allIncomingUpdates[i];
          }
          delete[] this->allIncomingUpdates;
        }
        else {
          delete[] myOutgoingUpdates;
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
        }
      }
  
      contentType& operator[](int idx) {
        int owner = idx * this->numThreads / this->size;
        if(owner == this->threadID) {
          return this->content[idx];
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
          return (*ret);
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
