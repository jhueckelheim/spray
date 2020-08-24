#ifndef KEEPERREDUCTION_HPP
#define KEEPERREDUCTION_HPP
#include <assert.h>
#include <stdlib.h>
#include <utility>
#include <deque>
#include <map>
#include <omp.h>

namespace spray {
  template <typename contentType>
    using pendingScalar = std::pair<int, contentType>;
  template <typename contentType>
    using pendingDeque = std::deque<pendingScalar<contentType>>;
  template <typename contentType>
    using pending2Deque = std::deque<pendingDeque<contentType>*>;
  template <typename contentType>
    using pending2Map = std::map<int,pendingDeque<contentType>*>;

  template <typename contentType> class KeeperReduction {
    public:
      KeeperReduction() {
        this->initialized = false;
        this->threadID = omp_get_thread_num();
      }
  
      KeeperReduction(int size, contentType* orig, int numThreads = -1) {
        if(numThreads == -1) {
          this->numThreads = omp_get_max_threads();
        }
        this->pendingUpdates = new pending2Deque<contentType>[this->numThreads];
        this->content = orig;
        this->initialized = true;
        this->size = size;
        this->memsize = 0;
        this->isOrig = true;
        this->locks = new omp_lock_t[this->numThreads];
        for(int i=0; i<this->numThreads; i++) {
          omp_init_lock(&(this->locks[i]));
        }
      }
  
      static void ompInit(KeeperReduction<contentType> *init, KeeperReduction<contentType> *orig) {
        assert(orig->initialized);
        init->initialized = true;
        init->size = orig->size;
        init->memsize = 0;
        init->isOrig = false;
        init->pendingUpdates = orig->pendingUpdates;
        init->content = orig->content;
        init->locks = orig->locks;
        init->numThreads = orig->numThreads;
      }
  
      ~KeeperReduction() {
        if(this->isOrig) {
          delete[] this->pendingUpdates;
          for(int i=0; i<this->numThreads; i++) {
            omp_destroy_lock(&(this->locks[i]));
          }
          delete(this->locks);
        }
        else {
          #pragma omp barrier
          auto myPendingUpdates = this->pendingUpdates[this->threadID];
          auto end2d = myPendingUpdates.end();
          for(auto it2d = myPendingUpdates.begin(); it2d != end2d; it2d++) {
            auto curDeque = *it2d;
            auto curEnd = curDeque->end();
            for(auto it = curDeque->begin(); it != curEnd; it++) {
              this->content[it->first] += it->second;
            }
            delete(curDeque);
          }
        }
      }
  
      contentType& operator[](int idx) {
        int owner = idx * this->numThreads / this->size;
        if(owner == this->threadID) {
          return this->content[idx];
        }
        else {
          if(this->myContributions.find(owner) == this->myContributions.end()) {
            this->myContributions[owner] = new pendingDeque<contentType>;
            omp_set_lock(&(this->locks[this->threadID]));
            this->pendingUpdates[this->threadID].push_back(this->myContributions[owner]);
            omp_unset_lock(&(this->locks[this->threadID]));
          }
          pendingScalar<contentType> neutral(idx, 0);
          this->myContributions[owner]->push_back(neutral);
          contentType& neutralContent = this->myContributions[owner]->back().second;
          this->memsize += sizeof(pendingScalar<contentType>);
          return neutralContent;
        }
      }
  
      long getMemSize() {
        return this->memsize;
      }
  
      static void ompReduce(KeeperReduction<contentType> *out, KeeperReduction<contentType> *in) {
        out->memsize += in->memsize;
      }
  
    private:
      contentType* content;
      bool initialized;
      int size;
      long memsize;
      bool isOrig;
      pending2Deque<contentType>* pendingUpdates;
      pending2Map<contentType> myContributions;
      omp_lock_t* locks;
      int numThreads;
      int threadID;
  };

  #pragma omp declare reduction(+ : KeeperReduction<double> : \
    KeeperReduction<double>::ompReduce(&omp_out, &omp_in))                         \
    initializer (KeeperReduction<double>::ompInit(&omp_priv, &omp_orig))
  
  #pragma omp declare reduction(+ : KeeperReduction<float> : \
    KeeperReduction<float>::ompReduce(&omp_out, &omp_in))                         \
    initializer (KeeperReduction<float>::ompInit(&omp_priv, &omp_orig))
}
#endif
