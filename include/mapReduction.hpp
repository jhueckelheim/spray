#ifndef MAP_REDUCTION_HPP
#define MAP_REDUCTION_HPP

#include "cpp-btree/btree_map.h"
#include <map>

namespace spray {
  template <typename mapType, typename contentType> class MapReduction {
    public:
      MapReduction() {
        this->isOrig = false;
      }
  
      MapReduction(contentType* orig) {
        this->isOrig = true;
        this->denseContent = orig;
      }
  
      void increment(int idx, contentType val) {
        if(this->mapContent.find(idx) == this->mapContent.end()) {
          this->mapContent[idx] = val;
        }
        else{
          this->mapContent[idx] += val;
        }
      }
  
      long getMemSize() {
        // TODO: need to implement a way to measure memory footprint of map
        return 0;
      }
  
      static void ompReduce(MapReduction<mapType,contentType> *out, MapReduction<mapType,contentType> *in) {
        int i,j;
        if(out->isOrig) {
          auto it = in->mapContent.begin();
          while(it != in->mapContent.end()) {
            out->denseContent[it->first] += it->second;
            it++;
          }
        }
        else {
          auto it = in->mapContent.begin();
          while(it != in->mapContent.end()) {
            out->increment(it->first, it->second);
            it++;
          }
        }
      }
  
      contentType& operator[](int idx) {
        if(this->mapContent.find(idx) == this->mapContent.end()) {
          this->mapContent[idx] = 0.0;
        }
        return this->mapContent[idx];
      }
  
    private:
      mapType mapContent;
      contentType* denseContent;
      bool isOrig;
  };
  
  template <typename contentType>
  using BtreeReduction = MapReduction<btree::btree_map<int,contentType>, contentType>;
  
  template <typename contentType>
  using STLMapReduction = MapReduction<std::map<int,contentType>, contentType>;

  #pragma omp declare reduction(+ : BtreeReduction<double> : \
    BtreeReduction<double>::ompReduce(&omp_out, &omp_in))
  
  #pragma omp declare reduction(+ : BtreeReduction<float> : \
    BtreeReduction<float>::ompReduce(&omp_out, &omp_in))
  
  #pragma omp declare reduction(+ : STLMapReduction<double> : \
    STLMapReduction<double>::ompReduce(&omp_out, &omp_in))
  
  #pragma omp declare reduction(+ : STLMapReduction<float> : \
    STLMapReduction<float>::ompReduce(&omp_out, &omp_in))
}
#endif
