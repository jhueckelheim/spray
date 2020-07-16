#ifndef MAP_REDUCTION_HPP
#define MAP_REDUCTION_HPP

#ifdef _USE_BTREE
#include "cpp-btree/btree_map.h"
#define mapType btree::btree_map<int,contentType>
#else
#include <map>
#define mapType std::map<int,contentType>
#endif

template <typename contentType> class MapArray {
  public:
    MapArray();
    MapArray(contentType* orig);
    void increment(int idx, contentType val);
    long getMemSize();
    static void ompReduce(MapArray<contentType> *out, MapArray<contentType> *in);
    contentType& operator[](int idx);

  private:
    mapType mapContent;
    contentType* denseContent;
    bool isOrig;
};

template class MapArray<double>;
template class MapArray<float>;

#pragma omp declare reduction(+ : MapArray<double> : \
  MapArray<double>::ompReduce(&omp_out, &omp_in))

#pragma omp declare reduction(+ : MapArray<float> : \
  MapArray<float>::ompReduce(&omp_out, &omp_in))

#endif
