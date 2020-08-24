#ifndef MAP_REDUCTION_HPP
#define MAP_REDUCTION_HPP

#include "cpp-btree/btree_map.h"
#include <map>

namespace spray {

template <typename mapType, typename contentType> struct MapReduction {
  MapReduction() {}
  MapReduction(contentType *orig) : denseContent(orig) {}

  long getMemSize() { return mapContent.max_size(); }

  static void ompReduce(MapReduction<mapType, contentType> *out,
                        MapReduction<mapType, contentType> *in) {
    if (out->denseContent) {
      for (const auto &it : in->mapContent)
        out->denseContent[it.first] += it.second;
    } else {
      for (const auto &it : in->mapContent)
        (*out)[it.first] += it.second;
    }
  }

  contentType &operator[](int idx) {
    if (denseContent)
      return (denseContent)[idx];

    auto it = this->mapContent.find(idx);
    if (it == this->mapContent.end())
      it = this->mapContent.insert(it, {idx, contentType()});
    return it->second;
  }

private:
  mapType mapContent;
  contentType * const denseContent = nullptr;
};

template <typename contentType>
using BtreeReduction =
    MapReduction<btree::btree_map<int, contentType>, contentType>;

template <typename contentType>
using STLMapReduction = MapReduction<std::map<int, contentType>, contentType>;

#pragma omp declare reduction(+ : BtreeReduction<double> : \
    BtreeReduction<double>::ompReduce(&omp_out, &omp_in))

#pragma omp declare reduction(+ : BtreeReduction<float> : \
    BtreeReduction<float>::ompReduce(&omp_out, &omp_in))

#pragma omp declare reduction(+ : STLMapReduction<double> : \
    STLMapReduction<double>::ompReduce(&omp_out, &omp_in))

#pragma omp declare reduction(+ : STLMapReduction<float> : \
    STLMapReduction<float>::ompReduce(&omp_out, &omp_in))
} // namespace spray
#endif
