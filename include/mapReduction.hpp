#ifndef MAP_REDUCTION_HPP
#define MAP_REDUCTION_HPP

#include "cpp-btree/btree_map.h"
#include <map>

namespace spray {

template <typename mapType, typename contentType> struct MapReduction {
  MapReduction() {}
  MapReduction(contentType *orig) : denseContent(orig) {}

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
} // namespace spray

#pragma omp declare reduction(+ : spray::BtreeReduction<double> : \
    spray::BtreeReduction<double>::ompReduce(&omp_out, &omp_in))

#pragma omp declare reduction(+ : spray::BtreeReduction<float> : \
    spray::BtreeReduction<float>::ompReduce(&omp_out, &omp_in))

#pragma omp declare reduction(+ : spray::STLMapReduction<double> : \
    spray::STLMapReduction<double>::ompReduce(&omp_out, &omp_in))

#pragma omp declare reduction(+ : spray::STLMapReduction<float> : \
    spray::STLMapReduction<float>::ompReduce(&omp_out, &omp_in))
#endif
