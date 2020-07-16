#include "mapReduction.hpp"

template <typename contentType>
MapArray<contentType>::MapArray() {
  this->isOrig = false;
}

template <typename contentType>
MapArray<contentType>::MapArray(contentType* orig) {
  this->isOrig = true;
  this->denseContent = orig;
}

template <typename contentType> contentType&
MapArray<contentType>::operator[](int idx) {
  if(this->mapContent.find(idx) == this->mapContent.end()) {
    this->mapContent[idx] = 0.0;
  }
  return this->mapContent[idx];
}

template <typename contentType> void
MapArray<contentType>::increment(int idx, contentType val) {
  if(this->mapContent.find(idx) == this->mapContent.end()) {
    this->mapContent[idx] = val;
  }
  else{
    this->mapContent[idx] += val;
  }
}

template <typename contentType> long
MapArray<contentType>::getMemSize() {
  // TODO: need to implement a way to measure memory footprint of map
  return 0;
}

template <typename contentType> void
MapArray<contentType>::ompReduce(MapArray<contentType> *out, MapArray<contentType> *in) {
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
