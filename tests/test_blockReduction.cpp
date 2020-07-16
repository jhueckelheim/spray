#include "blockReduction.hpp"

long testReduction(double* out, int n) {
  int i;
  BlockArray<double> arr_p(n, out);
  #pragma omp parallel for reduction(+:arr_p)
  for(i=1; i<n-1; i++) {
    //arr_p.increment(i-1, 1.0);
    //arr_p.increment(i  , 2.0);
    //arr_p.increment(i+1, 4.0);
    arr_p[i-1] += 1.0;
    arr_p[i  ] += 2.0;
    arr_p[i+1] += 4.0;
  }
  return arr_p.getMemSize();
}
