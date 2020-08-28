#include "blockReduction.hpp"

void testReduction(double* out, int n) {
  int i;
  spray::BlockReduction<double> arr_p(n, out);
  #pragma omp parallel for reduction(+:arr_p)
  for(i=1; i<n-1; i++) {
    arr_p[i-1] += 1.0;
    arr_p[i  ] += 2.0;
    arr_p[i+1] += 4.0;
  }
}
