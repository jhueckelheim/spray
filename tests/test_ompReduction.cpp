#include <omp.h>

void testReduction(double* out, int n) {
  int i;
  #pragma omp parallel for reduction(+:out[0:n])
  for(i=1; i<n-1; i++) {
    out[i-1] += 1.0;
    out[i  ] += 2.0;
    out[i+1] += 4.0;
  }
}

