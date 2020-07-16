long testReduction(double* out, int n) {
  int i;
  #pragma omp parallel for
  for(i=1; i<n-1; i++) {
    #pragma omp atomic update
    out[i-1] += 1.0;
    #pragma omp atomic update
    out[i  ] += 2.0;
    #pragma omp atomic update
    out[i+1] += 4.0;
  }
  return 0;
}
