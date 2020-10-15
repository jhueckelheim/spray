CXX=icpc
CXXF=-qopenmp -O3 -xHost -std=c++17 -mkl -I ../benchmark/intel/benchmark/include -L../benchmark/intel/benchmark/build/src 
CC=icc
CF=-qopenmp -O3 -xHost -mkl -I ../benchmark/intel/benchmark/include -L../benchmark/intel/benchmark/build/src 
