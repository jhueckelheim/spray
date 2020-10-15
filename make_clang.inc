CXX=clang++
CXXF=-fopenmp -O3 -march=native -isystem ../benchmark/clang/benchmark/include -L../benchmark/clang/benchmark/build/src -std=c++17
CC=clang
CF=-fopenmp -O3 -march=native ../benchmark/clang/benchmark/include -L../benchmark/clang/benchmark/build/src 
