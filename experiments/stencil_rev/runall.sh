#!/bin/sh
module use /soft/modulefiles

module load intel/2019
icpc benchmark.cpp -qopenmp -O3 -xHost -std=c++11 -mkl -I ../benchmark/intel/benchmark/include -L../benchmark/intel/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_intel_O3.json
python filtertimes.py timings.json > timings_intel_O3.csv
icpc benchmark.cpp -qopenmp -O2 -xHost -std=c++11 -mkl -I ../benchmark/intel/benchmark/include -L../benchmark/intel/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_intel_O2.json
python filtertimes.py timings.json > timings_intel_O2.csv
icpc benchmark.cpp -qopenmp -O1 -xHost -std=c++11 -mkl -I ../benchmark/intel/benchmark/include -L../benchmark/intel/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_intel_O1.json
python filtertimes.py timings.json > timings_intel_O1.csv

module load llvm/release-11.0.0
clang++ benchmark.cpp -fopenmp -O3 -march=native -isystem ../benchmark/clang/benchmark/include -L../benchmark/clang/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_clang_O3.json
python filtertimes.py timings.json > timings_clang_O3.csv
clang++ benchmark.cpp -fopenmp -O2 -march=native -isystem ../benchmark/clang/benchmark/include -L../benchmark/clang/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_clang_O2.json
python filtertimes.py timings.json > timings_clang_O2.csv
clang++ benchmark.cpp -fopenmp -O1 -march=native -isystem ../benchmark/clang/benchmark/include -L../benchmark/clang/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_clang_O1.json
python filtertimes.py timings.json > timings_clang_O1.csv

module unload llvm/release-11.0.0
module unload gcc
module load gcc/9.2.0
g++ benchmark.cpp -fopenmp -O3 -march=native -isystem ../benchmark/gnu/benchmark/include -L../benchmark/gnu/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_gnu_O3.json
python filtertimes.py timings.json > timings_gnu_O3.csv
g++ benchmark.cpp -fopenmp -O2 -march=native -isystem ../benchmark/gnu/benchmark/include -L../benchmark/gnu/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_gnu_O2.json
python filtertimes.py timings.json > timings_gnu_O2.csv
g++ benchmark.cpp -fopenmp -O1 -march=native -isystem ../benchmark/gnu/benchmark/include -L../benchmark/gnu/benchmark/build/src  -I../../include -lbenchmark -lpthread -DBSIZE=4096 -Dreal=float -o benchmark
ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings_gnu_O1.json
python filtertimes.py timings.json > timings_gnu_O1.csv
rm -rf ../benchmark
