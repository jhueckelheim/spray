ifeq ($(CC),icc)
include ../../make_intel.inc
else
ifeq ($(CC),clang)
include ../../make_clang.inc
else
include ../../make_gnu.inc
endif
endif

BLOCKSIZE=4096

all: exec

test: exec
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 1 100
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 2 100
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 3 100
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 4 100
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 5 100

benchmark: benchmark.cpp
	$(CXX) $^ $(CXXF) -I../../include -lbenchmark -lpthread -DBSIZE=$(BLOCKSIZE) -Dreal=float -o $@

timings.csv: benchmark
	ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings.json
	python filtertimes.py timings.json > timings.csv

clean:
	rm *.o exec

exec: main.cpp
	$(CXX) $^ $(CXXF) -I../../include/ -DBSIZE=$(BLOCKSIZE) -o $@
