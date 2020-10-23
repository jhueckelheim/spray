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
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 1 10
	ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 2 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 3 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 4 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 5 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 6 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 7 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 8 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 9 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 10 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 11 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 12 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 13 10
	ulimit -s unlimited; OMP_PLACES=sockets /usr/bin/time -v ./exec 10000000 2 14 10

benchmark: benchmark.cpp
	$(CXX) $^ $(CXXF) -I../../include -lbenchmark -lpthread -DBSIZE=$(BLOCKSIZE) -Dreal=float -o $@

timings.csv: benchmark
	ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings.json
	python filtertimes.py timings.json > timings.csv

clean:
	rm *.o exec

exec: main.cpp
	$(CXX) $^ $(CXXF) -I../../include/ -DBSIZE=$(BLOCKSIZE) -o $@
