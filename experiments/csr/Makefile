ifeq ($(CC),icc)
include ../../make_intel.inc
else
include ../../make_gnu.inc
endif

BLOCKSIZE=4096

all: exec

test: exec s3dkt3m2.mtx circuit5M.mtx debr.mtx
	OMP_PLACES=sockets ./exec s3dkt3m2.mtx 1000
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec circuit5M.mtx 10
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec debr.mtx 100

benchmark: benchmark.cpp mmio.o coo_csr_reader.o
	$(CXX) $^ -isystem ../benchmark/include -L../benchmark/build/src -I../../include -lbenchmark -lpthread $(CXXF) -DBSIZE=$(BLOCKSIZE) -o $@

benchmark_circuit: benchmark.cpp mmio.o coo_csr_reader.o
	$(CXX) $^ -isystem ../benchmark/include -L../benchmark/build/src -I../../include -lbenchmark -lpthread $(CXXF) -DBSIZE=$(BLOCKSIZE) -DCIRCUIT -o $@

timings.csv: benchmark
	ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings.json
	python filtertimes.py timings.json > timings.csv

timings_circuit.csv: benchmark_circuit
	ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark_circuit --benchmark_format=json --benchmark_repetitions=10 > timings_circuit.json
	python filtertimes.py timings_circuit.json > timings_circuit.csv

clean:
	rm *.o exec s3dkt3m2.mtx.gz circuit5M.tar.gz debr.tar.gz
	rm -rf debr
	rm -rf circuit5M

s3dkt3m2.mtx:
	wget ftp://math.nist.gov/pub/MatrixMarket2/misc/cylshell/s3dkt3m2.mtx.gz
	gunzip s3dkt3m2.mtx.gz

circuit5M.mtx:
	wget https://www.cise.ufl.edu/research/sparse/MM/Freescale/circuit5M.tar.gz
	tar vxzf circuit5M.tar.gz
	mv circuit5M/circuit5M.mtx .

debr.mtx:
	wget https://www.cise.ufl.edu/research/sparse/MM/AG-Monien/debr.tar.gz
	tar vxzf debr.tar.gz
	mv debr/debr.mtx .

exec: mmio.o coo_csr_reader.o main.cpp
	$(CXX) $^ $(CXXF) -I../../include -DBSIZE=$(BLOCKSIZE) -o $@

mmio.o: mmio.c
	$(CC) $^ $(CF) -c -o $@

coo_csr_reader.o: coo_csr_reader.cpp
	$(CXX) $^ $(CXXF) -I../../include -c -o $@
