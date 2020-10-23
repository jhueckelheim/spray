ifeq ($(CC),icc)
include ../../make_intel.inc
else
include ../../make_gnu.inc
endif

BLOCKSIZE=4096

all: exec

test: exec s3dkt3m2.mtx circuit5M.mtx debr.mtx
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 1   # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 2   # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 3   # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 4   # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 5   # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 6   # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 7   # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 8   # s3dkt3m2.mtx 
	#OMP_PLACES=sockets /usr/bin/time -v ./exec debr.mtx  1000 9  #  s3dkt3m2.mtx
	#OMP_PLACES=sockets /usr/bin/time -v ./exec debr.mtx  1000 10 #  s3dkt3m2.mtx
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 11  # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 12  # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 13  # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 14  # s3dkt3m2.mtx 
	OMP_PLACES=sockets /usr/bin/time -v ./exec  debr.mtx 1000 15  # s3dkt3m2.mtx 

benchmark: benchmark.cpp mmio.o coo_csr_reader.o
	$(CXX) $^ -I../../include -lbenchmark -lpthread $(CXXF) -DBSIZE=$(BLOCKSIZE) -o $@

benchmark_circuit: benchmark.cpp mmio.o coo_csr_reader.o
	$(CXX) $^ -I../../include -lbenchmark -lpthread $(CXXF) -DBSIZE=$(BLOCKSIZE) -DCIRCUIT -o $@

timings.csv: benchmark
	ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark --benchmark_format=json --benchmark_repetitions=10 > timings.json
	python filtertimes.py timings.json debr > timings_debr.csv
	python filtertimes.py timings.json s3dkt3m2 > timings_s3dkt3m2.csv

timings_circuit.csv: benchmark_circuit
	ulimit -s unlimited; OMP_STACKSIZE=1024M OMP_PLACES=sockets ./benchmark_circuit --benchmark_format=json --benchmark_repetitions=1 > timings_circuit.json
	python filtertimes.py timings_circuit.json circuit5M noaggregate > timings_circuit5M.csv

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
