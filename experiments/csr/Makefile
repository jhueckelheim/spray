ifeq ($(CC),icc)
include ../../make_intel.inc
else
include ../../make_gnu.inc
endif

all: exec

test: exec s3dkt3m2.mtx circuit5M.mtx
	./exec s3dkt3m2.mtx 10
	./exec circuit5M.mtx 1

clean:
	rm *.o exec

s3dkt3m2.mtx:
	wget ftp://math.nist.gov/pub/MatrixMarket2/misc/cylshell/s3dkt3m2.mtx.gz
	gunzip s3dkt3m2.mtx.gz

circuit5M.mtx:
	wget https://www.cise.ufl.edu/research/sparse/MM/Freescale/circuit5M.tar.gz
	tar vxzf circuit5M.tar.gz
	mv circuit5M/circuit5M.mtx .
	rm -r circuit5M

exec: mmio.o coo_csr_reader.o main.cpp ../../build/blockReduction.o
	$(CXX) $^ $(CXXF) -I../../include -o $@

mmio.o: mmio.c
	$(CC) $^ $(CF) -c -o $@

coo_csr_reader.o: coo_csr_reader.cpp
	$(CXX) $^ $(CXXF) -I../../include -c -o $@
