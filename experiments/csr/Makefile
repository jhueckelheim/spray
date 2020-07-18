CXX=g++
CXXF=-fopenmp -I../../include -O3

all: exec

test: exec s3dkt3m2.mtx
	./exec s3dkt3m2.mtx 10

clean:
	rm *.o exec

s3dkt3m2.mtx:
	wget ftp://math.nist.gov/pub/MatrixMarket2/misc/cylshell/s3dkt3m2.mtx.gz
	gunzip s3dkt3m2.mtx.gz

exec: mmio.o coo_csr_reader.o main.cpp ../../build/blockReduction.o
	$(CXX) $^ $(CXXF) -o $@

mmio.o: mmio.c
	$(CC) $^ $(CF) -c -o $@

coo_csr_reader.o: coo_csr_reader.cpp
	$(CXX) $^ $(CXXF) -c -o $@
