ifeq ($(CC),icc)
include make_intel.inc
else
include make_gnu.inc
endif

BLOCKSIZE=4096
TESTSIZE=16384

all: build/blockReduction.o build/mapReduction.o build/btreeReduction.o 

clean:
	rm build/* bin/*

build/blockReduction.o: src/blockReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -c -DBSIZE=$(BLOCKSIZE) -o $@

build/mapReduction.o: src/mapReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -c -o $@

build/btreeReduction.o: src/mapReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -D_USE_BTREE -c -o $@

#############################
# Testing
#############################

test: bin/test_atomicreduce bin/test_ompreduce bin/test_blockreduce bin/test_mapreduce bin/test_btreereduce
	bin/test_atomicreduce
	bin/test_ompreduce
	bin/test_blockreduce
	bin/test_mapreduce
	bin/test_btreereduce

build/main.o: tests/main.c
	$(CXX) $^ -c $(CXXF) -Iinclude -DNSIZE=$(TESTSIZE) -o $@

bin/test_atomicreduce: build/main.o tests/test_atomicReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_ompreduce: build/main.o tests/test_ompReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_mapreduce: build/main.o tests/test_mapReduction.cpp build/mapReduction.o
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_btreereduce: build/main.o tests/test_mapReduction.cpp build/btreeReduction.o
	$(CXX) $^ $(CXXF) -Iinclude -D_USE_BTREE -o $@

bin/test_blockreduce: build/main.o tests/test_blockReduction.cpp build/blockReduction.o
	$(CXX) $^ $(CXXF) -Iinclude -DBSIZE=$(BLOCKSIZE) -o $@
