ifeq ($(CC),icc)
include make_intel.inc
else
ifeq ($(CC),clang)
include make_clang.inc
else
include make_gnu.inc
endif
endif

BLOCKSIZE=4096
TESTSIZE=16384

all: test

clean:
	rm build/* bin/*

#############################
# Testing
#############################

test: bin/test_atomicreduce bin/test_ompreduce bin/test_blockreduce bin/test_containeratomicreduce bin/test_containerdensereduce bin/test_mapreduce bin/test_btreereduce bin/test_keeperreduce bin/test_awblockreduce
	bin/test_atomicreduce
	bin/test_ompreduce
	bin/test_blockreduce
	bin/test_containeratomicreduce
	bin/test_containerdensereduce
	bin/test_mapreduce
	bin/test_btreereduce
	bin/test_keeperreduce
	bin/test_awblockreduce

build/main.o: tests/main.c
	$(CXX) $^ -c $(CXXF) -Iinclude -DNSIZE=$(TESTSIZE) -o $@

bin/test_atomicreduce: build/main.o tests/test_atomicReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_ompreduce: build/main.o tests/test_ompReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_containeratomicreduce: build/main.o tests/test_containeratomicReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_containerdensereduce: build/main.o tests/test_containerdenseReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_mapreduce: build/main.o tests/test_mapReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_btreereduce: build/main.o tests/test_btreeReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -D_USE_BTREE -o $@

bin/test_blockreduce: build/main.o tests/test_blockReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -DBSIZE=$(BLOCKSIZE) -o $@

bin/test_keeperreduce: build/main.o tests/test_keeperReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@

bin/test_awblockreduce: build/main.o tests/test_AWBlockReduction.cpp
	$(CXX) $^ $(CXXF) -Iinclude -o $@
