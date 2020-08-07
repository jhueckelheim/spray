ifeq ($(CC),icc)
include ../../make_intel.inc
else
include ../../make_gnu.inc
endif

BLOCKSIZE=4096

all: exec

test: exec
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 1 100
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 2 100
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 3 100
	ulimit -s unlimited; OMP_STACKSIZE=256M OMP_PLACES=sockets ./exec 10000000 2 4 100

clean:
	rm *.o exec

exec: main.cpp
	$(CXX) $^ $(CXXF) -I../../include/ -DBSIZE=$(BLOCKSIZE) -o $@
