CC=clang
ifeq ($(BSIZE),)
BSIZE=8
endif
CF=-fopenmp -I../../include/ -std=c11 -O3 -lm -march=native -DBSIZE=$(BSIZE)

all: run_sprlocks run_spr run_orig

test: all
	./run_orig
	./run_spr
	./run_sprlocks

run_sprlocks: standalone.c
	$(CC) $^ $(CF) -DSPRAY -DUSELOCKS -o $@

run_spr: standalone.c
	$(CC) $^ $(CF) -DSPRAY -o $@

run_orig: standalone.c
	$(CC) $^ $(CF) -o $@
