CC=clang
ifeq ($(BSIZE),)
BSIZE=8
endif
CF=-fopenmp -I../../include/ -std=c11 -O3 -lm -march=native -DBSIZE=$(BSIZE) -g -fopenmp-version=51

all: run_sprlocks run_spr run_orig run_keeper

test: all
	./run_orig
	./run_spr
	./run_sprlocks
	./run_keeper

run_keeper: standalone.c
	$(CC) $^ $(CF) -DSPRAY_KEEPER -DUSELOCKS -o $@

run_sprlocks: standalone.c
	$(CC) $^ $(CF) -DSPRAY_NDBLOCK -DUSELOCKS -o $@

run_spr: standalone.c
	$(CC) $^ $(CF) -DSPRAY_NDBLOCK -o $@

run_orig: standalone.c
	$(CC) $^ $(CF) -o $@
