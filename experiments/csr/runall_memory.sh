#!/bin/sh
module use /soft/modulefiles
module load intel/2019

set -x
export OMP_PLACES=sockets
for nth in 1 2 4 8 16 28 56
do
	export OMP_NUM_THREADS=$nth
	make test CC=icc
done
