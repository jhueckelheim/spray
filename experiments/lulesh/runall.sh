#!/bin/sh
module use /soft/modulefiles
module load intel/2019

make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=LOCK 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=BTREE 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=BLOCK 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=ATOMIC 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=KEEPER 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=DENSE 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=MAP 
make -f Makefile CC=icc

set -x
for ns in 30 90
do
	export OMP_PLACES=sockets
	for nth in 1 2 4 8 16 28 56
	do
		export OMP_NUM_THREADS=$nth
		echo "ORIGINAL"
		./lulesh2.0        -s $ns
		echo "LOCK"
		./lulesh2.0_LOCK   -s $ns
		echo "BTREE"
		./lulesh2.0_BTREE  -s $ns
		echo "BLOCK"
		./lulesh2.0_BLOCK  -s $ns
		echo "ATOMIC"
		./lulesh2.0_ATOMIC -s $ns
		echo "KEEPER"
		./lulesh2.0_KEEPER -s $ns
		echo "DENSE"
		./lulesh2.0_DENSE  -s $ns
		echo "MAP"
		./lulesh2.0_MAP    -s $ns
	done
done
