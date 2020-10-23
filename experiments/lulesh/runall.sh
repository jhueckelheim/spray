#!/bin/sh
module use /soft/modulefiles
module load intel/2019

make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=LOCK 
#make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=BTREE 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=BLOCK 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=ATOMIC 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=KEEPER 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=DENSE 
#make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=MAP 
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=AWLOCK16
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=AWLOCK64
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=AWLOCK256
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=AWLOCK1024
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=AWLOCK4096
make -f Makefile_spray --always-make CC=icc REDUCTIONSTYLE=AWLOCK16384
make -f Makefile CC=icc

set -x
for ns in 90
do
	export OMP_PLACES=sockets
	for nth in 1 2 4 8 16 28 56
	do
		export OMP_NUM_THREADS=$nth
		echo "ORIGINAL"
		/usr/bin/time -v ./lulesh2.0        -s $ns -i 100
		echo "LOCK"
		/usr/bin/time -v ./lulesh2.0_LOCK   -s $ns -i 100
#		echo "BTREE"
#		./lulesh2.0_BTREE  -s $ns -i 100
		echo "BLOCK"
		/usr/bin/time -v ./lulesh2.0_BLOCK  -s $ns -i 100
		echo "ATOMIC"
		/usr/bin/time -v ./lulesh2.0_ATOMIC -s $ns -i 100
		echo "KEEPER"
		/usr/bin/time -v ./lulesh2.0_KEEPER -s $ns -i 100
		echo "DENSE"
		/usr/bin/time -v ./lulesh2.0_DENSE  -s $ns -i 100
#		echo "MAP"
#		./lulesh2.0_MAP    -s $ns -i 100
		echo "AWLOCK16"
		/usr/bin/time -v ./lulesh2.0_AWLOCK16  -s $ns -i 100
		echo "AWLOCK64"
		/usr/bin/time -v ./lulesh2.0_AWLOCK64  -s $ns -i 100
		echo "AWLOCK256"
		/usr/bin/time -v ./lulesh2.0_AWLOCK256  -s $ns -i 100
		echo "AWLOCK1024"
		/usr/bin/time -v ./lulesh2.0_AWLOCK1024  -s $ns -i 100
		echo "AWLOCK4096"
		/usr/bin/time -v ./lulesh2.0_AWLOCK4096  -s $ns -i 100
		echo "AWLOCK16384"
		/usr/bin/time -v ./lulesh2.0_AWLOCK16384  -s $ns -i 100
	done
done
