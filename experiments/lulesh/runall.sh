REDUCTIONSTYLE=LOCK make -f Makefile_spray
REDUCTIONSTYLE=BTREE make -f Makefile_spray
REDUCTIONSTYLE=BLOCK make -f Makefile_spray
REDUCTIONSTYLE=ATOMIC make -f Makefile_spray
REDUCTIONSTYLE=KEEPER make -f Makefile_spray
REDUCTIONSTYLE=DENSE make -f Makefile_spray
REDUCTIONSTYLE=MAP make -f Makefile_spray

./lulesh2.0_LOCK
./lulesh2.0_BTREE 
./lulesh2.0_BLOCK 
./lulesh2.0_ATOMIC
./lulesh2.0_KEEPER
./lulesh2.0_DENSE 
./lulesh2.0_MAP
