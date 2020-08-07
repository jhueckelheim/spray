# spreduce

SPRAY - Sparse Reductions for ArraYs

## Building the object files for other projects

    make

## Testing with a simple, built-in test case

    make test

## Building and running an experiment

After running

    make test

and getting the same checksum for all tests, you may run some of the larger experiments, for example:

    cd experiments/csr; make test

## Building with Intel compilers

When making, append a variable like so:

    make test CC=icc
