# spreduce

Sparse Reduction Experiments

## Building the object files for other projects

    make

## Testing with a simple, built-in test case

    make test

## Building and running an experiment

After running

    make test

navigate to the experiments folder to build and run the experiment, e.g.

    cd experiments/csr
    make test

## Building with Intel compilers

For all make files, append a variable like so:

    make test CC=icc
