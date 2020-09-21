# SPRAY

Sparse Reductions for ArraYs

## Usage

Consider this loop that performs updates to an output array at data-dependent indices:

    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }

Conceptually, this loop performs a sum-reduction to the elements in `x`, so we might attempt to parallelize this using OpenMP reductions:

    #pragma omp parallel for reduction(+:x[0:csr_data.nc])
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }

However, the OpenMP standard requires the compiler to implement this in an inefficient way, by initializing a copy of the _entire_ array on each thread, despite the fact that each thread might only update a small part of the output array. All these private copies must be initialized and eventually added together, resulting in wasted memory space and bandwidth. Worse, still, is that the amount of wasted memory actually _increases_ with the number of threads, and running the above program on a system with more cores will actually slow you down.

SPRAY is a header-only library that provides more appropriate strategies for reductions on large arrays, particularly in cases where each thread only updates a portion of the array. SPRAY allows threads to update overlapping indices of the output array at the same time, without creating complete private copies of the entire array. Instead, SPRAY uses a mix of small privatized data buffers, locks, and atomic updates to ensure that all updates to the output are eventually committed to the result.

To use it, all you have to do is to include the `spray.hpp` header file, pass your output array to a SPRAY constructor, and use the OpenMP reduction clause on it, like so:

    spray::AtomicReduction<real> x_p(x);
    #pragma omp parallel for reduction(+:x_p)
    for (int i = 0; i < csr_data.nr; i++) {
        for (int k = csr_data.rptr[i]; k < csr_data.rptr[i+1]; k++) {
            x_p[csr_data.cols[k]] += csr_data.vals[k]*res[i];
        }
    }

SPRAY provides a number of implementation strategies, all of which are correct in all cases, but may have different performance characteristics depending on the use case, and the hardware on which the program is executed. It is a good idea to try a few options to see which one works best. This can be done by replacing the `AtomicReduction` constructor above with one of the alternatives, e.g. `DenseReduction`, `BlockReduction`, or `BtreeReduction`. A given program can mix and match multiple strategies as needed.

## Building

SPRAY is a header-only library, there is nothing you need to build to get started.

## Testing with a simple, built-in test case

SPRAY comes with a few tests to make sure everything works as intended, build and run those with

    make test

If you prefer using Intel compilers, append a variable to the `make` command like so:

    make test CC=icc
