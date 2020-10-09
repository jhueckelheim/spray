#!/bin/sh
module use /soft/modulefiles
module load intel/2019

make timings_circuit.csv --always-make CC=icc
