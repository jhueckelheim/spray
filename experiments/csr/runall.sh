#!/bin/sh
module use /soft/modulefiles
module load intel/2019

make timings.csv --always-make CC=icc
