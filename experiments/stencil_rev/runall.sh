#!/bin/sh
module use /soft/modulefiles
module load intel/2019

make --always-make CC=icc timings.csv
