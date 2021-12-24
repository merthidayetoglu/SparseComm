#!/bin/bash

date

export FILENAME="../matrix.bin"
export NUMRHS=7
export NUMITER=1

export PERNODE=6
export PERSOCKET=3

export OMP_NUM_THREADS=1
mpirun -n 12 valgrind ./SparseComm

date
