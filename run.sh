#!/bin/bash

date

export FILENAME="/lus/theta-fs0/projects/hp-ptycho/merth/matrix_test.bin"
export NUMRHS=32

mpirun -n 16 -N 16 --map-by node:PE=1 -x OMP_NUM_THREADS=1 -x OMP_PLACES=cores ./SparseComm

date
