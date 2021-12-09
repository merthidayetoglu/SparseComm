#!/bin/bash

date

export FILENAME="/lus/theta-fs0/projects/hp-ptycho/merth/matrix_test.bin"
export NUMRHS=1024

mpirun -n 8 -N 8 --map-by node:PE=8 -x OMP_NUM_THREADS=8 -x OMP_PLACES=cores ./SparseComm

date
