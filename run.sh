#!/bin/bash

date

export FILENAME="/lus/theta-fs0/projects/hp-ptycho/merth/matrix_test.bin"
export NUMRHS=10
export NUMITER=3

export PERRACK
export PERNODE=1
export PERSOCK

mpirun -n 16 -N 16 --map-by node:PE=1 -x OMP_NUM_THREADS=1 -x OMP_PLACES=cores ./SparseComm

date
