#!/bin/bash

date

export FILENAME="/lus/theta-fs0/projects/hp-ptycho/merth/matrix_125gb.bin"
export NUMRHS=1024

mpirun -n 16 -N 16 --map-by node:PE=4 -x OMP_NUM_THREADS=4 -x OMP_PLACES=cores ./SparseComm

date
