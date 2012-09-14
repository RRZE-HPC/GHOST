OPENCL=0
LIKWID_MARKER=0
LIKWID_MARKER_FINE=0
DOUBLE=1
COMPLEX=1
DEBUG=0

INSTDIR=~/app/libspmvm
CUDA_INC=$(CUDA_HOME)/include
CUDA_LIB=$(CUDA_HOME)/lib64
LIKWID_INC=/home/hpc/unrz/unrza317/app/likwid/include
LIKWID_LIB=/home/hpc/unrz/unrza317/app/likwid/lib


CC	= mpicc
CFLAGS  = -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

FC	= ifort
FFLAGS  = -O3 -nogen-interface -fno-alias -cpp -warn all -openmp 

