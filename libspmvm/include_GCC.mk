ifeq ($(strip $(MPI)),1)
CC = mpicc
else
CC = gcc
endif

FC = gfortran

CFLAGS  = -O3 -fopenmp -Wall -Werror 
FFLAGS  = -O3 -fopenmp -Wall -Werror 
