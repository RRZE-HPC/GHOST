ifeq ($(strip $(MPI)),1)
CC = mpicc
else
CC = gcc -ldl -lm
endif

CFLAGS  = -fopenmp -fPIC -std=c99 
FC = gfortran

ifneq ($(strip $(DEBUG)),0)
CFLAGS += -g -O0
FFLAGS += -g -O0
else
CFLAGS  += -O3 -Wall -Werror 
FFLAGS  += -O3 -Wall -Werror 
endif
