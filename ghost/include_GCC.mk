ifeq ($(strip $(MPI)),1)
CC = mpicc
else
CC = gcc 
endif

CFLAGS = -fopenmp -fPIC -std=c99
LIBS = -ldl -lm

ifneq ($(strip $(DEBUG)),0)
CFLAGS += -g -O0
FFLAGS += -g -O0
else
CFLAGS  += -O3 -Wall # -Werror 
FFLAGS  += -O3 -Wall # -Werror
endif
