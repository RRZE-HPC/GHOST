ifeq ($(strip $(MPI)),1)
CC = mpicc
else
CC = icc
endif
FC  = ifort

CFLAGS  = -openmp -fPIC -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981
FFLAGS  = -openmp -fPIC -nogen-interface -cpp -warn all

ifeq ($(strip $(MIC)),1)
CFLAGS += -mmic
FFLAGS += -mmic
endif

ifneq ($(strip $(DEBUG)),0)
CFLAGS += -g -O0
FFLAGS += -g -O0
else
CFLAGS += -O3
FFLAGS += -O3
endif
