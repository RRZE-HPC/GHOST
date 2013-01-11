ifeq ($(strip $(MPI)),1)
CC = mpicc
else
CC = icc
endif
FC  = ifort

CFLAGS  = -openmp -fPIC -std=c99 
FFLAGS  = -openmp -fPIC -nogen-interface -cpp

ifeq ($(strip $(AVX)),1)
CFLAGS += -mavx
FFLAGS += -mavx
endif

ifeq ($(strip $(SSE)),1)
CFLAGS += -msse4.2
FFLAGS += -msse4.2
endif

ifeq ($(strip $(MIC)),1)
CFLAGS += -mmic
FFLAGS += -mmic
endif

ifneq ($(strip $(DEBUG)),0)
CFLAGS += -g -O0
FFLAGS += -g -O0
else
CFLAGS += -O3 -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981
FFLAGS += -O3 -warn all
endif
