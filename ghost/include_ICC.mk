ifeq ($(MPI),1)
CC = mpicc
CPPC = mpiicpc
FC = mpiifort
else
CC = icc
CPPC = icpc
FC = ifort
endif

CFLAGS  = -openmp -fPIC -std=c99
FFLAGS  = -fPIC -module $(MODDIR)
SHAREDFLAG = -shared

LIBS = -limf -lm -lrt -lsvml -lintlc -lpthread # -lirng 

ifneq ($(AVX),0)
CFLAGS += -mavx
FFLAGS += -mavx
endif

ifneq ($(SSE),0)
CFLAGS += -msse4.2
FFLAGS += -msse4.2
endif

ifneq ($(MIC),0)
CFLAGS += -mmic
FFLAGS += -mmic
endif

ifneq ($(DEBUG),0)
CFLAGS += -g -O0 #-check-pointers=rw -check-pointers-dangling=all -rdynamic
FFLAGS += -g -O0
else
CFLAGS += -O3 -Wall -Wremarks -Wcheck -diag-disable 981 #-vec-report3
FFLAGS += -O3 -warn all
endif
