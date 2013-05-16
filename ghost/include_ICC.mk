ifeq ($(MPI),1)
CC = mpiCC
FC = mpiifort
else
CC = icc
FC = ifort
endif

CFLAGS  = -openmp -fPIC -std=c99
FFLAGS  = -fPIC -module $(MODDIR)
SHAREDFLAG = -shared

LIBS = -limf -lm -lrt -lsvml -lirng -lintlc


ifeq ($(AVX),1)
CFLAGS += -mavx
endif

ifeq ($(SSE),1)
CFLAGS += -msse4.2
endif

ifeq ($(MIC),1)
CFLAGS += -mmic
FFLAGS += -mmic
endif
ifeq ($(MIC),2)
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
