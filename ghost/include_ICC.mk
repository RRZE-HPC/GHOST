ifeq ($(MPI),1)
CC = mpicc
FC = mpiifort
else
CC = icc
FC = ifort
endif

CFLAGS  = -openmp -fPIC -std=c99
FFLAGS  = -fPIC -module $(MODDIR)
SHAREDFLAG = -shared

LIBS = -limf -lm


ifeq ($(AVX),1)
CFLAGS += -mavx
endif

ifeq ($(SSE),1)
CFLAGS += -msse4.2
endif

ifeq ($(MIC),1)
CFLAGS += -mmic
endif

ifneq ($(DEBUG),0)
CFLAGS += -g -O0 #-check-pointers=rw -check-pointers-dangling=all -rdynamic
FFLAGS += -g -O0
else
CFLAGS += -O3 -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981
FFLAGS += -O3 -warn all
endif
