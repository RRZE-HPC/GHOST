ifeq ($(MPI),1)
CC = mpicc
CPPC = mpiCC
FC = mpif90
CFLAGS += -mt_mpi
CPPFLAGS += -mt_mpi
FFLAGS += -mt_mpi
else
CC = icc
CPPC = icpc
FC = ifort
endif
PP=cpp

CFLAGS  += -fPIC -std=c99 -DMKL_BLAS 
CPPFLAGS  += -fPIC -std=c99 -DMKL_BLAS
FFLAGS  += -fPIC
FMODFLAG = -module
SHAREDFLAG = -shared

LIBS = -limf -lm -lrt -lsvml -lintlc -lmkl_intel_lp64 -lmkl_core -lpthread -lifcore # -lirng 

ifeq ($(OPENMP),1)
CFLAGS += -openmp
CPPFLAGS += -openmp
FFLAGS += -openmp
LIBS += -lmkl_intel_thread
else 
LIBS += -lmkl_sequential
endif

ifneq ($(AVX),0)
CFLAGS += -mavx
CPPFLAGS += -mavx
FFLAGS += -mavx
endif

ifneq ($(SSE),0)
CFLAGS += -msse4.2
CPPFLAGS += -msse4.2
FFLAGS += -msse4.2
endif

ifneq ($(MIC),0)
CFLAGS += -mmic
CPPFLAGS += -mmic
FFLAGS += -mmic
endif

ifneq ($(DEBUG),0)
CFLAGS += -g -O0 #-check-pointers=rw -check-pointers-dangling=all -rdynamic
CPPFLAGS += -g -O0 #-check-pointers=rw -check-pointers-dangling=all -rdynamic
FFLAGS += -g -O0 
else
CFLAGS += -O3 -Wall -Wremarks -Wcheck -diag-disable 981 #-vec-report3
CPPFLAGS += -O3 -Wall -Wremarks -Wcheck -diag-disable 981 #-vec-report3
FFLAGS += -O3 -warn all
endif
