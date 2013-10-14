ifeq ($(MPI),1)
CC = mpicc
CPPC = mpiCC
FC = mpif90
else
CC = gcc
CPPC = g++
FC = gfortran
endif
PP=cpp

CFLAGS = -fopenmp -fPIC -std=c99 -DMKL_BLAS
FFLAGS = -ffixed-line-length-none -ffree-line-length-none -DMKL_BLAS
CPPFLAGS = -fopenmp -fPIC -DMKL_BLAS
SHAREDFLAG = -shared
FMODFLAG = -J
LIBS = -ldl -lm -lmkl_core -lmkl_gf_lp64 -lmkl_gnu_thread -lstdc++ -lpthread #openmpi: -lmpi_cxx


ifneq ($(strip $(DEBUG)),0)
CFLAGS += -g -O0
CPPFLAGS += -g -O0
FFLAGS += -g -O0
else
CFLAGS  += -O3 -Wall # -Werror 
CPPFLAGS  += -O3 -Wall # -Werror 
FFLAGS  += -O3 -Wall # -Werror
endif
