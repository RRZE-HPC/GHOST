#include ../config.mk

.PHONY:clean distclean all

CC=mpicc
CFLAGS=-O3 -fno-alias -openmp -fPIC -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 
CUDA_INC=$(CUDA_HOME)/include
LIKWID_INC=/home/hpc/unrz/unrza317/app/likwid/include
DOUBLE=1
OPENCL=1
LIKWID=1
LIBSPMVMPATH=/home/hpc/unrz/unrza317/app/libspmvm
LPATH+= -L$(LIBSPMVMPATH)/lib
IPATH+= -I$(LIBSPMVMPATH)/include
LIBS+=-l$(PREFIX)spmvm

ifeq ($(OPENCL),1)
MAKROS+= -DOPENCL
PREFIX= cl
IPATH += -I${CUDA_INC}
LIBS  += -lOpenCL
endif

ifeq ($(LIKWID),1)
MAKROS+= -DLIKWID
IPATH += -I${LIKWID_INC}
LPATH += -L${LIKWID_LIB}
LIBS  += -llikwid
endif

ifeq ($(LIKWID_MARKER),1)
MAKROS+= -DLIKWID_MARKER
endif

ifeq ($(LIKWID_MARKER_FINE),1)
MAKROS+= -DLIKWID_MARKER_FINE
endif

ifeq ($(DOUBLE),1)
MAKROS+= -DDOUBLE
endif

ifeq ($(COMPLEX),1)
MAKROS+= -DCOMPLEX
endif


%.o: %.c  
	$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 

all: spmvm lanczos minimal 

spmvm: spmvm/main_spmvm.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o spmvm/$(PREFIX)$@.x $^  $(LIBS)

lanczos: lanczos/main_lanczos.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o lanczos/$(PREFIX)$@.x $^  $(LIBS)

minimal: minimal/main_minimal.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o minimal/$(PREFIX)$@.x $^  $(LIBS)

clean:
	-rm -f */*.o

distclean: clean
	-rm -f */*.x
