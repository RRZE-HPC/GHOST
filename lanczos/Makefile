OPENCL=0
DOUBLE=1
LIKWID_MARKER=0

CUDA_INC=-I$(CUDA_HOME)/include
CUDA_LIB=-L$(CUDA_HOME)/lib64 -L$(CUDA_DRIVERDIR)/lib
LIKWID_INC=/home/hpc/unrz/unrza317/app/likwid/include
LIKWID_LIB=/home/hpc/unrz/unrza317/app/likwid/lib

LIBSPMVMPATH=/home/hpc/unrz/unrza317/app/libspmvm

LPATH+= -L$(LIBSPMVMPATH)/lib
IPATH+= -I$(LIBSPMVMPATH)/include
LIBS+=-l$(PREFIX)spmvm


.PHONY:clean distclean all

CC=mpicc
CFLAGS=-g -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

FC	= ifort
FFLAGS  =-O3 -fno-alias -cpp -warn -openmp  

ifeq ($(OPENCL),1)
MAKROS+= -DOPENCL
PREFIX= cl
LIBS  += -lOpenCL
IPATH += ${CUDA_INC}
LPATH += ${CUDA_LIB}
endif

ifeq ($(LIKWID_MARKER),1)
MAKROS+= -DLIKWID_MARKER
LPATH += -L${LIKWID_LIB}
IPATH += -I${LIKWID_INC}
LIBS  += -llikwid
endif

ifeq ($(LIKWID_MARKER_FINE),1)
MAKROS+= -DLIKWID_MARKER_FINE
LPATH += -L${LIKWID_LIB}
IPATH += -I${LIKWID_INC}
LIBS  += -llikwid
endif

ifeq ($(DOUBLE),1)
MAKROS+= -DDOUBLE
endif

ifeq ($(COMPLEX),1)
MAKROS+= -DCOMPLEX
endif


%.o: %.c  
	$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 

%.o: %.f 
	$(FC) $(FFLAGS) -o $@ -c $<


lanczos: main_lanczos.o imtql1.o pythag.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $(PREFIX)$@.x $^  $(LIBS)


clean:
	-rm -f *.o
	-rm -f *genmod*

distclean: clean
	-rm -f *.x
