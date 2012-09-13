OPENCL=0
DOUBLE=1

CUDA_INC=-I$(CUDA_HOME)/include
CUDA_LIB=-L$(CUDA_HOME)/lib64 -L$(CUDA_DRIVERDIR)/lib

LIBSPMVMPATH=/home/hpc/unrz/unrza317/app/libspmvm

LPATH+= -L$(LIBSPMVMPATH)/lib
IPATH+= -I$(LIBSPMVMPATH)/include
LIBS+=-l$(PREFIX)spmvm

.PHONY:clean distclean all

CC=mpicc
CFLAGS=-g -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

ifeq ($(OPENCL),1)
MAKROS+= -DOPENCL
PREFIX= cl
LIBS  += -lOpenCL
IPATH += ${CUDA_INC}
LPATH += ${CUDA_LIB}
endif

ifeq ($(DOUBLE),1)
MAKROS+= -DDOUBLE
endif

ifeq ($(COMPLEX),1)
MAKROS+= -DCOMPLEX
endif


%.o: %.c  
	$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 


spmvm: main_spmvm.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $(PREFIX)$@.x $^  $(LIBS)

clean:
	-rm -f *.o
	-rm -f *genmod*

distclean: clean
	-rm -f *.x
