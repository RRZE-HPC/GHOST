OPENCL=1
WDDT=0

CUDA_INC=-I$(CUDA_HOME)/include
CUDA_LIB=-L$(CUDA_HOME)/lib64 -L$(CUDA_DRIVERDIR)/lib

LIBSPMVMPATH=/home/hpc/unrz/unrza317/app/libspmvm

LPATH+= -L$(LIBSPMVMPATH)/lib
IPATH+= -I$(LIBSPMVMPATH)/include
LIBS+=-l$(PREFIX)spmvm

.PHONY:clean distclean all

CC=mpicc
CFLAGS=-g -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

ifeq ($(WDDT),1)
LIBS+=$(DDT)/lib/64/libdmallocth.a
endif


ifeq ($(OPENCL),1)
MAKROS+= -DOPENCL
PREFIX= cl
LIBS  += -lOpenCL
IPATH += ${CUDA_INC}
LPATH += ${CUDA_LIB}
endif

%.o: %.c  
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 


$(PREFIX)spmvm.x: main_spmvm.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $@ $^  $(LIBS) 

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

