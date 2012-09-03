DOUBLE=1
OPENCL=1
LIKWID_MARKER=0

CUDA_INC=$(CUDA_HOME)/include
LIKWID_INC=/home/hpc/unrz/unrza317/app/likwid/include
LIKWID_LIB=/home/hpc/unrz/unrza317/app/likwid/lib
LIBSPMVMPATH=/home/hpc/unrz/unrza317/app/libspmvm
LPATH+= -L$(LIBSPMVMPATH)/lib
IPATH+= -I$(LIBSPMVMPATH)/include
LIBS+=-l$(PREFIX)spmvm


.PHONY:clean distclean all

CC=mpicc
CFLAGS=-O3 -fno-alias -openmp -fPIC -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

FC	= mpif90
FFLAGS  = -g -O3 -fno-alias -cpp -warn all -openmp -fPIC

ifeq ($(OPENCL),1)
MAKROS+= -DOPENCL
PREFIX= cl
LIBS  += -lOpenCL
IPATH += -I${CUDA_INC}
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

all: spmvm lanczos minimal 

spmvm: spmvm/main_spmvm.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o spmvm/$(PREFIX)$@.x $^  $(LIBS)

lanczos: lanczos/main_lanczos.o lanczos/imtql1.o lanczos/pythag.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o lanczos/$(PREFIX)$@.x $^  $(LIBS)

minimal: minimal/main_minimal.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o minimal/$(PREFIX)$@.x $^  $(LIBS)

clean:
	-rm -f */*.o

distclean: clean
	-rm -f */*.x
