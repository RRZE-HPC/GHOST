CC=mpicc
CFLAGS=-O3 -openmp $(MAKROS) $(IPATH)
IPATH=$(LIKWID_INC) -I../SRC/include 
LPATH=$(LIKWID_LIB) -L..
LIBS=-l$(PREFIX)spmvm -llikwid

ifdef OPENCL
MAKROS+= -DOPENCL
PREFIX+= cl
IPATH += -I${CUDA_INSTALL_PATH}/include
LPATH += -L${CUDA_INSTALL_PATH}/lib64 -L${CUDA_DRIVERDIR}/lib
LIBS  += -lOpenCL
endif


%.o: %.c  
	$(CC) $(CFLAGS) -o $@ -c $< 

all: spmvm lanczos 


spmvm: spmvm/main_spmvm.o
	$(CC) $(CFLAGS) $(LPATH) -o spmvm/$(PREFIX)$@.x $^  $(LIBS)

lanczos: lanczos/main_lanczos.o
	$(CC) $(CFLAGS) $(LPATH) -o lanczos/$(PREFIX)$@.x $^  $(LIBS)

minimal: minimal/main_minimal.o
	$(CC) $(CFLAGS) $(LPATH) -o minimal/$(PREFIX)$@.x $^  $(LIBS)

