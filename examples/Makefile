include ../config.mk
include ../makes/make_$(SYSTEM).mk

.PHONY:clean distclean all

#CFLAGS=-g -O3 -openmp -mt_mpi $(MAKROS) $(IPATH)
IPATH+= -I../src/include 
LPATH+= -L..
MYLIBS=-l$(PREFIX)spmvm $(LIBS)

#ifdef OPENCL
#MAKROS+= -DOPENCL
#PREFIX+= cl
#IPATH += -I${CUDA_INSTALL_PATH}/include
#LPATH += -L${CUDA_INSTALL_PATH}/lib64 -L${CUDA_DRIVERDIR}/lib
#LIBS  += -lOpenCL
#endif

#ifdef LIKWID
#IPATH += $(LIKWID_INC)
#LPATH += $(LIKWID_LIB)
#LIBS  += -llikwid
#endif

%.o: %.c  
	$(CC) $(CFLAGS) -o $@ -c $< 

all: spmvm lanczos minimal 


spmvm: spmvm/main_spmvm.o
	$(CC) $(CFLAGS) $(LPATH) -o spmvm/$(PREFIX)$@.x $^  $(MYLIBS)

lanczos: lanczos/main_lanczos.o
	$(CC) $(CFLAGS) $(LPATH) -o lanczos/$(PREFIX)$@.x $^  $(MYLIBS)

minimal: minimal/main_minimal.o
	$(CC) $(CFLAGS) $(LPATH) -o minimal/$(PREFIX)$@.x $^  $(MYLIBS)

clean:
	-rm -f */*.o

distclean: clean
	-rm -f */*.x
