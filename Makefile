SYSTEM=default
include makes/make_$(SYSTEM).mk

.PHONY: examples utils clean distclean 

################################################################################

VPATH	=	./SRC/ ./OBJ/ ./SRC/lib/
IPATH	+=	-I./SRC/include

################################################################################


OBJS	=	$(COBJS) $(FOBJS) $(F90OBJS) $(SOBJS) 

COBJS	=  aux.o spmvm_util.o matricks.o mpihelper.o setup_communication.o timing.o \
		   mmio.o  hybrid_kernel_0.o hybrid_kernel_I.o hybrid_kernel_II.o hybrid_kernel_III.o
OCLOBJS = oclfun.o my_ellpack.o 
FOBJS	= 	matricks_GW.o imtql1.o pythag.o 
SOBJS	=	for_timing_start_asm.o for_timing_stop_asm.o


ifdef OPENCL
MAKROS=-DOPENCL 
PREFIX=cl
else
PREFIX=
endif

LIBSPMVM=lib$(PREFIX)spmvm.a


%.o: %.c  
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.f 
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.F 
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.f90
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.s
	$(AS) $(ASFLAGS) -o $@  $<


all: $(LIBSPMVM) examples utils

examples: $(LIBSPMVM)
	$(MAKE) -C examples/ 

utils: 
	$(MAKE) -C utils/


libspmvm.so: $(OBJS)
	$(CC) $(LDFLAGS) -shared -o $@ $^ $(LPATH) $(LIBS)

libclspmvm.so: $(OBJS) $(OCLOBJS)
	$(CC) $(LDFLAGS) -shared -o $@ $^ $(LPATH) $(LIBS)

libspmvm.a: $(OBJS)
	ar rcs  $@ $^
	-mv *.o OBJ
	-mv *genmod* OBJ

libclspmvm.a: $(OBJS) $(OCLOBJS)
	ar rcs  $@ $^ 
	-mv *.o OBJ
	-mv *genmod* OBJ

clean:
	-rm -f *.o
	-rm -f OBJ/*
	-rm -f core
	-rm -f examples/*/*.o
	-rm -f utils/*.o

distclean: clean
	-rm -f *.x
	-rm -f examples/*/*.x
	-rm -f utils/*.x
	-rm -f *.so
	-rm -f *.a
