SYSTEM=default
include makes/make_$(SYSTEM).mk

.PHONY: examples utils clean distclean all 

################################################################################

VPATH	=	./src/ ./obj/ ./src/lib/
IPATH	+=	-I./src/include

################################################################################


OBJS	=	$(COBJS) $(FOBJS) $(F90OBJS) $(SOBJS) 

COBJS	=  aux.o spmvm_util.o matricks.o mpihelper.o  \
		   timing.o mmio.o  hybrid_kernel_0.o hybrid_kernel_I.o \
		   hybrid_kernel_II.o hybrid_kernel_III.o spmvm_globals.o
OCLOBJS = oclfun.o my_ellpack.o 
FOBJS	= 	matricks_GW.o imtql1.o pythag.o 
SOBJS	=	for_timing_start_asm.o for_timing_stop_asm.o

MAKROS=#-DPIN

ifdef LIKWID
MAKROS+= -DLIKWID
endif

ifdef OPENCL
MAKROS+= -DOPENCL 
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
	-mv *.o obj
	-mv *genmod* obj

libclspmvm.a: $(OBJS) $(OCLOBJS)
	ar rcs  $@ $^ 
	-mv *.o obj
	-mv *genmod* obj

clean:
	-rm -f obj/*
	$(MAKE) -C examples clean
	$(MAKE) -C utils clean

distclean: clean
	-rm -f *.so
	-rm -f *.a
	$(MAKE) -C examples distclean
	$(MAKE) -C utils distclean
