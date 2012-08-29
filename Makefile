include config.mk

.PHONY: examples utils clean distclean all install 

VPATH	=	./src/
IPATH	+=	-I./src/include

OBJS	=	$(COBJS) $(FOBJS)

COBJS	=  aux.o spmvm_util.o spmvm.o matricks.o mpihelper.o  \
		   timing.o mmio.o  hybrid_kernel_0.o hybrid_kernel_I.o \
		   hybrid_kernel_II.o hybrid_kernel_III.o spmvm_globals.o
OCLOBJS =  spmvm_cl_util.o my_ellpack.o 
FOBJS	=  matricks_GW.o imtql1.o pythag.o 

LIBSPMVM=lib$(PREFIX)spmvm.a

%.o: %.c  
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.f 
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.s
	$(AS) $(ASFLAGS) -o $@  $<


all: $(LIBSPMVM) examples utils

examples: $(LIBSPMVM)
	$(MAKE) -C examples/  

utils: 
	$(MAKE) -C utils/


libspmvm.so: $(OBJS)
	$(CC) -shared -o $@ $^ $(LPATH) $(LIBS)

libclspmvm.so: $(OBJS) $(OCLOBJS)
	$(CC) -shared -o $@ $^ $(LPATH) $(LIBS)

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

install: $(LIBSPMVM)
	@mkdir -p $(INSTDIR)/lib
	@mkdir -p $(INSTDIR)/include
	@cp -f $(LIBSPMVM) $(INSTDIR)/lib
	@cp -f src/include/spmvm*.h $(INSTDIR)/include


	
