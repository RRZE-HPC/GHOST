include config.mk

.PHONY: clean distclean install libspmvm

VPATH	=	./src/
IPATH	+=	-I./src/include

OBJS	=	$(COBJS) $(FOBJS)

ifeq ($(OPENCL),1)
OBJS	+=	$(OCLOBJS)
endif

COBJS	=  aux.o spmvm_util.o spmvm.o matricks.o mpihelper.o  \
		   timing.o mmio.o  hybrid_kernel_0.o hybrid_kernel_I.o \
		   hybrid_kernel_II.o hybrid_kernel_III.o spmvm_globals.o
OCLOBJS =  spmvm_cl_util.o my_ellpack.o 
FOBJS	=  matricks_GW.o  

LIBSPMVM=lib$(PREFIX)spmvm.a

%.o: %.c  
	$(CC) $(CFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

%.o: %.f 
	$(FC) $(FFLAGS) -o $@ -c $<

libspmvm: $(OBJS)
	@ar rcs  $(LIBSPMVM) $^
	@mkdir -p obj/
	@mv *.o obj/
	@mv *genmod* obj/

clean:
	@rm -rf obj/

distclean: clean
	@rm -f *.a

install: libspmvm
	@mkdir -p $(INSTDIR)/lib
	@mkdir -p $(INSTDIR)/include
	@cp -f $(LIBSPMVM) $(INSTDIR)/lib
	@cp -f src/include/spmvm*.h $(INSTDIR)/include

uninstall: 
	@rm -f $(INSTDIR)/lib/lib*spmvm.a
	@rm -f $(INSTDIR)/include/spmvm*.h 

	
