#SYSTEM = tinygpu
#SYSTEM  = woody
#SYSTEM  = lima
include makes/make_$(SYSTEM).mk


################################################################################

VPATH	=	./SRC/ ./OBJ/ ./SRC/lib/
IPATH	+=	-I./SRC/include -I../hpcUtil/src/includes

################################################################################


OBJS	=	$(COBJS) $(FOBJS) $(F90OBJS) $(SOBJS) $(LINREG)

COBJS	 +=   util.o matricks.o mpihelper.o aux.o setup_communication.o \
             timing.o restartfile.o hybrid_kernel_0.o \
             hybrid_kernel_I.o hybrid_kernel_II.o hybrid_kernel_III.o hybrid_kernel_IV.o \
             hybrid_kernel_V.o hybrid_kernel_VI.o hybrid_kernel_VII.o hybrid_kernel_VIII.o \
             hybrid_kernel_IX.o hybrid_kernel_X.o hybrid_kernel_XI.o hybrid_kernel_XII.o \
	           hybrid_kernel_XIII.o hybrid_kernel_XIV.o hybrid_kernel_XV.o hybrid_kernel_0.o \
             hybrid_kernel_XVI.o hybrid_kernel_XVII.o 

CUDAOBJS = cudafun.o my_ellpack.o  

OCLOBJS = oclfun.o my_ellpack.o 

FOBJS	+= 	matricks_GW.o imtql1.o pythag.o 

SOBJS	+=	for_timing_start_asm.o for_timing_stop_asm.o


################################################################################
#####      Consequences:      #####
###################################

cllanczos: MAKROS += -DOCLKERNEL 
clspmvm: MAKROS += -DOCLKERNEL 

################################################################################

#$(error $(OBJECT_FILES))

%_cl.o: %.c  
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.f 
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.F 
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.f90
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.s
	$(AS) $(ASFLAGS) -o $@  $<


all: cllanczos lanczos

cllanczos: main_lanczos.o $(OBJS) $(OCLOBJS)
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv *.o OBJ
	 -mv *genmod* OBJ

lanczos: main_lanczos.o $(OBJS) 
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv *.o OBJ
	 -mv *genmod* OBJ

clspmvm: main_spmvm.o $(OBJS) $(OCLOBJS)
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv *.o OBJ
	 -mv *genmod* OBJ

spmvm: main_spmvm.o $(OBJS) 
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv *.o OBJ
	 -mv *genmod* OBJ

minimal: main_minimal.o $(OBJS) 
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv *.o OBJ
	 -mv *genmod* OBJ
	 

clean:
	-rm -f *.o
	-rm -f OBJ/*.o
	-rm -f core
