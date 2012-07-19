#SYSTEM = tinygpu
#SYSTEM  = woody
#SYSTEM  = lima
include makes/make_$(SYSTEM).mk


################################################################################

VPATH	=	./SRC/ ./OBJ/ ./SRC/lib/
IPATH	+=	-I./SRC/include

################################################################################


OBJS	=	$(COBJS) $(FOBJS) $(F90OBJS) $(SOBJS) $(LINREG)

COBJS	 +=  aux.o spmvm_util.o matricks.o mpihelper.o setup_communication.o \
             timing.o restartfile.o \
			 hybrid_kernel_V.o hybrid_kernel_X.o hybrid_kernel_XII.o
# hybrid_kernel_0.o \
#            hybrid_kernel_I.o hybrid_kernel_II.o hybrid_kernel_III.o hybrid_kernel_IV.o \
#            hybrid_kernel_V.o hybrid_kernel_VI.o hybrid_kernel_VII.o hybrid_kernel_VIII.o \
#            hybrid_kernel_IX.o hybrid_kernel_X.o hybrid_kernel_XI.o hybrid_kernel_XII.o \
#	           hybrid_kernel_XIII.o hybrid_kernel_XIV.o hybrid_kernel_XV.o hybrid_kernel_0.o \
#            hybrid_kernel_XVI.o hybrid_kernel_XVII.o 

OCLOBJS = oclfun.o my_ellpack.o 

FOBJS	+= 	matricks_GW.o imtql1.o pythag.o 

SOBJS	+=	for_timing_start_asm.o for_timing_stop_asm.o


################################################################################
#####      Consequences:      #####
###################################

clspmvm-static: MAKROS = -DOPENCL
cllanczos: MAKROS = -DOPENCL 
cllanczos-static: MAKROS = -DOPENCL 
cllanczos-dynamic: MAKROS = -DOPENCL 
clspmvm: MAKROS = -DOPENCL 
libclspmvm.a: MAKROS = -DOPENCL 
libclspmvm.so: MAKROS = -DOPENCL 

################################################################################


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


all: cllanczos lanczos

spmvm-static: main_spmvm.o libspmvm.a
	$(CC) $(CFLAGS) -o $@$(SUFX).x $^  $(LIBS)

clspmvm-static: main_spmvm.o libclspmvm.a
	$(CC) $(CFLAGS) -o $@$(SUFX).x $^  $(LIBS)

lanczos-static: main_lanczos.o libspmvm.a
	$(CC) $(CFLAGS) -o $@$(SUFX).x $^  $(LIBS)

cllanczos-static: main_lanczos.o libclspmvm.a
	$(CC) $(CFLAGS) -o $@$(SUFX).x $^  $(LIBS)

cllanczos-dynamic: main_lanczos.o libclspmvm.so
	$(CC) -O3 -DOPENCL -I./SRC/include $(LIKWID_INC) -L. $(LIKWID_LIB) -pthread -lclspmvm -llikwid -openmp -o $@$(SUFX).x $<

lanczos-dynamic: main_lanczos.o libspmvm.so
	$(CC) -O3 -I./SRC/include -L. -lspmvm -openmp -o $@$(SUFX).x $<

cllanczos: main_lanczos.o $(OBJS) $(OCLOBJS)
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv *.o OBJ
	 -mv *genmod* OBJ

lanczos: main_lanczos.o $(OBJS) 
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv -f *.o OBJ
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

libspmvm.so: $(OBJS)
	$(CC) $(LDFLAGS) -shared -o $@ $^ $(LIBS)

libclspmvm.so: $(OBJS) $(OCLOBJS)
	$(CC) $(LDFLAGS) -shared -o $@ $^ $(LIBS)

libspmvm.a: $(OBJS)
	ar rcs  $@ $^ 

libclspmvm.a: $(OBJS) $(OCLOBJS)
	ar rcs  $@ $^ 

clean:
	-rm -f *.o
	-rm -f OBJ/*.o
	-rm -f core
