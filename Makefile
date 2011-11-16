SYSTEM = gpu
#SYSTEM  = woody
#SYSTEM  = lima
include makes/make_$(SYSTEM).mk

################################################################################
#####  Decisions to be made: ######
###################################

TIMING = CYCLES_INT
NUMA	= PLACE
#MAKROS += -DINDIVIDUAL
#VERSION = III
#VERSION = V
VERSION = VI
#MAKROS += -DLIKWID
#MAKROS += -DCOMMONLY
#MAKROS += -DKV${VERSION}
#MAKROS += -DFAST_EXIT
MAKROS += -DTUNE_SETUP
#MAKROS	+= -DCMEM
MAKROS += -DREVBUF
MAKROS += -DNLDD
#IO = parallel

################################################################################

VPATH	=	./SRC/ ./OBJ/ ./SRC/lib/
IPATH	+=	-I./SRC/include -I../hpcUtil/src/includes
IPATH	+=	-I./SRC #necessary for vampir-trace

################################################################################

##OBJECT_FILES =  $(patsubst %.c,%.o, $(wildcard *.c))
#OBJECT_FILES := $(OBJECT_FILES) $(patsubst %.F,%.o, $(wildcard *.F))
#OBJECT_FILES := $(OBJECT_FILES) $(patsubst %.f90,%.o, $(wildcard *.f90))

ifeq (${IO},parallel)
MAIN	=	Hybrid_SpMVM_parRead.o
COBJS	+= 	parallel_IO.o #setup_communication_parRead.o
MAKROS	+=	-DPIO
else
MAIN	=	Hybrid_SpMVM.o
endif

OBJS	=	$(COBJS) $(FOBJS) $(F90OBJS) $(SOBJS) $(LINREG)

COBJS	 +=    matricks.o mpihelper.o aux.o setup_communication.o hybrid_kernel_${VERSION}.o check_lcrp.o \
             parallel_IO.o setup_communication_parallel.o Correctness_check.o restartfile.o hybrid_kernel_0.o \
             hybrid_kernel_I.o hybrid_kernel_II.o hybrid_kernel_III.o hybrid_kernel_IV.o \
             hybrid_kernel_V.o hybrid_kernel_VI.o hybrid_kernel_VII.o hybrid_kernel_VIII.o \
             hybrid_kernel_IX.o hybrid_kernel_X.o hybrid_kernel_XI.o hybrid_kernel_XII.o \
	           hybrid_kernel_XIII.o hybrid_kernel_XIV.o hybrid_kernel_XV.o hybrid_kernel_0.o \
             hybrid_kernel_XVI.o hybrid_kernel_XVII.o 

CUDAOBJS = cudafun.o my_ellpack.o 

COBJS	+=#	timing.o matricks.o aux.o resort_JDS.o check_divide.o restartfile.o \
		parallel_IO.o #myODJDS.o # test_comp.o #Offdiagonal_JDS.o

F90OBJS	+=	#actcrs.o do_stats.o analyse_mat.o myblockJDS_resorted.o #test_OMP.o 

FOBJS	+= 	matricks_GW.o 

SOBJS	+=	for_timing_start_asm.o for_timing_stop_asm.o# \
		#myJDS_pure_asm.o myblockjds_asm.o

LINREG	=	#g02caft.o p01abft.o x04baft.o x04aaft.o x02alft.o x02akft.o p01abzt.o

################################################################################
#####      Consequences:      #####
###################################

ifeq (${CUDASWITCH},DOCUDA)
	OBJS += ${CUDAOBJS}
	MAKROS += -DCUDAKERNEL #-DTEXCACHE
endif

ifeq (${NUMA},PLACE)
	MAKROS 	+= -DPLACE
endif

ifeq (${TIMING},CYCLES_INT)
	MAKROS	+=	-DCYCLES_INT
	COBJS	+=	timing.o
#	SOBJS	+=      sp_mvm_timing_asm.o get_timing_overhead_asm.o
else
	MAKROS	+=	-DCYCLES_EXT
#	SOBJS	+=	sp_mvm_asm.o
endif


################################################################################

#$(error $(OBJECT_FILES))

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

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

HybridSpMVM: $(MAIN) $(OBJS) 
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv *.o OBJ
	 -mv *genmod* OBJ

mpitest : mpitest.o matricks.o aux.o timing.o $(SOBJS) $(CUDAOBJS)
	$(CC) $(LDFLAGS) -o $@$(SUFX).x $^  $(LIBS)
	 -mv *.o OBJ

clean:
	-rm -f *.o
	-rm -f OBJ/*.o
	-rm -f core
	-rm -f *.x
