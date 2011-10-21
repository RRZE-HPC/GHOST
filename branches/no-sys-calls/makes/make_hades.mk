#INTEL_F_HOME = /usr/intel-ifort-10.0
INTEL_F_HOME = /opt/intel_fc_80/

ccomp	 =  /opt/mpich-1.2.6-intel80_shmem/bin/mpicc

CC	= ${ccomp} ${CFLAGS}
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias #-openmp# -DWRITE_RESTART #-parallel
#CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp-stubs -openmp# -DWRITE_RESTART #-parallel

FC	= ifort
FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all #-openmp #-vec-report3 -opt-report -unroll0 #-openmp #-C -debug extended -traceback #-parallel

LDFLAGS = -g -O3 -static #-openmp #-openmp -static #-parallel

LIBS = -L/opt/intel_cc_80/lib -L$(INTEL_F_HOME)/lib -lifcore -openmp -lpthread -lguide
IPATH	+=	-I/$(INTEL_F_HOME)/include
 
AS	= as
ASFLAGS = -g -gstabs

ENDG    =       $(shell echo ${MAKROS} | sed -e 's/\-D/_/g' | sed -e 's/\ //g')
SUFX    =      ${ENDG}_${SYSTEM}
