INTEL_F_HOME = /usr/intel-ifort-10.0

CC	= mpicc.openmpi-icc ${CFLAGS}
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp #-openmp -DWRITE_RESTART #-parallel

FC	= mpif90.openmpi-icc
FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all #-vec-report3 -opt-report -unroll0 #-openmp #-C -debug extended -traceback #-parallel


LDFLAGS = -g -O3 ${RFLAGS} -openmp #-static #-parallel

LIBS = -L$(INTEL_F_HOME)/lib -lifcore

RFLAGS	+= -Xlinker -rpath -Xlinker ${INTEL_F_HOME}/lib
 
AS	= as
ASFLAGS = -g -gstabs

ENDG    =       $(shell echo ${MAKROS} | sed -e 's/\-D/_/g' | sed -e 's/\ //g')
SUFX    =      ${ENDG}_${SYSTEM}
