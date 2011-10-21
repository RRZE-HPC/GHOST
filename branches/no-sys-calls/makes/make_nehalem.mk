#INTEL_F_HOME = /usr/intel-ifort-10.0


CC	= icc
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp #-openmp-stubs #-openmp -DWRITE_RESTART #-parallel
#CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -Wall -fno-alias -openmp #-openmp-stubs #-openmp -DWRITE_RESTART #-parallel

FC	= ifort
FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all -openmp #-unroll0 -C -debug extended -traceback #-parallel
#FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all -vec-report3 -opt-report -unroll0 #-openmp #-C -debug extended -traceback #-parallel

LDFLAGS = -g -O3 -openmp  #-static #-parallel

LIBS = -L$(INTEL_F_HOME)/lib -lifcore
 
AS	= as
ASFLAGS = -g -gstabs

ENDG    =       $(shell echo ${MAKROS} | sed -e 's/\-D/_/g' | sed -e 's/\ //g')
SUFX    =      ${ENDG}_${SYSTEM}
