#INTEL_F_HOME = /usr/intel-ifort-10.0
#LIKWID_DIR = /apps/likwid/devel/
LIKWID_DIR = /apps/likwid/stable

CC	= mpicc
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp #-C -Wall# -DWRITE_RESTART #-parallel
#CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp-stubs -openmp# -DWRITE_RESTART #-parallel

FC	= mpif90
FFLAGS  = -g -O3 -132 ${MAKROS} -fno-alias -cpp -warn all -openmp #-vec-report3 -opt-report -unroll0 #-openmp #-C -debug extended -traceback #-parallel

LDFLAGS = -g -O3 ${RFLAGS} -i_dynamic -openmp #-static #-parallel

LIBS = -L$(INTEL_F_HOME)/lib -lifcore -L${LIKWID_DIR}/lib -llikwid -pthread
IPATH   += -I${LIKWID_DIR}/include


#IPATH	+=	-I/$(INTEL_F_HOME)/include
 
AS	= as
ASFLAGS = -g -gstabs

ENDG    =       $(shell echo ${MAKROS} | sed -e 's/\-D/_/g' | sed -e 's/\ //g')
SUFX    =      ${ENDG}_${SYSTEM}
