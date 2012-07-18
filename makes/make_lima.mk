LIKWID_DIR = /apps/likwid/devel/

CC	= mpicc
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp -fPIC #-Wremarks -Wcheck

FC	= mpif90
FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all -openmp -fPIC

LDFLAGS = -g -O3 ${RFLAGS} -openmp  -i_dynamic

LIBS = -L$(INTEL_F_HOME)/compiler/lib/intel64 -lifcore -pthread $(LIKWID_LIB) -llikwid 
IPATH += -I${CUDA_INSTALL_PATH}/include $(LIKWID_INC)

AS	= as
ASFLAGS = -g -gstabs 

SUFX    =      _${SYSTEM}
