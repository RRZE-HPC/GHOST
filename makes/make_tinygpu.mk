LIKWID_DIR = /apps/likwid/devel/

CC	= mpicc
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp -Wall -Werror-all -fPIC #-Wremarks -Wcheck

FC	= mpif90
FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all -openmp -fPIC #-vec-report3 -opt-report -unroll0 #-openmp #-C -debug extended -traceback #-parallel

LDFLAGS = -g -O3 ${RFLAGS} -openmp  -i_dynamic #-openmp -static #-parallel

LIBS = -L$(INTEL_F_HOME)/compiler/lib/intel64 -lifcore -pthread -L${CUDA_INSTALL_PATH}/lib64 -L${CUDA_DRIVERDIR}/lib -lOpenCL
IPATH += -I${CUDA_INSTALL_PATH}/include

AS	= as
ASFLAGS = -g -gstabs 

SUFX    =      _${SYSTEM}
