LIKWID_DIR = /apps/likwid/devel/

CC	= mpicc
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp #-C -Wall# -DWRITE_RESTART #-parallel
#CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp-stubs -openmp# -DWRITE_RESTART #-parallel

FC	= mpif90
FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all -openmp #-vec-report3 -opt-report -unroll0 #-openmp #-C -debug extended -traceback #-parallel

LDFLAGS = -g -O3 ${RFLAGS} -openmp  -i_dynamic #-openmp -static #-parallel

LIBS = -L$(INTEL_F_HOME)/lib -lifcore -L${LIKWID_DIR}/lib -llikwid -pthread -L${CUDA_INSTALL_PATH}/lib64 -L${CUDA_DRIVERDIR}/lib -lOpenCL
IPATH   += -I${LIKWID_DIR}/include -I${CUDA_INSTALL_PATH}/include

NVCC = nvcc
NVCCFLAGS = -O3 ${MAKROS} -gencode=arch=compute_13,code=\"sm_13,compute_13\" -gencode=arch=compute_20,code=\"sm_20,compute_20\" ${IPATH} -I${CUDA_INSTALL_PATH}/include

#IPATH	+=	-I/$(INTEL_F_HOME)/include
 
AS	= as
ASFLAGS = -g -gstabs

ENDG    =       $(shell echo ${MAKROS} | sed -e 's/\-D/_/g' | sed -e 's/\ //g')
SUFX    =      ${ENDG}_${SYSTEM}
