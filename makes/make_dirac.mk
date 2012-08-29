LIKWID_DIR = /global/homes/w/wellein/MK/likwid

CC	= mpicc
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp

FC	= mpif90
FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all -openmp

LDFLAGS = -g -O3 ${RFLAGS} -openmp  -i_dynamic

LPATH = -L$(INTEL_F_HOME)/lib 
LIBS = -lifcore -pthread

ifdef OPENCL
MAKROS+= -DOPENCL
PREFIX+= cl
IPATH += ${CUDA_INCLUDE}
LPATH += ${CUDA_LIB64} -L/usr/syscom/gpu/usrx/lib64
LIBS  += -lOpenCL
endif

ifdef LIKWID
MAKROS+= -DLIKWID
IPATH += -I${LIKWID_DIR}/include
LPATH += -L${LIKWID_DIR}/lib
LIBS  += -llikwid
endif

AS	= as
ASFLAGS = -g -gstabs
