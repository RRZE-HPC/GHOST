CC	= mpicc
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp -fPIC -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

FC	= mpif90
FFLAGS  = -g -O3 ${MAKROS} -fno-alias -cpp -warn all -openmp -fPIC

LDFLAGS = -g -O3 ${RFLAGS} -openmp  -i_dynamic -fPIC

LIBS = -L$(INTEL_F_HOME)/compiler/lib/intel64 -lifcore -pthread 
	
ifdef LIKWID
LIBS += $(LIKWID_LIB) -llikwid 
IPATH += $(LIKWID_INC)
endif

ifdef OPENCL
LIBS +=	-L${CUDA_INSTALL_PATH}/lib64 -L${CUDA_DRIVERDIR}/lib -lOpenCL
IPATH += -I${CUDA_INSTALL_PATH}/include
endif


AS	= as
ASFLAGS = -g -gstabs 
