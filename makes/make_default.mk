CC	= mpicc
CFLAGS  = -g ${MAKROS} ${IPATH} -fno-alias -openmp -fPIC #-Wall -Werror-all -Wremarks -Wcheck 

FC	= mpif90
FFLAGS  = -g ${MAKROS} -fno-alias -cpp -warn all -openmp -fPIC

LDFLAGS = -g ${RFLAGS} -openmp  -i_dynamic -fPIC

LIBS = -L$(INTEL_F_HOME)/compiler/lib/intel64 -lifcore -pthread $(LIKWID_LIB) -llikwid -L${CUDA_INSTALL_PATH}/lib64 -L${CUDA_DRIVERDIR}/lib -lOpenCL
IPATH += -I${CUDA_INSTALL_PATH}/include $(LIKWID_INC)

AS	= as
ASFLAGS = -g -gstabs 
