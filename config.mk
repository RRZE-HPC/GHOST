OPENCL=1
LIKWID=1
LIKWID_MARKER=0
LIKWID_MARKER_FINE=0
DOUBLE=1
COMPLEX=0
DEBUG=0

INSTDIR=~/app/libspmvm
CUDA_INC=$(CUDA_HOME)/include
CUDA_LIB=$(CUDA_HOME)/lib64
LIKWID_INC=/home/hpc/unrz/unrza317/app/likwid/include
LIKWID_LIB=/home/hpc/unrz/unrza317/app/likwid/lib

CC	= mpicc
CFLAGS  = -g -O3 ${MAKROS} ${IPATH} -fno-alias -openmp -fPIC -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

FC	= mpif90
FFLAGS  = -g -O3 -fno-alias -cpp -warn all -openmp -fPIC


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
# DO NOT EDIT BELOW UNLESS YOU KNOW WHAT YOU ARE DOING                         #
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

ifeq ($(OPENCL),1)
MAKROS+= -DOPENCL
PREFIX+= cl
IPATH += -I${CUDA_INC}
LIBS  += -lOpenCL
endif

ifeq ($(LIKWID),1)
MAKROS+= -DLIKWID
IPATH += -I${LIKWID_INC}
LIBS  += -llikwid
endif

ifeq ($(LIKWID_MARKER),1)
MAKROS+= -DLIKWID_MARKER
endif

ifeq ($(LIKWID_MARKER_FINE),1)
MAKROS+= -DLIKWID_MARKER_FINE
endif

ifeq ($(DOUBLE),1)
MAKROS+= -DDOUBLE
endif

ifeq ($(COMPLEX),1)
MAKROS+= -DCOMPLEX
endif

MAKROS+= -DDEBUG=$(DEBUG)
