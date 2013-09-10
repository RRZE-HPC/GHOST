LPATH+= -L$(strip $(GHOSTPATH))/lib
IPATH+= -I$(strip $(GHOSTPATH))/include/ghost -I$(strip $(GHOSTPATH))/include/ghost/likwid
LIBS+= -lghost

ifneq ($(strip $(MIC)),0)
LPATH += -L$(INTEL_C_HOME)/mkl/lib/mic
LIBS+= -limf -lsvml -lirng -lintlc
MAKROS+= -DMIC
endif

ifeq ($(LONGIDX),1)
MAKROS+= -DLONGIDX
endif

ifeq ($(strip $(ITAC)),1)
MAKROS+= -DVT
IPATH += -I$(ITC_INC)
endif

ifeq ($(strip $(LIKWID)),1)
MAKROS+= -DLIKWID_PERMON
IPATH += -I$(strip $(LIKWIDPATH))/include
LPATH := -L$(strip $(LIKWIDPATH))/lib $(LPATH)
LIBS += -llikwid
endif

ifeq ($(strip $(OPENCL)),1)
MAKROS+= -DOPENCL
#LIBS  += -lintelocl #-lOpenCL
LIBS  += -lOpenCL
IPATH += -I$(CUDA_HOME)/include -I$(strip $(GHOSTPATH))/include/ghost/cl
#LPATH += -L/usr/lib64/OpenCL/vendors/intel #${CUDA_LIB}
LPATH += -L$(CUDA_DRIVERDIR)/lib
endif

ifeq ($(strip $(CUDA)),1)
MAKROS+= -DCUDA
LIBS  += -lcudart
IPATH += -isystem $(CUDA_HOME)/include -I $(strip $(GHOSTPATH))/include/ghost/cu
LPATH += -L$(CUDA_HOME)/lib64
endif

ifeq ($(strip $(MPI)),1)
MAKROS+= -DGHOST_MPI
endif
