include ../../ghost/config.mk
COMPILER:=$(strip $(COMPILER))
LIBTYPE:=$(strip $(LIBTYPE))
LONGIDX:=$(strip $(LONGIDX))
MPI:=$(strip $(MPI))
VSX:=$(strip $(VSX))
MIC:=$(strip $(MIC))
AVX:=$(strip $(AVX))
SSE:=$(strip $(SSE))
OPENCL:=$(strip $(OPENCL))
CUDA:=$(strip $(CUDA))
LIKWID:=$(strip $(LIKWID))
DEBUG:=$(strip $(DEBUG))
GHOSTPATH:=$(strip $(GHOSTPATH))
LIKWIDPATH:=$(strip $(LIKWIDPATH))

include ../../ghost/include_$(strip $(COMPILER)).mk

WDDT=0
WTV=0

.PHONY:clean distclean all

LPATH+= -L$(strip $(GHOSTPATH))/lib
IPATH+= -I$(strip $(GHOSTPATH))/include/ghost
LIBS+= -lghost -lifcore

ifeq ($(WDDT),1)
LIBS+= $(DDT)/lib/64/libdmallocth.a 
endif
ifeq ($(WTV),1)
LIBS+= /apps/toolworks/totalview.8.9.0-0/linux-x86-64/lib/libdbfork_64.a 
endif

ifeq ($(strip $(MIC)),1)
LIBS+= -limf -lsvml -lirng -lintlc
endif
ifeq ($(strip $(MIC)),2)
LIBS+= -limf -lsvml -lirng -lintlc
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
MAKROS+= -DMPI
endif

%.o: %.c ../../ghost/config.mk 
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< $(LIBS) 


spmvm.x: main_spmvm.o 
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $@ $^  $(LIBS) 

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

