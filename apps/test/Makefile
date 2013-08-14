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

.PHONY:clean distclean

LPATH+= -L$(strip $(GHOSTPATH))/lib
IPATH+= -I$(strip $(GHOSTPATH))/include/ghost
LIBS+=  -lghost -lifcore

ifeq ($(strip $(MIC)),1)
LIBS+= -limf -lsvml -lirng -lintlc
endif
ifeq ($(strip $(MIC)),2)
LIBS+= -limf -lsvml -lirng -lintlc
endif

ifeq ($(strip $(CUDA)),1)
MAKROS+= -DCUDA
LIBS  += -lcudart
IPATH += -isystem$(CUDA_HOME)/include -I$(strip $(GHOSTPATH))/include/ghost/cu
LPATH +=  -L$(CUDA_HOME)/lib64
endif
ifeq ($(strip $(OPENCL)),1)
MAKROS+= -DOPENCL
LIBS  += -lOpenCL
IPATH += -I$(CUDA_HOME)/include -I$(strip $(GHOSTPATH))/include/ghost/cl
LPATH += -L$(CUDA_DRIVERDIR)/lib
endif
ifeq ($(strip $(LIKWID)),1)
MAKROS+= -DLIKWID_PERMON
IPATH += -I$(strip $(LIKWIDPATH))/include
LPATH := -L$(strip $(LIKWIDPATH))/lib $(LPATH)
LIBS += -llikwid
endif

%.o: %.c ../../ghost/config.mk 
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 

all: vec.x

vec.x: vec.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(LPATH) -o $@ $<  $(LIBS)

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

