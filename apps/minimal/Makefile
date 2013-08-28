include ../../ghost/config.mk
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
ifeq ($(strip $(MPI)),1)
MAKROS+= -DGHOST_MPI
endif
ifeq ($(LONGIDX),1)
MAKROS+= -DLONGIDX
endif


%.o: %.c ../../ghost/config.mk 
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 


minimal.x: minimal.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(LPATH) -o $@ $<  $(LIBS)

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

