include ../../ghost/config.mk
include ../../ghost/include_$(strip $(COMPILER)).mk

WDDT=0

.PHONY:clean distclean all

LPATH+= -L$(strip $(GHOSTPATH))/lib
IPATH+= -I$(strip $(GHOSTPATH))/include/ghost
LIBS+=  -lghost

ifeq ($(WDDT),1)
LIBS+= $(DDT)/lib/64/libdmallocth.a 
endif

ifeq ($(strip $(MIC)),1)
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
IPATH += ${CUDA_INC}
#LPATH += -L/usr/lib64/OpenCL/vendors/intel #${CUDA_LIB}
LPATH += ${CUDA_LIB}
endif

%.o: %.c  
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

