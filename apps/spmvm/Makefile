include ../../libspmvm/config.mk
include ../../libspmvm/include_$(strip $(COMPILER)).mk

WDDT=0

CUDA_INC=-I$(CUDA_HOME)/include
CUDA_LIB=-L$(CUDA_HOME)/lib64 -L$(CUDA_DRIVERDIR)/lib

GHOSTPATH=$(HOME)/local

LPATH+= -L$(GHOSTPATH)/lib
IPATH+= -I$(GHOSTPATH)/include
LIBS+=-l$(PREFIX)ghost

.PHONY:clean distclean all

#CC=icc
#CFLAGS= -g -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

ifeq ($(WDDT),1)
LIBS+= $(DDT)/lib/64/libdmallocth.a 
endif

ifeq ($(strip $(MIC)),1)
LIBS+= -limf -lsvml -lirng -lintlc
endif

ifeq ($(strip $(LIKWID)),1)
MAKROS+= -DLIKWID_PERMON
IPATH += -I$(strip $(LIKWIDPATH))/include
LPATH += -I$(strip $(LIKWIDPATH))/lib
LIBS += -llikwid
endif

ifeq ($(strip $(OPENCL)),1)
MAKROS+= -DOPENCL
PREFIX= cl
#LIBS  += -lintelocl #-lOpenCL
LIBS  += -lOpenCL
IPATH += ${CUDA_INC}
#LPATH += -L/usr/lib64/OpenCL/vendors/intel #${CUDA_LIB}
LPATH += ${CUDA_LIB}
endif

%.o: %.c  
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< $(LIBS) 


$(PREFIX)spmvm.x: main_spmvm.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $@ $^  $(LIBS) 

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

