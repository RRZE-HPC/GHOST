ifeq ($(strip $(MPI)),1)
CC = mpcc
else
CC = xlc_r
endif

CFLAGS  = -std=c99 -qsmp=omp
SHAREDFLAG = -qmkshrobj

ifneq ($(strip $(DEBUG)),0)
CFLAGS += -O3 -qstrict -q64 -qtune=pwr7 -qarch=pwr7 -qaltivec -qhot -qsimd=auto
#CFLAGS += -O0 
else
CFLAGS += -O3 -qstrict -q64 -qtune=pwr7 -qarch=pwr7 -qaltivec -qhot -qsimd=auto
endif
