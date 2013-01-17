CC = xlc

CFLAGS  = -std=c99 
SHAREDFLAG = -qmkshrobj

ifneq ($(strip $(DEBUG)),0)
CFLAGS += -g -O0
else
CFLAGS += -O3 -qstrict -q64 -qtune=pwr7 -qarch=pwr7 -qaltivec -qhot -qsimd=auto
endif
