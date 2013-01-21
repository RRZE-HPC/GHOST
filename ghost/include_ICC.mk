ifeq ($(strip $(MPI)),1)
CC = mpicc
else
CC = icc
endif

<<<<<<< HEAD:ghost/include_ICC.mk
CFLAGS  = -openmp -fPIC -std=c99 
SHAREDFLAG = -shared
=======
CFLAGS  = -openmp -fPIC -std=c99 -fno-alias 
>>>>>>> fb6e93433357ebb21bc81dc637387cfff39b0d9b:ghost/include_ICC.mk
FFLAGS  = -openmp -fPIC -nogen-interface -cpp

LIBS = -limf -lm


ifeq ($(strip $(AVX)),1)
CFLAGS += -mavx
endif

ifeq ($(strip $(SSE)),1)
CFLAGS += -msse4.2
endif

ifeq ($(strip $(MIC)),1)
CFLAGS += -mmic
endif

ifneq ($(strip $(DEBUG)),0)
CFLAGS += -g -O0 #-check-pointers=rw -check-pointers-dangling=all -rdynamic
else
CFLAGS += -O3 -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981
endif
