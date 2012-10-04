CC  = mpicc
FC  = ifort

CFLAGS  = -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981
FFLAGS  = -openmp -nogen-interface -cpp -warn all


ifneq ($(strip $(DEBUG)),0)
CFLAGS += -g -O0
FFLAGS += -g -O0
else
CFLAGS += -O3
FFLAGS += -O3
endif
