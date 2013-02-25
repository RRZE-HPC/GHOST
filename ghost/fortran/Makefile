# this makefile is just for testing, the fortran interface
# should be compiled in a way consistent with the rest of ghost
# once it works.

FC=ifort
FFLAGS=-O0 -g -I../include/
LIB=-L${HOME}/local/lib -lghost
OBJ=minimal.o ghost_types.o ghost_fun.o ghost_util.o fghost.o

default: minimal.x

minimal.x: ${OBJ}
	${FC} ${FFLAGS} -o minimal.x ${OBJ} ${LIB}

%.o: %.F90
	${FC} ${FFLAGS} -c $<

clean: 
	-rm *.o *.mod minimal.x
	
# dependencies
ghost_fun.o: ghost_types.o
ghost_util.o: ghost_types.o
fghost.o: ghost_types.o
minimal.o: ghost_types.o ghost_fun.o ghost_util.o fghost.o

