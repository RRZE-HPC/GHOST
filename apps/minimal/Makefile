OPENCL=0
DOUBLE=1

LIBSPMVMPATH=/home/hpc/unrz/unrza317/app/libspmvm

LPATH+= -L$(LIBSPMVMPATH)/lib
IPATH+= -I$(LIBSPMVMPATH)/include
LIBS+=-l$(PREFIX)spmvm

.PHONY:clean distclean all

CC=mpicc
CFLAGS=-g -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

ifeq ($(DOUBLE),1)
MAKROS+= -DDOUBLE
endif

ifeq ($(COMPLEX),1)
MAKROS+= -DCOMPLEX
endif


%.o: %.c  
	$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 


minimal: main_minimal.o
	$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $(PREFIX)$@.x $^  $(LIBS)

clean:
	-rm -f *.o
	-rm -f *genmod*

distclean: clean
	-rm -f *.x
