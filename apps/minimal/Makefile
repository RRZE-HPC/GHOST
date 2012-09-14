LIBSPMVMPATH=/home/hpc/unrz/unrza317/app/libspmvm

LPATH+= -L$(LIBSPMVMPATH)/lib
IPATH+= -I$(LIBSPMVMPATH)/include
LIBS+=  -lspmvm

.PHONY:clean distclean

CC=mpicc
CFLAGS=-g -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

%.o: %.c  
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 


$(PREFIX)minimal.x: main_minimal.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(LPATH) -o $@ $^  $(LIBS)

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

