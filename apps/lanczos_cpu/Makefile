include ../../libspmvm/config.mk
include ../../libspmvm/include_$(strip $(COMPILER)).mk

LPATH+= -L$(strip $(INSTDIR))/lib
IPATH+= -I$(strip $(INSTDIR))/include
LIBS+=-l$(PREFIX)ghost


.PHONY:clean distclean all

CC=mpicc
CFLAGS=-g -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

FC	= ifort
FFLAGS  =-O3 -nogen-interface -fno-alias -cpp -warn -openmp  

%.o: %.c 
	echo $(INSTPATH)
	@echo "==> Compile $< -> $@"
	$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 

%.o: %.f 
	@echo "==> Compile $< -> $@"
	@$(FC) $(FFLAGS) -o $@ -c $<


$(PREFIX)lanczos.x: lanczos.o imtql1.o pythag.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $@ $^  $(LIBS)


clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x
