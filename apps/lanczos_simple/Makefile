include ../../ghost/config.mk
include ../../ghost/include_$(strip $(COMPILER)).mk

.PHONY:clean distclean

FC	= gfortran
FFLAGS  =-O3 -warn -openmp  

LPATH+= -L$(strip $(GHOSTPATH))/lib
IPATH+= -I$(strip $(GHOSTPATH))/include/ghost
LIBS+=  -lghost

%.o: %.c 
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 

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
