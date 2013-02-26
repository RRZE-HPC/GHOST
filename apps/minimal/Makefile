include ../../ghost/config.mk
include ../../ghost/include_$(strip $(COMPILER)).mk

.PHONY:clean distclean

LPATH+= -L$(strip $(GHOSTPATH))/lib
IPATH+= -I$(strip $(GHOSTPATH))/include/ghost
LIBS+=  -lghost -lifcore

ifeq ($(strip $(CUDA)),1)
MAKROS+= -DCUDA
LIBS  += -lcudart
IPATH += -isystem$(CUDA_HOME)/include -I$(strip $(GHOSTPATH))/include/ghost/cu
LPATH +=  -L$(CUDA_HOME)/lib64
endif

%.o: %.c ../../ghost/config.mk 
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 


minimal.x: minimal.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(LPATH) -o $@ $<  $(LIBS)

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

