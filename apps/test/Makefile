include ../../ghost/config.mk
include ../../ghost/include_$(strip $(COMPILER)).mk
include ../common.mk

.PHONY:clean distclean all

all: vec.x omp.x

%.o: %.c ../../ghost/config.mk 
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 

vec.x: vec.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(LPATH) -o $@ $<  $(LIBS)

omp.x: omp.o
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(LPATH) -o $@ $<  $(LIBS)

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

