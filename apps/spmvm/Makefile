include ../../ghost/config.mk
include ../../ghost/include_$(strip $(COMPILER)).mk
include ../common.mk

.PHONY:clean distclean all

WDDT=0
WTV=0

ifeq ($(WDDT),1)
LIBS+= $(DDT)/lib/64/libdmallocth.a 
endif
ifeq ($(WTV),1)
LIBS+= /apps/toolworks/totalview.8.9.0-0/linux-x86-64/lib/libdbfork_64.a 
endif


%.o: %.c ../../ghost/config.mk 
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< $(LIBS) 


spmvm.x: main_spmvm.o 
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $@ $^  $(LIBS) 

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

