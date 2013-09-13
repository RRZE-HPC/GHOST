include ../../ghost/config.mk
include ../../ghost/include_$(strip $(COMPILER)).mk
include ../common.mk

.PHONY:clean distclean

Q=@  

%.o: %.c ../../ghost/config.mk 
	@echo "==> Compile $< -> $@"
	${Q}$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 


hybrid.x: hybrid.o
	@echo "==> Link $@"
	${Q}$(CC) $(CFLAGS) $(LPATH) -o $@ $<  $(LIBS)

clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x

