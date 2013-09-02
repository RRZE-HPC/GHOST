include ../../ghost/config.mk
include ../../ghost/include_$(strip $(COMPILER)).mk
include ../common.mk

.PHONY:clean distclean

%.o: %.c ../../ghost/config.mk
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(IPATH) -o $@ -c $< 

%.o: %.f ../../ghost/config.mk 
	@echo "==> Compile $< -> $@"
	@$(FC) $(FFLAGS) -o $@ -c $<


lanczos.x: lanczos.o $(OBJS)
	@echo "==> Link $@"
	@$(CC) $(CFLAGS) $(MAKROS) $(LPATH) -o $@ $^  $(LIBS)


clean:
	@echo "==> Clean"
	@rm -f *.o

distclean: clean
	@echo "==> Distclean"
	@rm -f *.x
