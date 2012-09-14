include config.mk

.PHONY:clean distclean all

mm2binCRS.x: src/mmtocrs.c src/mmio.c
	$(CC) $(CFLAGS) $(LPATH) -o $@ $^
	@mv $@ bin/

clean:
	-rm -f *.o

distclean: clean
	-rm -f bin/*.x
