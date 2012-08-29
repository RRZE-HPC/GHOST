include ../config.mk

.PHONY:clean distclean all

IPATH+= -I../src/include 
all: MMtoCRS.x

MMtoCRS.x: mmtocrs.c ../src/mmio.c
	$(CC) $(CFLAGS) $(LPATH) -o $@ $^

clean:
	-rm -f *.o

distclean: clean
	-rm -f *.x
