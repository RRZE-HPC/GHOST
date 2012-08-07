CC=icc

.PHONY:clean all

all: MMtoCRS.x

MMtoCRS.x: mmtocrs.c ../src/mmio.c
	$(CC) -I../src/include -o $@ $^

clean:
	-rm -f *.o
