include ../config.mk

.PHONY:clean distclean all

IPATH+= -I../src/include 
LPATH+= -L..
MYLIBS=-l$(PREFIX)spmvm $(LIBS)

%.o: %.c  
	$(CC) $(CFLAGS) -o $@ -c $< 

all: spmvm lanczos minimal 

spmvm: spmvm/main_spmvm.o
	$(CC) $(CFLAGS) $(LPATH) -o spmvm/$(PREFIX)$@.x $^  $(MYLIBS)

lanczos: lanczos/main_lanczos.o
	$(CC) $(CFLAGS) $(LPATH) -o lanczos/$(PREFIX)$@.x $^  $(MYLIBS)

minimal: minimal/main_minimal.o
	$(CC) $(CFLAGS) $(LPATH) -o minimal/$(PREFIX)$@.x $^  $(MYLIBS)

clean:
	-rm -f */*.o

distclean: clean
	-rm -f */*.x
