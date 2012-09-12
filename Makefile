APPS=$(wildcard */)

all: $(APPS)
	for i in $(APPS); do make -C $$i; done

clean:
	for i in $(APPS); do make -C $$i clean; done

distclean:
	for i in $(APPS); do make -C $$i distclean; done
