APPS=$(wildcard */)

all: $(APPS)
	@(for i in $(APPS); do echo "==> Enter $$i"; make -s -C $$i; done)

clean:
	@(for i in $(APPS); do echo "==> Enter $$i"; make -s -C $$i clean; done)

distclean:
	@(for i in $(APPS); do echo "==> Enter $$i"; make -s -C $$i distclean; done)
