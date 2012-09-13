include config.mk

.PHONY: clean distclean install uninstall all

VPATH	=	./src/
OBJDIR   =   ./obj
IPATH	+=	-I./src/include

OBJS	=	$(COBJS) $(FOBJS)

ifeq ($(OPENCL),1)
OBJS	+=	$(CLOBJS)
endif


COBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(filter-out $(wildcard src/cl_*.c), $(wildcard src/*.c)))))
CLOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cl_*.c))))
FOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.f,%.o, $(wildcard src/*.f))))


LIBSPMVM=lib$(PREFIX)spmvm.a

$(OBJDIR)/%.o: %.c 
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

$(OBJDIR)/%.o: %.f 
	@echo "==> Compile $< -> $@"
	@$(FC) $(FFLAGS) -o $@ -c $<

all: $(OBJDIR) $(LIBSPMVM)

$(OBJDIR):
	@mkdir $(OBJDIR)

$(LIBSPMVM): $(OBJS)
	@echo "==> Create library $(LIBSPMVM)"
	@ar rcs  $(LIBSPMVM) $^

clean:
	@echo "==> Clean"
	@rm -rf obj/

distclean: clean
	@echo "==> Distclean"
	@rm -f *.a

install: all
	@mkdir -p $(INSTDIR)/lib
	@mkdir -p $(INSTDIR)/include
	@echo "==> Install library to $(INSTDIR)/lib"
	@cp -f $(LIBSPMVM) $(INSTDIR)/lib
	@echo "==> Install headers to $(INSTDIR)/include"
	@cp -f src/include/spmvm*.h $(INSTDIR)/include

uninstall: 
	@echo "==> Uninstall from $(INSTDIR)"
	@rm -f $(INSTDIR)/lib/lib*spmvm.a
	@rm -f $(INSTDIR)/include/spmvm*.h 

	
