include config.mk

.PHONY: clean distclean install uninstall all header

VPATH      = ./src/
OBJDIR     = ./obj
TYPEHEADER = ./src/include/spmvm_type.h
IPATH     += -I./src/include

OBJS	=	$(COBJS) $(FOBJS)

ifeq ($(strip $(OPENCL)),1)
MAKROS+= -DOPENCL
PREFIX+= cl
IPATH += -I$(strip $(CL_INC))
OBJS  += $(CLOBJS)
endif

ifeq ($(strip $(LIKWID_MARKER)),1)
MAKROS+= -DLIKWID_MARKER
IPATH += -I$(strip $(LIKWID_INC))
endif

ifeq ($(strip $(LIKWID_MARKER_FINE)),1)
MAKROS+= -DLIKWID_MARKER_FINE
IPATH += -I$(strip $(LIKWID_INC))
endif

ifeq ($(strip $(DOUBLE)),1)
TYPES:= $(TYPES)"\#define DOUBLE \n"
endif

ifeq ($(strip $(COMPLEX)),1)
TYPES:= $(TYPES)"\#define COMPLEX \n"
endif

MAKROS+= -DDEBUG=$(DEBUG)

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

all: $(TYPEHEADER) $(OBJDIR) $(LIBSPMVM)

$(TYPEHEADER):  
	@echo "==> Generate type header"
	@echo "#ifndef _SPMVM_TYPES_H_" > $(TYPEHEADER)
	@echo -e "#define _SPMVM_TYPES_H_\n" >> $(TYPEHEADER)
	@echo -e $(TYPES) >> $(TYPEHEADER)
	@echo -e "\n#endif" >> $(TYPEHEADER)

$(OBJDIR):
	@mkdir $(OBJDIR)

$(LIBSPMVM): $(OBJS)
	@echo "==> Create library $(LIBSPMVM)"
	@ar rcs  $(LIBSPMVM) $^

clean:
	@echo "==> Clean"
	@rm -rf obj/
	@rm -rf $(TYPEHEADER)

distclean: clean
	@echo "==> Distclean"
	@rm -f *.a

install: all
	@mkdir -p $(strip $(INSTDIR))/lib
	@mkdir -p $(strip $(INSTDIR))/include
	@echo "==> Install library to $(strip $(INSTDIR))/lib"
	@cp -f $(LIBSPMVM) $(strip $(INSTDIR))/lib
	@echo "==> Install headers to $(strip $(INSTDIR))/include"
	@cp -f src/include/spmvm*.h $(strip $(INSTDIR))/include

uninstall: 
	@echo "==> Uninstall from $(strip $(INSTDIR))"
	@rm -f $(strip $(INSTDIR))/lib/lib*spmvm.a
	@rm -f $(strip $(INSTDIR))/include/spmvm*.h 

	
