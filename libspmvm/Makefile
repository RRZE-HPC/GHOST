include config.mk
include include_$(strip $(COMPILER)).mk

.PHONY: clean distclean install uninstall all header

VPATH      = ./src/
OBJDIR     = ./obj
TYPEHEADER = ./src/include/spmvm_type.h
IPATH     += -I./src/include

OBJS	=	$(COBJS) $(FOBJS)

ifeq ($(strip $(MPI)),1)
MAKROS+= -DMPI
OBJS  += $(MPIOBJS)
endif

ifeq ($(strip $(AVX)),1)
MAKROS+= -DAVX
OBJS  += $(AVXOBJS)
endif

ifeq ($(strip $(MIC)),1)
MAKROS+= -DMIC
OBJS  += $(MICOBJS)
endif

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
else
TYPES:= $(TYPES)"\#define SINGLE \n"
endif

ifeq ($(strip $(COMPLEX)),1)
TYPES:= $(TYPES)"\#define COMPLEX \n"
endif

ifeq ($(strip $(LIBTYPE)),static)
SUFFIX=a
endif
ifeq ($(strip $(LIBTYPE)),shared)
SUFFIX=so
endif

LIBSPMVM=lib$(PREFIX)spmvm.$(SUFFIX)

MAKROS+= -DDEBUG=$(DEBUG)

COBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o,\
				$(filter-out $(wildcard src/cl_*.c src/mpi_*.c src/mic_*.c src/avx_*.c), $(wildcard src/*.c)))))
CLOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cl_*.c))))
MPIOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/mpi_*.c))))
MICOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/mic_*.c))))
AVXOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/avx_*.c))))
FOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.f,%.o, $(wildcard src/*.f))))



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

ifeq ($(strip $(LIBTYPE)),static)
$(LIBSPMVM): $(OBJS)
	@echo "==> Create static library $(LIBSPMVM)"
	@ar rcs  $(LIBSPMVM) $^
endif
ifeq ($(strip $(LIBTYPE)),shared)
$(LIBSPMVM): $(OBJS)
	@echo "==> Create shared library $(LIBSPMVM)"
	@$(CC) -shared -o $(LIBSPMVM) $^
endif

clean:
	@echo "==> Clean"
	@rm -rf obj/
	@rm -rf $(TYPEHEADER)

distclean: clean
	@echo "==> Distclean"
	@rm -f *.a *.so

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
	@rm -f $(strip $(INSTDIR))/lib/lib*spmvm.so
	@rm -f $(strip $(INSTDIR))/include/spmvm*.h 

	
