include config.mk
include include_$(strip $(COMPILER)).mk

.PHONY: clean distclean install uninstall all header

VPATH      = ./src/ ./src/cl/ ./src/mpi/ ./src/mic/ ./src/avx/ ./src/sse/
OBJDIR     = ./obj
TYPEHEADER = ./include/ghost_types.h
IPATH     += -I./include/

OBJS	=	$(COBJS) $(FOBJS)

ifeq ($(strip $(MPI)),1)
MAKROS+= -DMPI
OBJS  += $(MPIOBJS)
endif

ifeq ($(strip $(AVX)),1)
MAKROS+= -DAVX
OBJS  += $(AVXOBJS)
endif

ifeq ($(strip $(SSE)),1)
MAKROS+= -DSSE
OBJS  += $(SSEOBJS)
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

ifeq ($(strip $(LIKWID)),1)
MAKROS+= -DLIKWID_PERMON
IPATH += -I$(strip $(LIKWID_INC))
endif

ifeq ($(strip $(MATDATA)),s)
TYPES:= $(TYPES)"\#define GHOST_MAT_SP \n"
else ifeq ($(strip $(MATDATA)),d)
TYPES:= $(TYPES)"\#define GHOST_MAT_DP \n"
else ifeq ($(strip $(MATDATA)),c)
TYPES:= $(TYPES)"\#define GHOST_MAT_SP \n"
TYPES:= $(TYPES)"\#define GHOST_MAT_COMPLEX \n"
else ifeq ($(strip $(MATDATA)),z)
TYPES:= $(TYPES)"\#define GHOST_MAT_DP \n"
TYPES:= $(TYPES)"\#define GHOST_MAT_COMPLEX \n"
endif

ifeq ($(strip $(VECDATA)),s)
TYPES:= $(TYPES)"\#define GHOST_VEC_SP \n"
else ifeq ($(strip $(VECDATA)),d)
TYPES:= $(TYPES)"\#define GHOST_VEC_DP \n"
else ifeq ($(strip $(VECDATA)),c)
TYPES:= $(TYPES)"\#define GHOST_VEC_SP \n"
TYPES:= $(TYPES)"\#define GHOST_VEC_COMPLEX \n"
else ifeq ($(strip $(VECDATA)),z)
TYPES:= $(TYPES)"\#define GHOST_VEC_DP \n"
TYPES:= $(TYPES)"\#define GHOST_VEC_COMPLEX \n"
endif

ifeq ($(strip $(LIBTYPE)),static)
SUFFIX=a
endif
ifeq ($(strip $(LIBTYPE)),shared)
SUFFIX=so
endif

LIBSPMVM=lib$(PREFIX)spmvm.$(SUFFIX)

MAKROS+= -DDEBUG=$(DEBUG)

COBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/*.c))))
CLOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cl/*.c))))
MPIOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/mpi/*.c))))
MICOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/mic/*.c))))
AVXOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/avx/*.c))))
SSEOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/sse/*.c))))
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
	@echo "#ifndef __GHOST_TYPES_H__" > $(TYPEHEADER)
	@echo -e "#define __GHOST_TYPES_H__\n" >> $(TYPEHEADER)
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
	@$(CC) $(CFLAGS) -shared -o $(LIBSPMVM) $^
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
	@cp -f include/spmvm*.h $(strip $(INSTDIR))/include
	@cp -f include/ghost_types.h $(strip $(INSTDIR))/include

#ghost_types
uninstall: 
	@echo "==> Uninstall from $(strip $(INSTDIR))"
	@rm -f $(strip $(INSTDIR))/lib/lib*spmvm.a
	@rm -f $(strip $(INSTDIR))/lib/lib*spmvm.so
	@rm -f $(strip $(INSTDIR))/include/spmvm*.h 

