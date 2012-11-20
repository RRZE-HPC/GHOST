include config.mk
include include_$(strip $(COMPILER)).mk

.PHONY: clean distclean install uninstall all header

VPATH      = ./src/ ./src/cl/ ./src/mpi/ ./src/mic/ ./src/avx/ ./src/sse/ ./src/spm_formats/
OBJDIR     = ./obj
TYPES_H    = ./include/ghost_types.h
FORMATS_H  = ./include/ghost_spmformats.h
IPATH     += -I./include/ -I./include/spm_formats/

OBJS	=	$(COBJS) $(FOBJS)
FORMATS = $(notdir $(wildcard include/spm_formats/*.h))

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

LIBSPMVM=lib$(PREFIX)ghost.$(SUFFIX)

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

all: $(TYPES_H) $(FORMATS_H) $(OBJDIR) $(LIBSPMVM)

$(TYPES_H):  
	@echo "==> Generate types header"
	@echo "#ifndef __GHOST_TYPES_H__" > $(TYPES_H)
	@echo -e "#define __GHOST_TYPES_H__\n" >> $(TYPES_H)
	@echo -e $(TYPES) >> $(TYPES_H)
	@echo -e "\n#endif" >> $(TYPES_H)

$(FORMATS_H):  
	@echo "==> Generate formats header"
	@echo "#ifndef __GHOST_FORMATS_H__" > $(FORMATS_H)
	@echo -e "#define __GHOST_FORMATS_H__\n" >> $(FORMATS_H)
	@(for format in $(FORMATS); do \
	echo "#include \"$$format\"" >> $(FORMATS_H); done)
	@echo -e "\n#endif" >> $(FORMATS_H)

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
	@rm -rf $(TYPES_H)
	@rm -rf $(FORMATS_H)

distclean: clean
	@echo "==> Distclean"
	@rm -f *.a *.so

install: all
	@mkdir -p $(strip $(INSTDIR))/lib
	@mkdir -p $(strip $(INSTDIR))/include
	@echo "==> Install library to $(strip $(INSTDIR))/lib"
	@cp -f $(LIBSPMVM) $(strip $(INSTDIR))/lib
	@echo "==> Install headers to $(strip $(INSTDIR))/include"
	@cp -f include/ghost*.h $(strip $(INSTDIR))/include

uninstall: 
	@echo "==> Uninstall from $(strip $(INSTDIR))"
	@rm -f $(strip $(INSTDIR))/lib/lib*ghost.a
	@rm -f $(strip $(INSTDIR))/lib/lib*ghost.so
	@rm -f $(strip $(INSTDIR))/include/spmvm*.h 

