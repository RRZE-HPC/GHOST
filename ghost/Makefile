include config.mk
include include_$(strip $(COMPILER)).mk

.PHONY: clean distclean install uninstall all header

VPATH      = ./src/ ./src/cl/ ./src/mpi/ ./src/mic/ ./src/avx/ ./src/sse/ ./src/spm_formats/ ./plugins/src/
OBJDIR     = ./obj
PLUGINDIR  = ./plugins
TYPES_H    = ./include/ghost_types_gen.h
FORMATS_H  = ./include/ghost_spmformats.h
IPATH     += -I./include/ -I./plugins/include/

OBJS	=	$(COBJS) $(FOBJS)
FORMATS = $(notdir $(wildcard $(PLUGINDIR)/include/*.h))

ifeq ($(strip $(MPI)),1)
MAKROS+= -DMPI
IPATH += -I./include/mpi
OBJS  += $(MPIOBJS)
endif

ifeq ($(strip $(AVX)),1)
MAKROS+= -DAVX
endif

ifeq ($(strip $(SSE)),1)
MAKROS+= -DSSE
endif

ifeq ($(strip $(MIC)),1)
MAKROS+= -DMIC
endif

ifeq ($(strip $(VSX)),1)
MAKROS+= -DVSX
endif

ifeq ($(strip $(OPENCL)),1)
MAKROS+= -DOPENCL
IPATH += -I$(CUDA_HOME)/include -I./include/cl
OBJS  += $(CLOBJS)
endif

ifeq ($(strip $(LIKWID)),1)
MAKROS+= -DLIKWID_PERFMON
IPATH += -I$(strip $(LIKWIDPATH))/include
endif

ifeq ($(strip $(LONGIDX)),1)
TYPES:= $(TYPES)"\#define GHOST_LONGIDX \n"
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

LIBGHOST=libghost.$(SUFFIX)

MAKROS+= -DDEBUG=$(DEBUG)
MAKROS+= -DPLUGINPATH=\"$(strip $(GHOSTPATH))/lib/ghost/\"
MAKROS+= -DHEADERPATH=\"$(strip $(GHOSTPATH))/include/ghost/\"

PLUGINS = $(addprefix $(PLUGINDIR)/,$(notdir $(patsubst %.c,%.so, $(wildcard $(PLUGINDIR)/src/*.c))))
COBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/*.c))))
CLOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cl/*.c))))
MPIOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/mpi/*.c))))
FOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.f,%.o, $(wildcard src/*.f))))


$(PLUGINDIR)/%.so: %.c config.mk
	@echo "==> Create plugin $< -> $@"
	@$(CC) $(CFLAGS) $(SHAREDFLAG) ${MAKROS} ${IPATH} -o $@ $<

$(OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

all: $(TYPES_H) $(FORMATS_H) $(OBJDIR) $(LIBGHOST) $(PLUGINS)

$(TYPES_H): config.mk  
	@echo "==> Generate types header"
	@echo "#ifndef __GHOST_TYPES_GEN_H__" > $(TYPES_H)
	@echo -e "#define __GHOST_TYPES_GEN_H__\n" >> $(TYPES_H)
	@echo -e $(TYPES) >> $(TYPES_H)
	@echo -e "\n#endif" >> $(TYPES_H)

$(FORMATS_H): config.mk
	@echo "==> Generate formats header"
	@echo "#ifndef __GHOST_SPMFORMATS_H__" > $(FORMATS_H)
	@echo -e "#define __GHOST_SPMFORMATS_H__\n" >> $(FORMATS_H)
	@(for format in $(FORMATS); do \
	echo "#include <$$format>" >> $(FORMATS_H); done)
	@echo -e "\n#endif" >> $(FORMATS_H)

$(OBJDIR):
	@mkdir $(OBJDIR)

ifeq ($(strip $(LIBTYPE)),static)
$(LIBGHOST): $(OBJS)
	@echo "==> Create static library $(LIBGHOST)"
	@ar rcs  $(LIBGHOST) $^
endif
ifeq ($(strip $(LIBTYPE)),shared)
$(LIBGHOST): $(OBJS)
	@echo "==> Create shared library $(LIBGHOST)"
	@$(CC) $(CFLAGS) $(SHAREDFLAG) -o $(LIBGHOST) $^
endif

clean:
	@echo "==> Clean"
	@rm -rf obj/
	@rm -rf $(TYPES_H)
	@rm -rf $(FORMATS_H)

distclean: clean
	@echo "==> Distclean"
	@rm -f *.a *.so
	@rm -f $(PLUGINDIR)/*.so

install: all
	@mkdir -p $(strip $(GHOSTPATH))/lib
	@mkdir -p $(strip $(GHOSTPATH))/include/ghost
	@mkdir -p $(strip $(GHOSTPATH))/lib/ghost
	@echo "==> Install library to $(strip $(GHOSTPATH))/lib/"
	@cp -f $(LIBGHOST) $(strip $(GHOSTPATH))/lib
	@echo "==> Install headers to $(strip $(GHOSTPATH))/include/ghost/"
	@cp -rf include/* $(strip $(GHOSTPATH))/include/ghost
	@echo "==> Install plugins to $(strip $(GHOSTPATH))/lib/ghost/"
	@cp -f $(PLUGINDIR)/*.so $(strip $(GHOSTPATH))/lib/ghost
	@cp -f $(PLUGINDIR)/src/*.cl $(strip $(GHOSTPATH))/lib/ghost

uninstall: 
	@echo "==> Uninstall from $(strip $(GHOSTPATH))"
	@rm -f $(strip $(GHOSTPATH))/lib/libghost.a
	@rm -f $(strip $(GHOSTPATH))/lib/libghost.so
	@rm -rf $(strip $(GHOSTPATH))/include/ghost/
	@rm -rf $(strip $(GHOSTPATH))/lib/ghost/ 

