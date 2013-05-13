include config.mk

COMPILER:=$(strip $(COMPILER))
LIBTYPE:=$(strip $(LIBTYPE))
LONGIDX:=$(strip $(LONGIDX))
MPI:=$(strip $(MPI))
VSX:=$(strip $(VSX))
MIC:=$(strip $(MIC))
AVX:=$(strip $(AVX))
SSE:=$(strip $(SSE))
OPENCL:=$(strip $(OPENCL))
CUDA:=$(strip $(CUDA))
LIKWID:=$(strip $(LIKWID))
DEBUG:=$(strip $(DEBUG))
GHOSTPATH:=$(strip $(GHOSTPATH))
LIKWIDPATH:=$(strip $(LIKWIDPATH))

include include_$(COMPILER).mk

.NOTPARALLEL: $(PLUGINS)

.PHONY: clean distclean install uninstall all header 

VPATH      = ./src/ ./src/fortran/ ./src/fortran/modules/ ./src/cl/ ./src/mpi/ ./src/cu/ 
OBJDIR     = ./obj
MODDIR     = ./mod
C_OBJDIR   = ./obj/c
D_OBJDIR   = ./obj/d
S_OBJDIR   = ./obj/s
Z_OBJDIR   = ./obj/z
PLUGINDIR  = ./plugins
TYPES_H    = ./include/ghost_types_gen.h
FORMATS_H  = ./include/ghost_spmformats.h
IPATH     += -I./include/ -I./include/fortran/
Q          = @ 

OBJS	=	$(COBJS) $(FMODULES) $(FOBJS) 
FORMATS = $(notdir $(wildcard $(PLUGINDIR)/*/include/*.h))

ifeq ($(MPI),1)
MAKROS+= -DMPI
IPATH += -I./include/mpi
OBJS  += $(MPIOBJS)
endif

ifeq ($(AVX),1)
MAKROS+= -DAVX
endif
ifeq ($(AVX),2)
MAKROS+= -DAVX -DAVX_INTR
endif

ifeq ($(SSE),1)
MAKROS+= -DSSE
endif
ifeq ($(SSE),2)
MAKROS+= -DSSE -DSSE_INTR
endif

ifeq ($(MIC),1)
MAKROS+= -DMIC
endif
ifeq ($(MIC),2)
MAKROS+= -DMIC -DMIC_INTR
endif

ifeq ($(VSX),1)
MAKROS+= -DVSX
endif
ifeq ($(VSX),2)
MAKROS+= -DVSX -DVSX_INTR
endif

ifeq ($(OPENCL),1)
MAKROS+= -DOPENCL
IPATH += -I$(CUDA_HOME)/include -I./include/cl
OBJS  += $(CLOBJS)
endif

ifeq ($(CUDA),1)
include include_NVCC.mk
MAKROS+= -DCUDA
IPATH += -isystem $(CUDA_HOME)/include -I./include/cu
OBJS  += $(CUOBJS)
endif

ifeq ($(LIKWID),1)
MAKROS+= -DLIKWID_PERFMON
IPATH += -I$(LIKWIDPATH)/include
endif

ifeq ($(LONGIDX),1)
TYPES:= $(TYPES)"\#define GHOST_LONGIDX \n"
endif

ifeq ($(LIBTYPE),static)
SUFFIX=a
endif
ifeq ($(LIBTYPE),shared)
SUFFIX=so
endif

LIBGHOST=libghost.$(SUFFIX)

MAKROS+= -DDEBUG=$(DEBUG)
MAKROS+= -DPLUGINPATH=\"$(GHOSTPATH)/lib/ghost/\"
MAKROS+= -DHEADERPATH=\"$(GHOSTPATH)/include/ghost/\"

IPATH += $(addsuffix include/, $(addprefix -I, $(wildcard $(PLUGINDIR)/*/)))
VPATH += $(addsuffix src/, $(wildcard $(PLUGINDIR)/*/))

PLUGINS = $(addsuffix .so, $(addprefix $(PLUGINDIR)/c_, $(notdir $(patsubst %/, %, $(filter %/, $(wildcard $(PLUGINDIR)/*/)))))) \
	$(addsuffix .so, $(addprefix $(PLUGINDIR)/d_, $(notdir $(patsubst %/, %, $(filter %/, $(wildcard $(PLUGINDIR)/*/)))))) \
	$(addsuffix .so, $(addprefix $(PLUGINDIR)/s_, $(notdir $(patsubst %/, %, $(filter %/, $(wildcard $(PLUGINDIR)/*/)))))) \
	$(addsuffix .so, $(addprefix $(PLUGINDIR)/z_, $(notdir $(patsubst %/, %, $(filter %/, $(wildcard $(PLUGINDIR)/*/))))))
PLUGINOBJS = $(addprefix $(C_OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard plugins/*/src/*.c)))) \
	$(addprefix $(D_OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard plugins/*/src/*.c)))) \
	$(addprefix $(S_OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard plugins/*/src/*.c)))) \
	$(addprefix $(Z_OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard plugins/*/src/*.c))))

ifeq ($(CUDA),1)
PLUGINOBJS += $(addprefix $(C_OBJDIR)/,$(notdir $(patsubst %.cu,%.cu.o, $(wildcard plugins/*/src/*.cu)))) \
	$(addprefix $(D_OBJDIR)/,$(notdir $(patsubst %.cu,%.cu.o, $(wildcard plugins/*/src/*.cu)))) \
	$(addprefix $(S_OBJDIR)/,$(notdir $(patsubst %.cu,%.cu.o, $(wildcard plugins/*/src/*.cu)))) \
	$(addprefix $(Z_OBJDIR)/,$(notdir $(patsubst %.cu,%.cu.o, $(wildcard plugins/*/src/*.cu))))
endif

COBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/*.c))))
CLOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cl/*.c))))
CUOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cu/*.c))))
MPIOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/mpi/*.c))))
FOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.F90,%.o, $(wildcard src/fortran/*.F90))))
FMODULES = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.F90,%.o, $(wildcard src/fortran/modules/*.F90))))

all: $(OBJDIR) $(MODDIR) $(C_OBJDIR) $(D_OBJDIR) $(S_OBJDIR) $(Z_OBJDIR) $(PLUGINS) $(LIBGHOST)

$(PLUGINS): $(PLUGINOBJS)
	$(eval dt := $(shell echo $(notdir $@) | cut -c 1))
	$(eval pl := $(shell echo $(basename $(notdir $@)) | cut -c 3-))
	$(eval objs := $(shell find $(PLUGINDIR)/$(pl)/src/ -name "*.c" -printf "./obj/$(dt)/%f " | sed 's/\.c/\.o/g'))
ifeq ($(CUDA),1)
	$(eval objs += $(shell find $(PLUGINDIR)/$(pl)/src/ -name "*.cu" -printf "./obj/$(dt)/%f " | sed 's/\.cu/\.cu.o/g'))
endif
	@echo "==> Create plugin $(objs) -> $@"
	${Q}$(CC) $(CFLAGS) $(SHAREDFLAG) ${MAKROS} ${IPATH} -o $@ $(objs)
	
$(C_OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile complex float variant $< -> $@"
	${Q}$(CC) $(CFLAGS) ${MAKROS} -DGHOST_DT_C ${IPATH} -o $@ -c $<

$(D_OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile double variant $< -> $@"
	${Q}$(CC) $(CFLAGS) ${MAKROS} -DGHOST_DT_D ${IPATH} -o $@ -c $<

$(S_OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile float variant $< -> $@"
	${Q}$(CC) $(CFLAGS) ${MAKROS} -DGHOST_DT_S  ${IPATH} -o $@ -c $<

$(Z_OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile complex double variant $< -> $@"
	${Q}$(CC) $(CFLAGS) ${MAKROS} -DGHOST_DT_Z ${IPATH} -o $@ -c $<

$(OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile $< -> $@"
	${Q}$(CC) $(CFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

$(OBJDIR)/%.o: %.F90 config.mk
	@echo "==> Compile $< -> $@"
	${Q}$(FC) $(FFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

ifeq ($(CUDA),1)
$(C_OBJDIR)/%.cu.o: %.cu config.mk
	@echo "==> Compile complex float variant $< -> $@"
	${Q}$(NVCC) $(NVCCFLAGS) ${MAKROS} -DGHOST_DT_C ${IPATH} -o $@ -c $<
$(D_OBJDIR)/%.cu.o: %.cu config.mk
	@echo "==> Compile double variant $< -> $@"
	${Q}$(NVCC) $(NVCCFLAGS) ${MAKROS} -DGHOST_DT_D ${IPATH} -o $@ -c $<
$(S_OBJDIR)/%.cu.o: %.cu config.mk
	@echo "==> Compile float variant $< -> $@"
	${Q}$(NVCC) $(NVCCFLAGS) ${MAKROS} -DGHOST_DT_S ${IPATH} -o $@ -c $<
$(Z_OBJDIR)/%.cu.o: %.cu config.mk
	@echo "==> Compile complex double variant $< -> $@"
	${Q}$(NVCC) $(NVCCFLAGS) ${MAKROS} -DGHOST_DT_Z ${IPATH} -o $@ -c $<
endif

#$(FORMATS_H): config.mk
#	@echo "==> Generate formats header"
#	@echo "#ifndef __GHOST_SPMFORMATS_H__" > $(FORMATS_H)
#	@echo -e "#define __GHOST_SPMFORMATS_H__\n" >> $(FORMATS_H)
#	@(for format in $(FORMATS); do \
#	echo "#include <$$format>" >> $(FORMATS_H); done)
#	@echo -e "\n#endif" >> $(FORMATS_H)

$(C_OBJDIR):
	@mkdir -p $(C_OBJDIR)

$(D_OBJDIR):
	@mkdir -p $(D_OBJDIR)

$(S_OBJDIR):
	@mkdir -p $(S_OBJDIR)

$(Z_OBJDIR):
	@mkdir -p $(Z_OBJDIR)

$(OBJDIR):
	@mkdir $(OBJDIR)

$(MODDIR):
	@mkdir $(MODDIR)

ifeq ($(LIBTYPE),static)
$(LIBGHOST): $(OBJS)
	@echo "==> Create static library $(LIBGHOST)"
	${Q}ar rcs  $(LIBGHOST) $^
endif
ifeq ($(LIBTYPE),shared)
$(LIBGHOST): $(OBJS)
	@echo "==> Create shared library $(LIBGHOST)"
	${Q}$(CC) $(CFLAGS) $(SHAREDFLAG) -o $(LIBGHOST) $^
endif

clean:
	@echo "==> Clean"
	@rm -rf $(OBJDIR)
	@rm -rf $(MODDIR)
	@rm -rf $(PLOBJDIR)
	@rm -rf $(TYPES_H)
	@rm -rf $(FORMATS_H)

distclean: clean
	@echo "==> Distclean"
	@rm -f *.a *.so
	@rm -f $(PLUGINDIR)/*.so

install: all
	@mkdir -p $(GHOSTPATH)/lib
	@mkdir -p $(GHOSTPATH)/include/ghost
	@mkdir -p $(GHOSTPATH)/lib/ghost
	@echo "==> Install library to $(GHOSTPATH)/lib/"
	@cp -f $(LIBGHOST) $(GHOSTPATH)/lib
	@echo "==> Install headers to $(GHOSTPATH)/include/ghost/"
	@cp -rf include/* $(GHOSTPATH)/include/ghost
	@echo "==> Install plugins to $(GHOSTPATH)/lib/ghost/"
	@cp -f $(PLUGINDIR)/*.so $(GHOSTPATH)/lib/ghost
	@cp -f $(PLUGINDIR)/*/src/*.cl $(GHOSTPATH)/lib/ghost
	@echo "==> Install module files to $(GHOSTPATH)/lib/ghost/"
	@cp -f $(MODDIR)/*.mod $(GHOSTPATH)/lib/ghost

uninstall: 
	@echo "==> Uninstall from $(GHOSTPATH)"
	@rm -f $(GHOSTPATH)/lib/libghost.a
	@rm -f $(GHOSTPATH)/lib/libghost.so
	@rm -rf $(GHOSTPATH)/include/ghost/
	@rm -rf $(GHOSTPATH)/lib/ghost/ 

