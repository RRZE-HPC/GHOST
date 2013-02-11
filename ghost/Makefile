include config.mk
include include_$(strip $(COMPILER)).mk

#.NOTPARALLEL:

.PHONY: clean distclean install uninstall all header 

VPATH      = ./src/ ./src/cl/ ./src/mpi/ ./src/cu/ 
OBJDIR     = ./obj
C_OBJDIR   = ./obj/c
D_OBJDIR   = ./obj/d
S_OBJDIR   = ./obj/s
Z_OBJDIR   = ./obj/z
PLUGINDIR  = ./plugins
TYPES_H    = ./include/ghost_types_gen.h
FORMATS_H  = ./include/ghost_spmformats.h
IPATH     += -I./include/

OBJS	=	$(COBJS) $(FOBJS) 
FORMATS = $(notdir $(wildcard $(PLUGINDIR)/*/include/*.h))

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

ifeq ($(strip $(CUDA)),1)
include include_NVCC.mk
MAKROS+= -DCUDA
IPATH += -I$(CUDA_HOME)/include -I./include/cu
OBJS  += $(CUOBJS)
endif

ifeq ($(strip $(LIKWID)),1)
MAKROS+= -DLIKWID_PERFMON
IPATH += -I$(strip $(LIKWIDPATH))/include
endif

ifeq ($(strip $(LONGIDX)),1)
TYPES:= $(TYPES)"\#define GHOST_LONGIDX \n"
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

ifeq ($(strip $(CUDA)),1)
PLUGINOBJS += $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.cu,%.cu.o, $(wildcard plugins/*/src/*.cu))))
endif
COBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/*.c))))
CLOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cl/*.c))))
CUOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cu/*.c))))
MPIOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/mpi/*.c))))
FOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.f,%.o, $(wildcard src/*.f))))

all:  $(FORMATS_H) $(OBJDIR) $(C_OBJDIR) $(D_OBJDIR) $(S_OBJDIR) $(Z_OBJDIR) $(PLUGINS) $(LIBGHOST)

$(PLUGINS): $(PLUGINOBJS)
	$(eval dt := $(shell echo $(notdir $@) | cut -c 1))
	$(eval pl := $(shell echo $(basename $(notdir $@)) | cut -c 3-))
	$(eval objs := $(shell find $(PLUGINDIR)/$(pl)/src/ -name "*.c" -printf "./obj/$(dt)/%f " | sed 's/\.c/\.o/g'))
#	echo $(wildcard $(PLUGINDIR)/$(pl)/src/*.c)
#	$(eval objs := $(addprefix ./plugins/$(dt)_objs/,$(notdir $(patsubst %.c, %.o, $(wildcard $(PLUGINDIR)/$(pl)/src/*.c)))))
#ifeq ($(strip $(CUDA)),1)
#	$(eval objs += $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.cu, %.cu.o, $(wildcard $(basename $@)/src/c_*.cu)))))
#endif
	@echo "==> Create plugin $(objs) -> $@"
	@$(CC) $(CFLAGS) $(SHAREDFLAG) ${MAKROS} ${IPATH} -o $@ $(objs)
#	@echo "==> Clean up specialized files"
#	@rm -r $(PLUGINDIR)/$(notdir $(basename $@))
	
$(C_OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile complex float variant $< -> $@"
	@$(CC) $(CFLAGS) ${MAKROS} -DGHOST_MAT_COMPLEX -DGHOST_MAT_SP -DGHOST_VEC_COMPLEX -DGHOST_VEC_SP ${IPATH} -o $@ -c $<

$(D_OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile double variant $< -> $@"
	@$(CC) $(CFLAGS) ${MAKROS} -DGHOST_MAT_DP -DGHOST_VEC_DP ${IPATH} -o $@ -c $<

$(S_OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile single variant $< -> $@"
	@$(CC) $(CFLAGS) ${MAKROS} -DGHOST_MAT_SP -DGHOST_VEC_SP ${IPATH} -o $@ -c $<

$(Z_OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile complex double variant $< -> $@"
	@$(CC) $(CFLAGS) ${MAKROS} -DGHOST_MAT_COMPLEX -DGHOST_MAT_DP -DGHOST_VEC_COMPLEX -DGHOST_VEC_DP ${IPATH} -o $@ -c $<

$(OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile $< -> $@"
	@$(CC) $(CFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

ifeq ($(strip $(CUDA)),1)
$(OBJDIR)/%.cu.o: %.cu config.mk
	@echo "==> Compile $< -> $@"
	@$(NVCC) $(NVCCFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<
endif


#$(TYPES_H): config.mk 
#	@echo "==> Generate types header"
#	@echo "#ifndef __GHOST_TYPES_GEN_H__" > $(TYPES_H)
#	@echo -e "#define __GHOST_TYPES_GEN_H__\n" >> $(TYPES_H)
#	@echo -e $(TYPES) >> $(TYPES_H)
#	@echo -e "\n#endif" >> $(TYPES_H)

$(FORMATS_H): config.mk
	@echo "==> Generate formats header"
	@echo "#ifndef __GHOST_SPMFORMATS_H__" > $(FORMATS_H)
	@echo -e "#define __GHOST_SPMFORMATS_H__\n" >> $(FORMATS_H)
	@(for format in $(FORMATS); do \
	echo "#include <$$format>" >> $(FORMATS_H); done)
	@echo -e "\n#endif" >> $(FORMATS_H)

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
	@rm -rf $(OBJDIR)
	@rm -rf $(PLOBJDIR)
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
	@cp -f $(PLUGINDIR)/*/src/*.cl $(strip $(GHOSTPATH))/lib/ghost

uninstall: 
	@echo "==> Uninstall from $(strip $(GHOSTPATH))"
	@rm -f $(strip $(GHOSTPATH))/lib/libghost.a
	@rm -f $(strip $(GHOSTPATH))/lib/libghost.so
	@rm -rf $(strip $(GHOSTPATH))/include/ghost/
	@rm -rf $(strip $(GHOSTPATH))/lib/ghost/ 

