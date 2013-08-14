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

.SECONDARY: $(FMODULES)
.PHONY: clean distclean install uninstall all header 

Q       = @  
VPATH   = ./src/ ./src/fortran/ ./src/fortran/modules/ ./src/cl/ ./src/mpi/ ./src/cu/ ./src/likwid/ 
OBJDIR  = ./obj
MODDIR  = ./mod
IPATH  += -I./include/ -I./include/fortran/ -I./include/likwid/
OBJS	= $(LIKWOBJS) $(COBJS) $(CPPOBJS) $(FOBJS) 
FFLAGS += $(FMODFLAG) $(MODDIR)

ifeq ($(MPI),1)
MAKROS+= -DGHOST_MPI
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

ifeq ($(strip $(ITAC)),1)
MAKROS+= -DVT
IPATH += -I$(ITC_INC)
CFLAGS += -trace
endif

ifeq ($(LIKWID),1)
MAKROS+= -DLIKWID_PERFMON
IPATH += -I$(LIKWIDPATH)/include
endif

ifeq ($(LONGIDX),1)
MAKROS+= -DLONGIDX
endif

ifeq ($(LIBTYPE),static)
SUFFIX=a
endif
ifeq ($(LIBTYPE),shared)
SUFFIX=so
endif

LIBGHOST=libghost.$(SUFFIX)

MAKROS+= -DDEBUG=$(DEBUG)
MAKROS+= -DCLKERNELPATH=\"$(GHOSTPATH)/lib/ghost/\"
MAKROS+= -DHEADERPATH=\"$(GHOSTPATH)/include/ghost/\"

COBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/*.c))))
LIKWOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/likwid/*.c))))
CPPOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.cpp,%.o, $(wildcard src/*.cpp))))
CLOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cl/*.c))))
CUOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/cu/*.c))))
CUOBJS += $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.cu,%.cu.o, $(wildcard src/cu/*.cu))))
MPIOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.c,%.o, $(wildcard src/mpi/*.c))))
FOBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.F90,%.o, $(wildcard src/fortran/*.F90))))
FMODULES = $(addprefix $(MODDIR)/,$(notdir $(patsubst %.F90,%.mod, $(wildcard src/fortran/modules/*.F90))))

all: $(OBJDIR) $(MODDIR) $(LIBGHOST)

$(OBJDIR)/%.o: %.c config.mk
	@echo "==> Compile $< -> $@"
	${Q}$(CC) $(CFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

$(OBJDIR)/%.o: %.cpp config.mk
	@echo "==> Compile $< -> $@"
	${Q}${CPPC} $(CPPFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

$(OBJDIR)/%.f90: %.F90 config.mk $(FMODULES) 
	@echo "==> Preprocess $< -> $@"
	${Q}${PP} -o $@ $<

$(OBJDIR)/%.o: $(OBJDIR)/%.f90 config.mk $(FMODULES) 
	@echo "==> Compile $< -> $@"
	${Q}$(FC) $(FFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

$(OBJDIR)/%.cu.o: %.cu config.mk 
	@echo "==> Compile $< -> $@"
	${Q}$(NVCC) $(NVCCFLAGS) ${MAKROS} ${IPATH} -o $@ -c $<

$(MODDIR)/%.mod: %.F90 config.mk
	$(eval target := $(addprefix $(OBJDIR)/,$(patsubst %.mod,%.o,$(notdir $@))))
	@echo "==> Compile $< -> $(target)"
	${Q}$(FC) $(FFLAGS) ${MAKROS} ${IPATH} -o $(target) -c $<

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
	${Q}${CC} $(CFLAGS) $(SHAREDFLAG) -o $(LIBGHOST) $^
endif

clean:
	@echo "==> Clean"
	@rm -rf $(OBJDIR)
	@rm -rf $(MODDIR)

distclean: clean
	@echo "==> Distclean"
	@rm -f *.a *.so

install: all
	@mkdir -p $(GHOSTPATH)/lib/
	@mkdir -p $(GHOSTPATH)/include/ghost/
	@mkdir -p $(GHOSTPATH)/lib/ghost/
	@echo "==> Install Library to $(GHOSTPATH)/lib/"
	@cp -f $(LIBGHOST) $(GHOSTPATH)/lib/
	@echo "==> Install Headers to $(GHOSTPATH)/include/ghost/"
	@cp -rf include/* $(GHOSTPATH)/include/ghost/
	@echo "==> Install OpenCL kernels to $(GHOSTPATH)/lib/ghost/"
	@cp -rf src/cl/*.cl $(GHOSTPATH)/lib/ghost/
	@echo "==> Install Fortran module files to $(GHOSTPATH)/lib/ghost/"
	@cp -f $(MODDIR)/*.mod $(GHOSTPATH)/lib/ghost/

uninstall: 
	@echo "==> Uninstall from $(GHOSTPATH)"
	@rm -f $(GHOSTPATH)/lib/libghost.a
	@rm -f $(GHOSTPATH)/lib/libghost.so
	@rm -rf $(GHOSTPATH)/include/ghost/
	@rm -rf $(GHOSTPATH)/lib/ghost/ 

