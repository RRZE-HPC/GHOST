COMPILER=ICC    # one of: ICC, GCC, XLC
LIBTYPE=shared  # static or shared (default)
LONGIDX=0       # 0/1 = indices are 32/64 bit integers
FORTRAN=0       # 0/1 = build without/with Fortran interface
OPENMP=1        # 0/1 = build without/with OpenMP
MPI=0           # 0 = no MPI support        | 1 = with MPI support
VSX=0           # 0 = no VSX kernels        | 1 = build VSX kernels
MIC=0           # 0 = no MIC | 1 = compile for MIC | 2 = w/ intrinsics
AVX=0           # 0 = no AVX | 1 = compile for AVX | 2 = w/ intrinsics
SSE=1           # 0 = no SSE | 1 = compile for SSE | 2 = w/ intrinsics
OPENCL=0        # 0 = no OpenCL kernels     | 1 = build OpenCL kernels
CUDA=0          # 0 = no CUDA kernels       | 1 = build CUDA kernels
LIKWID=0        # 0 = no instrumentation    | 1 = Likwid Marker API instrumentation
ITAC=0          # 0 = no instrumentation    | 1 = VT instrumentation
DEBUG=0        # -1 = only debug compilation | >0 = level of DEBUG output

GHOSTPATH=$(HOME)/local      # where to install ghost
LIKWIDPATH=/apps/likwid/likwid-3.1beta/     # path to LIKWID installation (if enabled)


##### DO NOT EDIT BELOW

COMPILER:=$(strip $(COMPILER))#
LIBTYPE:=$(strip $(LIBTYPE))#
LONGIDX:=$(strip $(LONGIDX))#
FORTRAN:=$(strip $(FORTRAN))#
OPENMP:=$(strip $(OPENMP))#
MPI:=$(strip $(MPI))#
VSX:=$(strip $(VSX))#
MIC:=$(strip $(MIC))#
AVX:=$(strip $(AVX))#
SSE:=$(strip $(SSE))#
OPENCL:=$(strip $(OPENCL))#
CUDA:=$(strip $(CUDA))#
LIKWID:=$(strip $(LIKWID))#
ITAC:=$(strip $(ITAC))#
DEBUG:=$(strip $(DEBUG))#
GHOSTPATH:=$(strip $(GHOSTPATH))#
LIKWIDPATH:=$(strip $(LIKWIDPATH))#
