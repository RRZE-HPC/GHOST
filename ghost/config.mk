COMPILER=ICC    # one of: ICC, GCC, XLC
LIBTYPE=shared  # static or shared
#MATDATA=d       # s = float, d = double, c = complex float, z = complex double
#VECDATA=d       # s = float, d = double, c = complex float, z = complex double
LONGIDX=0       # 0/1 = indices are 32/64 bit integers
MPI=1           # 0 = no MPI support        | 1 = with MPI support
VSX=0           # 0 = no VSX kernels        | 1 = build VSX kernels
MIC=0           # 0 = no MIC kernels        | 1 = build MIC kernels
AVX=0           # 0 = no AVX kernels        | 1 = build AVX kernels
SSE=0           # 0 = no SSE kernels        | 1 = build SSE kernels
OPENCL=0        # 0 = no OpenCL kernels     | 1 = build OpenCL kernels
CUDA=0          # 0 = no CUDA kernels       | 1 = build CUDA kernels
LIKWID=0        # 0 = no instrumentation    | 1 = Likwid Marker API instrumentation
DEBUG=1         # 0 = no DEBUG output       | >0 = level of DEBUG output

GHOSTPATH=$(HOME)/local      # where to install ghost
LIKWIDPATH=$(HOME)/local     # path to LIKWID installation (if enabled)
