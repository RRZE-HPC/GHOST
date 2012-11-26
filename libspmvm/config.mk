COMPILER=ICC    # ICC or GCC
LIBTYPE=shared  # static or shared
MATDATA=d       # s = float, d = double, c = complex float, z = complex double
VECDATA=d       # s = float, d = double, c = complex float, z = complex double
MPI=0           # 0 = no MPI support        | 1 = with MPI support
MIC=0           # 0 = no MIC kernels        | 1 = build MIC kernels
AVX=0           # 0 = no AVX kernels        | 1 = build AVX kernels
SSE=0           # 0 = no SSE kernels        | 1 = build SSE kernels
OPENCL=0        # 0 = standard CPU kernels  | 1 = OpenCL (GPU) kernels
LIKWID=0        # 0 = no instrumentation    | 1 = Likwid Marker API instrumentation
DEBUG=0         # 0 = no DEBUG output       | >0 = level of DEBUG output

INSTDIR=$(HOME)/app/libspmvm        # where to install LibSpMVM
