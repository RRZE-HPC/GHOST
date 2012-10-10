COMPILER=ICC           # ICC or GCC
DOUBLE=1               # 0 = single precision data | 1 = double precision data
COMPLEX=0              # 0 = real data             | 1 = complex data
MPI=0                  # 0 = no MPI support        | 1 = with MPI support
OPENCL=0               # 0 = standard CPU kernels  | 1 = OpenCL (GPU) kernels
LIKWID_MARKER=0        # 0 = no calls              | 1 = Likwid Marker API calls
LIKWID_MARKER_FINE=0   # 0 = few calls             | 1 = many calls 
DEBUG=0                # 0 = no DEBUG output       | >0 = level of DEBUG output

INSTDIR=$(HOME)/app/libspmvm        # where to install LibSpMVM

CL_INC=$(CUDA_HOME)/include   # where to find OpenCL headers (necessary if enabled)
LIKWID_INC=$(HOME)/app/likwid/include
