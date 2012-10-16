COMPILER=ICC           # ICC or GCC
LIBTYPE=shared         # static or shared
DOUBLE=1               # 0 = single precision data | 1 = double precision data
COMPLEX=0              # 0 = real data             | 1 = complex data
MPI=1                  # 0 = no MPI support        | 1 = with MPI support
MIC=0                  # 0 = no MIC kernels        | 1 = build MIC kernels
OPENCL=0               # 0 = standard CPU kernels  | 1 = OpenCL (GPU) kernels
LIKWID_MARKER=0        # 0 = no calls              | 1 = Likwid Marker API calls
LIKWID_MARKER_FINE=0   # 0 = few calls             | 1 = many calls 
DEBUG=0                # 0 = no DEBUG output       | >0 = level of DEBUG output

INSTDIR=/home/hpc/h021z/di56xuq2/app/libspmvm        # where to install LibSpMVM
