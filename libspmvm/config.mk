DOUBLE=1               # 0 = single precision data | 1 = double precision data
COMPLEX=0              # 0 = real data             | 1 = complex data
OPENCL=0               # 0 = standard CPU kernels  | 1 = OpenCL (GPU) kernels
LIKWID_MARKER=0        # 0 = no calls              | 1 = Likwid Marker API calls
LIKWID_MARKER_FINE=0   # 0 = few calls             | 1 = many calls 
DEBUG=0                # 0 = no DEBUG output       | >0 = level of DEBUG output

INSTDIR=/home/hpc/h021z/di56xuq2/app/libspmvm        # where to install LibSpMVM

CC	= mpicc           # the C compiler
CFLAGS  = -O3 -fno-alias -openmp -Wall -Werror-all -Wremarks -Wcheck -diag-disable 981 

FC	= ifort           # the Fortran compiler
FFLAGS  = -O3 -nogen-interface -fno-alias -cpp -warn all -openmp 


