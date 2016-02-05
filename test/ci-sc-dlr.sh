#!/usr/bin/env bash
set -e

## default options and declarations
# kernel lib
PRGENV="gcc-4.9.2-openmpi" # intel-13.0.1-mpich gcc-4.8.2-openmpi
BUILD_TYPE=Release
INSTALL_PREFIX=../../
VECT_EXT="native" # none SSE AVX AVX2 CUDA

# list of modules to load
MODULES_BASIC="cmake ccache cppcheck lapack gsl"

## parse command line arguments
usage() { echo "Usage: $0 [-e <PrgEnv/module-string>] [-b <Release|Debug|...>] [-v <native|none|SSE|AVX|AVX2|CUDA>]" 1>&2; 
exit 1; }

while getopts "e:b:v:p:h" o; do
    case "${o}" in
        e)
            PRGENV=${OPTARG}
            ;;
        b)
            BUILD_TYPE=${OPTARG}
            ;;
        v)
            VECT_EXT=${OPTARG}
            ;;
        p)
            INSTALL_PREFIX=${OPTARG}
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

echo "Options: PRGENV=${PRGENV}, BUILD_TYPE=${BUILD_TYPE}"

## prepare system for compilation
# configure modulesystem
export MODULEPATH=/tools/modulesystem/modulefiles
module() { eval `/usr/bin/modulecmd bash $*`; }

# load modules
module load "PrgEnv/$PRGENV"
# set compiler names
set -- $(mpicc -show)
export CC=$1
set -- $(mpicxx -show)
export CXX=$1
set -- $(mpif90 -show)
export FC=$1

echo "compilers: CC=$CC, CXX=$CXX, FC=$FC"

for m in $MODULES_BASIC; do module load $m; done
if [ "${VECT_EXT}" = "CUDA" ]; then
  module load cuda
fi

module list

# use ccache to speed up build
#if [[ "$PRGENV" = "gcc"* ]]; then
#  export FC="ccache gfortran" CC="ccache gcc" CXX="ccache g++"
#elif [[ "$PRGENV" = "intel"* ]]; then
#  export FC=ifort CC=icc CXX=icpc
#fi

# "gcc -fsanitize=address" requires this
ulimit -v unlimited


# setup which optimized kernels should be built
if [ "${VECT_EXT}" = "none" ]; then
  SELL_CS="33"
  BLOCKSZ="13"
else
  SELL_CS="1,4,16,32"
  BLOCKSZ="1,2,4,8"
fi;

# setup vector extension flags
VECT_FLAGS=""
if [ "${VECT_EXT}" = "CUDA" ]; then
  VECT_FLAGS="${VECT_FLAGS} -DUSE_CUDA=On"
elif [ "${VECT_EXT}" != "native" ]; then
  VECT_FLAGS="${VECT_FLAGS} -DUSE_CUDA=Off"
  if [[ "${VECT_EXT}" =~ SSE|AVX|AVX2 ]]; then
    VECT_FLAGS="${VECT_FLAGS} -DGHOST_BUILD_SSE=On"
  else
    VECT_FLAGS="${VECT_FLAGS} -DGHOST_BUILD_SSE=Off"
  fi
  if [[ "${VECT_EXT}" =~ AVX|AVX2 ]]; then
    VECT_FLAGS="${VECT_FLAGS} -DGHOST_BUILD_AVX=On"
  else
    VECT_FLAGS="${VECT_FLAGS} -DGHOST_BUILD_AVX=Off"
  fi
  if [[ "${VECT_EXT}" =~ AVX2 ]]; then
    VECT_FLAGS="${VECT_FLAGS} -DGHOST_BUILD_AVX2=On"
  else
    VECT_FLAGS="${VECT_FLAGS} -DGHOST_BUILD_AVX2=Off"
  fi
fi

error=0
# build and install
mkdir build_${PRGENV}_${BUILD_TYPE}_${VECT_EXT}       || exit 1
cd build_${PRGENV}_${BUILD_TYPE}_${VECT_EXT}          || exit 1
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX/install-${PRGENV}-${BUILD_TYPE}-${VECT_EXT} \
-DCFG_BLOCKVECTOR_SIZES=${BLOCKSZ} -DCFG_SELL_CHUNKHEIGHTS=${SELL_CS} \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_SHARED_LIBS=ON ${VECT_FLAGS} ..              || error=1

if [[ "${BUILD_TYPE}" =~ *Rel* ]]; then
  make doc                                  || error=1
fi
make -j 24 || make || exit 1
make check                                || error=1
make install                              || error=1
cd ..                                     || exit 1

#return error code
exit $error
