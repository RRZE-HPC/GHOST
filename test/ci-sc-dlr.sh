#!/usr/bin/env bash
set -e

## default options and declarations
# kernel lib
PRGENV="gcc-4.9.2-openmpi" # intel-13.0.1-mpich gcc-4.8.2-openmpi
BUILD_TYPE=Release

# list of modules to load
MODULES_BASIC="cmake ccache cppcheck lapack gsl/gsl-1.16/sled11.x86_64.gcc-4.8.2.release"

## parse command line arguments
usage() { echo "Usage: $0 [-e <PrgEnv/module-string>] [-b <Release|Debug|...>]" 1>&2; 
exit 1; }

while getopts "e:b:h" o; do
    case "${o}" in
        e)
            PRGENV=${OPTARG}
            ;;
        b)
            BUILD_TYPE=${OPTARG}
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
# by default use something not tested in PHIST (can't leave the strings empty)
SELL_CS="33"
BLOCKSZ="13"
if [[ "${BUILD_TYPE}" = *"Rel"* ]]; then
  SELL_CS="1,4,16,32"
  BLOCKSZ="1,2,4,8"
fi;

error=0
# build and install
mkdir build_${PRGENV}_${BUILD_TYPE}       || exit 1
cd build_${PRGENV}_${BUILD_TYPE}          || exit 1
cmake -DCMAKE_INSTALL_PREFIX=../../install-${PRGENV}-${BUILD_TYPE} \
-DCFG_BLOCKVECTOR_SIZES=${BLOCKSZ} -DCFG_SELL_CHUNKHEIGHTS=${SELL_CS} \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_SHARED_LIBS=ON ..              || error=1

if [[ "${BUILD_TYPE}" = *"Rel"* ]]; then
  make doc                                  || error=1
fi
make -j 6 || make || exit 1
make check                                || error=1
make install                              || error=1
cd ..                                     || exit 1

#return error code
exit $error
