What is GHOST?
==============

GHOST stands for General, Hybrid and Optimized Sparse Toolkit. It provides
functionality for computation with very large sparse or dense matrices.

Requirements
============

1. A C/C++ compiler ([Tested compilers](https://bitbucket.org/essex/ghost/wiki/Compatibility))
1. [CMake >= 2.8](http://www.cmake.org)
1. [hwloc >= 1.7](http://www.open-mpi.org/projects/hwloc) ([Install notes](https://bitbucket.org/essex/ghost/wiki/Dependencies))
1. A BLAS library ([Intel MKL](http://software.intel.com/en-us/intel-mkl) or [GSL](http://www.gnu.org/software/gsl/))

Optional
========

In order to use GHOST at its best you can decide to make use of the following optional dependencies:

1. An OpenMP-capable C/C++ compiler
1. MPI ([Tested versions](https://bitbucket.org/essex/ghost/wiki/Compatibility))
1. [CUDA](http://www.nvidia.com/cuda) for employing GPU computation ([Tested versions](https://bitbucket.org/essex/ghost/wiki/Compatibility))
1. [SCOTCH](http://www.labri.fr/perso/pelegrin/scotch/) for sparse matrix re-ordering ([Tested versions](https://bitbucket.org/essex/ghost/wiki/Compatibility), [Install notes](https://bitbucket.org/essex/ghost/wiki/Dependencies))

Installation
============

Preparation
-----------

First of all, clone the git repository:

`git clone git@bitbucket.org:essex/ghost.git && cd ghost/`

It is preferrable to perform an out-of-source build, i.e., create a build directory first:

`mkdir build && cd build/`

Quick install
-------------

To do a quick build with the system compilers and only MPI and OpenMP enabled:

`cmake .. -DCMAKE_INSTALL_PREFIX=<where-to-install> && make install`

Once the Makefile is present you can type

`make && make install`

For interactive specification of build options and variables, use ccmake to configure and generate a Makefile:

`ccmake ..`

If you do not want to use the system compilers, invoke (c)cmake as follows (e.g., Intel Compilers):

`CC=icc CXX=icpc ccmake ..`

Further options
---------------

The build system will try to find a Cblas header `*cblas.h` in default locations.
In some cases (if a MKL or GSL Cblas is detected), a hint about the necessary BLAS libraries will be created for the future build of executables which link to GHOST.
If finding a Cblas header fails, or if you want to overwrite the found Cblas header, you can pass `-DCBLAS_INCLUDE_DIR=<dir-of-cblas-header>` to (c)cmake or set the value in ccmake.
If the detected Cblas is neither MKL or GSL, the variable `BLAS_LIBRARIES` has to be set manually before linking.

The same applies for hwloc and the variables `HWLOC_INCLUDE_DIR` and `HWLOC_LIBRARIES`.

Documentation
=============

See the [Wiki](https://bitbucket.org/essex/ghost/wiki) pages on Bitbucket and the [Doxygen](https://grid.rrze.uni-erlangen.de/~unrza317) pages.

To create your own Doxygen documentation switch to the `build/` directory or your checked out code and type `make doc`.
You can now open `doc/html/index.html` with a web browser.
