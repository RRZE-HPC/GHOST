What is GHOST?
==============

GHOST stands for General, Hybrid and Optimized Sparse Toolkit. It provides
functionality for computation with very large sparse or dense matrices.

Requirements
============

1. A C/C++ compiler ([Tested compilers](https://bitbucket.org/essex/ghost/wiki/Compatibility))
1. [CMake >= 2.8](http://www.cmake.org)
1. [hwloc >= 1.7](http://www.open-mpi.org/projects/hwloc)
1. A BLAS library ([Intel MKL](http://software.intel.com/en-us/intel-mkl) or [GSL](http://www.gnu.org/software/gsl/))

Optional
========

In order to use GHOST at its best you can decide to make use of the following optional dependencies:

1. An OpenMP-capable C/C++ compiler
1. MPI ([Tested versions](https://bitbucket.org/essex/ghost/wiki/Compatibility))
1. [CUDA](http://www.nvidia.com/cuda) for employing GPU computation ([Tested versions](https://bitbucket.org/essex/ghost/wiki/Compatibility))
1. [SCOTCH](http://www.labri.fr/perso/pelegrin/scotch/) for sparse matrix re-ordering ([Tested versions](https://bitbucket.org/essex/ghost/wiki/Compatibility))

Installation
============

First of all, clone the git repository:
`git clone git@bitbucket.org:essex/ghost.git
cd ghost/`

It is preferrable to perform an out-of-source build, i.e., create a build directory first:
`mkdir build
cd build/`


ccmake ..


make
make install

In order to set non-standard compilers, invoke ccmake as follows (e.g., Intel Compiler):

...
CC=icc CXX=icpc ccmake ..
...
Check the INSTALL file for installation notes.

Documentation
=============

For general documentation see the [Wiki](https://bitbucket.org/essex/ghost/wiki) pages on Bitbucket.
For API documentation switch to the build/ directory or your checked out code and type `make doc`.
You can now open build/doc/html/index.html with a web browser.
