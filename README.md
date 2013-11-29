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

1. An OpenMP-capable C/C++ compiler
1. MPI ([Tested versions](https://bitbucket.org/essex/ghost/wiki/Compatibility))
1. CUDA ([Tested versions](https://bitbucket.org/essex/ghost/wiki/Compatibility))

Installation
============

Check the INSTALL file for installation notes.

Documentation
=============

For general documentation see the [Wiki](https://bitbucket.org/essex/ghost/wiki) pages on Bitbucket.
For API documentation switch to the build/ directory or your checked out code and type `make doc`.
You can now open build/doc/html/index.html with a web browser.
