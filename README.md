README
======

GHOST stands for General, Hybrid and Optimized Sparse Toolkit. It provides
functionality for computation with very large sparse or dense matrices.

Requirements
============

1. hwloc
2. MKL (more supported BLAS libraries to come)


Supported compilers
-------------------

GHOST is tested with a range of compilers.

1. Intel

	This is the recommended compiler suite which showed to yield best performance 
	with GHOST. A specific feature of it is the presence of the library function 
	`kmp_setblocktime()` which allows to set the OpenMP block time at run-time which
	is crucial in order to get the best performance.

2. GNU
	
	In order to improve the performance with GCC, it is recommended to set the
	environment variable `GOMP_WAIT_POLICY="PASSIVE"` and in many cases
	`OMP_SCHEDULE="STATIC"` (the default in GCC 4.8.1 is `"DYNAMIC"`).


Data types
==========

In principle there are two main types of data, namely **sparse matrices** and
**dense matrices** / **vectors** / **multi-vectors**:
	1. Sparse matrices



I/O
===

Input data (i.e., matrices and vectors) for GHOST have to be present in special binary data formats.

Sparse matrices
---------------

Sparse matrices are stored in CRS format. The binary file is assembled as follows:

1. 4 bytes: Indicator whether data is stored in little or big endian format.

	* 0: little endian
	* else: big endian

	All the following data will be assumed to have this endianess.

2. 4 bytes: Indicator for version of data format. (this version: 1)

3. 4 bytes: Indicator for base of indices (row pointers and column indices).

	* 0: indices are 0-based (as in C)
	* 1: indices are 1-based (as in Fortran)

	Any other base except 0 and 1 is not supported at this time.
	
4. 4 bytes: Symmetry information bit field; multiple may be combined.

    * 1: general
    * 2: symmetric
    * 4: skew-symmetric
    * 8: hermitian

	Example: Matrix is hermitian and skew-symmetric: 12

5. 4 bytes: Data type bit field; multiple may be combined.

	* 1: float
	* 2: double 
	* 4: real
	* 8: complex

	Example: Matrix has complex float values: 9

6. 8 bytes: An integer telling the number of rows (N) of the matrix

7. 8 bytes: An integer telling the number of columns (M) of the matrix

8. 8 bytes: An integer telling the number of nonzero entries (E) of the matrix

9. (N+1)*8 bytes: The row pointers (integer) of the CRS matrix
	
10. E*8 bytes: The columns indices of the matrix entries

12. E*sizeof(datatype) bytes: The values of the matrix entries
	
> **NOTE:** In the case of any set symmetry flag except general, only the 
> right part of the matrix is to be stored. The number of entries and row 
> pointers always relate to the _stored_ matrix and not to the full one (if 
> symmetric). Due to this constraint, the stored matrix is always a valid one.


Dense matrices/vectors
----------------------

Dense matrices, vectors and multi-vectors (groups of vectors) are represented 
by the same data structure.

1. 4 bytes: Indicator whether data is stored in little or big endian format.

	* 0: little endian
	* else: big endian

	All the following data will be assumed to have this endianess.

2. 4 bytes: Indicator for version of data format. (this version: 1)

3. 4 bytes: Indicator for the storage order of the values.

	* 0: column-major
	* 1: row-major

	For one-dimensional vectors, this indicator has no influence.
	
4. 4 bytes: Data type bit field; multiple may be combined.

	* 1: float
	* 2: double 
	* 4: real
	* 8: complex

	Example: Matrix has complex float values: 9

5. 8 bytes: An integer telling the number of rows (N) of the vector/matrix

6. 8 bytes: An integer telling the number of columns (M) of the vector/matrix

7. N*M*sizeof(datatype) bytes: The vector/matrix entries


Tasking
=======

The tasking functionality is a key feature of GHOST. It is designed to be as
general as possible and to allow the use of (a-)synchronous tasks in many
situations.
The implementation is based on a task queue and a thread pool.
There is a single task queue which holds all tasks, regardless of their
properties such as hardware affinity.

Besides, there is a single thread pool which contains all threads which may
ever be used by all task. The number of threads can be specified by the
arguments of `ghost_thpool_init`. Typical scenarios may be to have one GHOST thread per physical
core or one GHOST thread per hardware thread (i.e., using the hardware's SMT
capabilities).
Affinity control of the GHOST threads is done by means of the
[hwloc](http://www.open-mpi.org/projects/hwloc/) library.

Initialization
--------------

Both the task queues and the thread pool have to be initialized at some point.
This can either be done inside the application. By this, the user has full
control over the number of threads and the number of task queues.
If initialization is not done by the user, it will be detected in
`ghost_task_init()`, i.e., at the time when the first task is being created, and
done there with reasonable default values.


Thread lifecycle
----------------

A pthread is created for each thread of the thread pool in `ghost_thpool_init()`.
Each of the threads has `thread_main()` as the starting routine where it runs 
in an infinite loop.
In this loop, the threads are waiting on a semaphore which counts the number
of tasks in all queues. As soon as a task gets added to a queue, the first
thread returning from `sem_wait` will have the chance to execute this task.
If, for any reason, the thread cannot execute the task (or any other task in the
queue), the above-mentioned semaphore gets increased by one and the thread
re-enters it's main loop.
Once the task queues are empty and subject to be killed (i.e., at termination of
the application), the global variable `#killed` is set to one and the threads
break out of the infinite loop.


Task types
----------

A GHOST task shall be initialized via `ghost_task_init()`, given a number of
parameters. For detailed documentation see `ghost_task_t`.

There are several types of tasks which can be distinguished by the value of
their `flags` variable. They can be combined by bitwise OR-ing the flags.

1. **GHOST_TASK_DEFAULT** identifies default tasks.
They are added to a single task queue. However, if none of the threads in this
locality domain takes the task out of the queue it may be executed by a thread
running in a different locality domain.

2. **GHOST_TASK_LD_STRICT** bounds a task to a locality domain. A
strict LD task is guaranteed to be executed only by threads running in the given
locality domain.

3. **GHOST_TASK_PRIO_HIGH** assigns high priority to a task.
It will be added to the head of the given queue(s) and thus they are probably
chosen earlier than the other tasks in the queues.

4. **GHOST_TASK_USE_PARENTS** is the flag to use in case a task is being 
added from within another task. When this flag is set, the child task can use
all cores which are actually reserved for the adding task.


Task selection
--------------


