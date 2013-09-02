README {#mainpage}
======

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
right part of the matrix is to be stored. The number of entries and row 
pointers always relate to the _stored_ matrix and not to the full one (if 
symmetric). Due to this constraint, the stored matrix is always a valid one.


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

The tasking functionality is based on a number of task queues and a thread pool.
There may be multiple task queues in order to control the affinty of task
execution.  In a regular use case, one would have one task queue per locality
domain; and a locality domain may be equal to a NUMA domain. 

Initialization
--------------

Both the task queues and the thread pool have to be initialized at some point.
This can either be done inside the application. By this, the user has full
control over the number of threads and the number of task queues.
If initialization is not done at this point, it will be detected in
`ghost_task_init()` (which is always the first tasking-related function being
called) and done there with reasonable default values.


Thread lifecycle
----------------

A pthread is created for each thread of the thread pool in `ghost_thpool_init()`.
Each of the threads has `thread_main()` as the starting routine where it runs 
in an infinite loop.
In the main loop, the threads are waiting on a semaphore which counts the number
of tasks in all queues. As soon as a task gets added to a queue, the first
thread returning from wait will have the chance to execute this task.


Task selection
--------------

