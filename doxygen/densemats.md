Dense matrices and views
========================

A dense matrix in GHOST (`ghost_densemat_t`) can be either an actual dense matrix or a view on another dense matrix.
A view is a dense matrix which does not have its data stored in memory but only views data stored by another dense matrix.
Views can be identified by the flag `::GHOST_DENSEMAT_VIEW` in the the ghost_densemat_t::traits.

A view can be either a *dense* view or a *scattered* (in which case the flag `::GHOST_DENSEMAT_SCATTERED` is set) view.
A dense view is a view in which the data is contiguous with a certain stride and leading dimension.
Otherwise, the view is scattered.
BLAS routines can be called only with dense matrix views.

Host (CPU) Representation
-------------------------

Dense matrix data on the host is stored as an array of pointers ghost_densemat_t::val.
Each pointer points to a row for row-major matrices are a column for column-major matrices.
Hence, skipping a row for a in a row-major densemat view just means setting the pointer (analog for columns in col-major densemats).
However, if a component in the leading dimension should be skipped, the ghost_bitmap_t in ghost_densemat_t::ldmask has to be manipulated.
The address of the first entry can be obtained with ghost_densemat_valptr().

Device (GPU) Representation
---------------------------

The dense matrix data on a device is stored as a one-dimensional array ghost_densemat_t::cu_val.
For non-full views, the according bits in ghost_densemat_t::ldmask (leading dimension) or ghost_densemat_t::trmask (trailing dimension) have to be manipulated.


Example for row-major densemats
-------------------------------

Analog for column-major densemats.
We have a 4x4 source matrix A:  
~~~
x x x x  
x x x x  
x x x x  
x x x x  
~~~
  
~~~
ghost_densemat_traits_t::flags  = GHOST_DENSEMAT_DEFAULT  
ghost_densemat_t::ldmask = 1 1 1 1  
ghost_densemat_t::trmask = 1 1 1 1  
~~~

Create a view of A which skips row 0:  
~~~
. . . .  
x x x x  
x x x x  
x x x x  
~~~
  
~~~
ghost_densemat_traits_t::flags  = GHOST_DENSEMAT_VIEW  
ghost_densemat_t::ldmask = 1 1 1 1  
ghost_densemat_t::trmask = 0 1 1 1  
~~~

Create a view of A which skips row 0 and 3:  
~~~
. . . .  
x x x x  
x x x x  
. . . .  
~~~
  
~~~
ghost_densemat_traits_t::flags  = GHOST_DENSEMAT_VIEW  
ghost_densemat_t::ldmask = 1 1 1 1  
ghost_densemat_t::trmask = 0 1 1 0  
~~~

Create a view of A which skips row 1:  
~~~
x x x x  
. . . .  
x x x x  
x x x x  
~~~
  
~~~
ghost_densemat_traits_t::flags  = GHOST_DENSEMAT_VIEW|GHOST_DENSEMAT_SCATTERED  
ghost_densemat_t::ldmask = 1 1 1 1  
ghost_densemat_t::trmask = 1 0 1 1  
~~~

Create a view of A which skips column 3:  
~~~
x x x .  
x x x .  
x x x .  
x x x .  
~~~
  
~~~
ghost_densemat_traits_t::flags  = GHOST_DENSEMAT_VIEW  
ghost_densemat_t::ldmask = 1 1 1 0  
ghost_densemat_t::trmask = 1 1 1 1  
~~~

Create a view of A which skips column 3 and row 0:  
~~~
. . . .  
x x x .  
x x x .  
x x x .  
~~~
  
~~~
ghost_densemat_traits_t::flags  = GHOST_DENSEMAT_VIEW  
ghost_densemat_t::ldmask = 1 1 1 0  
ghost_densemat_t::trmask = 0 1 1 1  
~~~

Create a view of A which skips column 1 and row 0:  
~~~
. . . .  
x . x x  
x . x x  
x . x x  
~~~
  
~~~
ghost_densemat_traits_t::flags  = GHOST_DENSEMAT_VIEW|GHOST_DENSEMAT_SCATTERED  
ghost_densemat_t::ldmask = 1 0 1 1  
ghost_densemat_t::trmask = 0 1 1 1  
~~~

Create a view of A which skips column 3 and rows 0 and 2:  
~~~
. . . .  
x x x .  
. . . .  
x x x .  
~~~
  
~~~
ghost_densemat_traits_t::flags  = GHOST_DENSEMAT_VIEW|GHOST_DENSEMAT_SCATTERED  
ghost_densemat_t::ldmask = 1 1 1 0  
ghost_densemat_t::trmask = 0 1 0 1  
~~~
