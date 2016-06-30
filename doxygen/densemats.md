Dense matrices and views
========================

A dense matrix in GHOST (`ghost_densemat`) can be either an actual dense matrix or a view of either another dense matrix or plain data.
A view is a dense matrix which does not have its data stored in memory but only views data stored elsewhere
Views can be identified by the flag `::GHOST_DENSEMAT_VIEW` in the the ghost_densemat::traits.

A view can be either a *compact* view or a *scattered* (in which case the one of the flags `::GHOST_DENSEMAT_SCATTERED_LD` or `::GHOST_DENSEMAT_SCATTERED_TR` is set) view.
A compact view is a view in which the data is contiguous with a certain stride and leading dimension.
Otherwise, the view is scattered, where we distinguish between scattered views in leading (`::GHOST_DENSEMAT_SCATTERED_LD`) or trailing (`::GHOST_DENSEMAT_SCATTERED_TR`) dimension.
Some function are not implemented natively for scattered views.
If this is the case, a warning will be printed and a temporary densemat will be created for this function.
Afterwards, the scattered view will be updated with the result data and the temporary densemat gets destroyed.

The dense matrix data is stored as a one-dimensional array ghost_densemat::val (or ghost_densemat::cu_val for GPU densemats).
For scattered views, the according bits in ghost_densemat::rowmask or ghost_densemat::colmask have to be manipulated.


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
ghost_densematraits_t::flags  = GHOST_DENSEMAT_DEFAULT  
ghost_densemat::rowmask = NULL  
ghost_densemat::colmask = NULL  
~~~

Create a view of A which skips row 0:  
~~~
. . . .  
x x x x  
x x x x  
x x x x  
~~~
  
~~~
ghost_densematraits_t::flags  = GHOST_DENSEMAT_VIEW  
ghost_densemat::rowmask = NULL 
ghost_densemat::colmask = NULL  
~~~

Create a view of A which skips row 0 and 3:  
~~~
. . . .  
x x x x  
x x x x  
. . . .  
~~~
  
~~~
ghost_densematraits_t::flags  = GHOST_DENSEMAT_VIEW  
ghost_densemat::rowmask = NULL 
ghost_densemat::colmask = NULL  
~~~

Create a view of A which skips row 1:  
~~~
x x x x  
. . . .  
x x x x  
x x x x  
~~~
  
~~~
ghost_densematraits_t::flags  = GHOST_DENSEMAT_VIEW|GHOST_DENSEMAT_SCATTERED_TR  
ghost_densemat::rowmask = 1 0 1 1  
ghost_densemat::colmask = NULL  
~~~

Create a view of A which skips column 3:  
~~~
x x x .  
x x x .  
x x x .  
x x x .  
~~~
  
~~~
ghost_densematraits_t::flags  = GHOST_DENSEMAT_VIEW  
ghost_densemat::rowmask = NULL
ghost_densemat::colmask = NULL  
~~~

Create a view of A which skips column 3 and row 0:  
~~~
. . . .  
x x x .  
x x x .  
x x x .  
~~~
  
~~~
ghost_densematraits_t::flags  = GHOST_DENSEMAT_VIEW  
ghost_densemat::rowmask = NULL  
ghost_densemat::colmask = NULL  
~~~

Create a view of A which skips column 1 and row 0:  
~~~
. . . .  
x . x x  
x . x x  
x . x x  
~~~
  
~~~
ghost_densematraits_t::flags  = GHOST_DENSEMAT_VIEW|GHOST_DENSEMAT_SCATTERED_LD  
ghost_densemat::rowmask = NULL  
ghost_densemat::colmask = 1 0 1 1  
~~~

Create a view of A which skips column 1 and rows 0 and 2:  
~~~
. . . .  
x . x x  
. . . .  
x . x x  
~~~
  
~~~
ghost_densematraits_t::flags  = GHOST_DENSEMAT_VIEW|GHOST_DENSEMAT_SCATTERED_LD|GHOST_DENSEMAT_SCATTERED_TR  
ghost_densemat::rowmask = 0 1 0 1  
ghost_densemat::colmask = 1 0 1 1  
~~~
