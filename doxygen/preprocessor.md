Preprocessing
=============

GHOST comes with a simple preprocessor in `bin/ghost_pp`.
It is a simple awk script and its only purpose currently is the duplication of code lines via `GHOST_UNROLL`.


GHOST_UNROLL
------------

This macro is used in the intrinsics implementation of compute kernels.
The code line has to start with a "#GHOST_UNROLL#". After that, the actual code follows in a single line.
Everything that should be substituted with a serial index has to be an "@" sign.
After the code line, another "#" followed with by the unroll size has to be specified.

Example:
~~~{.c}
#GHOST_UNROLL#int bla@ = @*4;#4
~~~
would result in the following code after preprocessing:
~~~{.c}
int bla0 = 0*4;
int bla1 = 1*4;
int bla2 = 2*4;
int bla3 = 3*4;
~~~
