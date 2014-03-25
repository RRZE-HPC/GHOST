Code instrumentation
====================

For performance engineering it is of interest to instrument certain (hot) regions of the source code.
Instrumentation can be switched on at compilation time and it affects by default all mathematical kernels in GHOST.
The region name will be the name of the kernel, e.g., axpy or spmv.
At the moment, instrumentation can be configured to either register a region's runtime or use the [Likwid](http://code.google.com/p/likwid/) Marker API.

Add more regions
----------------

A user may define more regions using the macros #GHOST_INSTR_START(tag) and #GHOST_INSTR_STOP(tag) where tag is a string with the desired region name.


Customize existing regions
--------------------------

If a mathematical kernel gets called in different surroundings, only a single region with the kernel name would be created for this kernel.
In order to avoid that, the functions ghost_instr_prefix_set() and ghost_instr_suffix_set() can be used to set a specific pre-/suffix string for each surrounding.

For example, in the following code only a single region called **dot** containing all dot products would be created:
~~~{.c}
int main() {
    ghost_densemat_t *x, *y, *z;
    ...
    ghost_dot(&dot_xy,x,y);
    ...
    ghost_dot(&dot_xz,x,z);
    ...
}
~~~

In order to have a separate region for each dot product, the following lines have to be inserted:
~~~{.c}
int main() {
    ghost_densemat_t *x, *y, *z;
    ...
    ghost_instr_suffix_set("_xy");
    ghost_dot(&dot_xy,x,y);
    ghost_instr_suffix_set(""); // clear suffix
    ...
    ghost_instr_suffix_set("_xz");
    ghost_dot(&dot_xz,x,z);
    ghost_instr_suffix_set(""); // clear suffix
    ...
}
~~~

Now, two regions (**dot_xy** and **dot_xz**) will be created.

Gather region information
-------------------------

### Likwid

In order to get information about instrumented regions using hardware performance monitoring with Likwid, the program binary call has to be wrapped with `likwid-perfctr`.
For detailed documentation see the [http://code.google.com/p/likwid/wiki/LikwidPerfCtr](Likwid homepage).

\note It is important to match the cores used by GHOST to the cores specified to `likwid-perfctr`.

### Timing

Region-specific timing information is stored in the struct ghost_timing_region_t. 
In order to get timing information about a certain region ghost_timing_region_create() is used.

The function ghost_timing_summarystring() can be used for generating a summary string about all measured regions.
