#ifndef FGHOST_H
#define FGHOST_H

/* file fghost.h. This file declares macros only, 
   it should be included at the top of a Fortran  
   source file using ghost to make common flags   
   etc. available.
*/

/* this file is compatible with C and Fortran. */
/* It defines macros for options etc.          */
#include "ghost_constants.h"

/* these four macros can be used to define the kind parameter for the 
   scalar data type. As Fortran does not have typedefs, it is up to the
   user to use the correct data type, e.g. complex(kind=name) or        
   real(kind=name)
   
   The place to put the macro is in the data section of a program unit,
   for instance
   
   PROGRAM GhostUser
   USE, intrinsic :: iso_c_binding
   IMPLICIT NONE
   GHOST_REGISTER_DT_Z(vecdt)
   ...
   
   will give you
   
   integer, parameter :: vecdt_t=8
   integer, parameter :: vecdt = <something understood by ghost)
   
   The corresponding vector entries are then
   
   COMPLEX(TYPE=vecdt_t)
   */
#define GHOST_REGISTER_DT_D(name) \
        integer, parameter :: name ## _t = KIND(c_double); \
    integer, parameter :: name = IOR(GHOST_BINCRS_DT_DOUBLE,GHOST_BINCRS_DT_REAL) \

#define GHOST_REGISTER_DT_S(name) \
        integer, parameter :: name ## _t = KIND(c_float); \
    integer, parameter :: name = IOR(GHOST_BINCRS_DT_FLOAT,GHOST_BINCRS_DT_REAL) \

#define GHOST_REGISTER_DT_C(name) \
        integer, parameter :: name ## _t = KIND(c_float); \
    integer, parameter :: name = IOR(GHOST_BINCRS_DT_FLOAT,GHOST_BINCRS_DT_COMPLEX) \

#define GHOST_REGISTER_DT_Z(name) \
        integer, parameter :: name ## _t = KIND(c_double); \
    integer, parameter :: name = IOR(GHOST_BINCRS_DT_DOUBLE,GHOST_BINCRS_DT_COMPLEX) \


#endif
