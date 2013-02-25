!! This module contains a partial 'Stage 2' interface
!! to ghost. It should be extended gradually with functions
!! which are very stable in ghost and which are often used.
MODULE fghost

USE, INTRINSIC :: iso_c_binding
USE ghost_types

IMPLICIT NONE

! define interface to the member functions we want to make
! available. These are just prototypes, we use their name 
! to indicate their output and input args.
ABSTRACT INTERFACE

INTEGER(ghost_mnnz_t) FUNCTION callback_MNNZ_CTX(ctx)
USE ghost_types
IMPLICIT NONE
TYPE(ghost_context_t), intent(in) :: ctx
END FUNCTION callback_MNNZ_CTX

INTEGER(ghost_midx_t) FUNCTION callback_MIDX_CTX(ctx)
USE ghost_types
IMPLICIT NONE
TYPE(ghost_context_t), intent(in) :: ctx
END FUNCTION callback_MIDX_CTX

END INTERFACE

#define INSTANTIATE_VEC_T(_PREFIX_,_TYPE_,_KIND_) \
TYPE fghost_ ## _PREFIX_ ## vec_t; \
  TYPE(ghost_vec_t), POINTER :: c_object; \
  _TYPE_(_KIND_), POINTER, DIMENSION(:) :: val; \
END TYPE fghost_ ## _PREFIX_ ## vec_t \

INSTANTIATE_VEC_T(s,real,c_float)
INSTANTIATE_VEC_T(d,real,c_double)
INSTANTIATE_VEC_T(c,complex,c_float)
INSTANTIATE_VEC_T(z,complex,c_double)

TYPE fghost_context_t
  
  TYPE(ghost_context_t), POINTER :: c_object
  
  PROCEDURE(callback_MNNZ_CTX), NOPASS, POINTER :: gnnz 
  PROCEDURE(callback_MIDX_CTX), NOPASS, POINTER :: gnrows 
  PROCEDURE(callback_MIDX_CTX), NOPASS, POINTER :: gncols

END TYPE fghost_context_t

interface c2f
module procedure c2f_context
module procedure c2f_svec,c2f_dvec,c2f_cvec,c2f_zvec
end interface c2f

CONTAINS

SUBROUTINE c2f_context(cctx,fctx)
USE, INTRINSIC :: iso_c_binding
USE ghost_types
IMPLICIT NONE
 TYPE(ghost_context_t), target, intent(in) :: cctx
 TYPE(fghost_context_t), intent(out) :: fctx

fctx%c_object => cctx
call C_F_PROCPOINTER(cctx%gnnz,fctx%gnnz)
call C_F_PROCPOINTER(cctx%gnrows,fctx%gnrows)
call C_F_PROCPOINTER(cctx%gncols,fctx%gncols)

END SUBROUTINE c2f_context

#define INSTANTIATE_C2F_VEC(_PREFIX_,_TYPE_,_KIND_) \
SUBROUTINE c2f_ ## _PREFIX_ ## vec(cvec,fvec); \
USE, INTRINSIC :: iso_c_binding; \
USE ghost_types; \
IMPLICIT NONE; \
 TYPE(ghost_vec_t), target, intent(in) :: cvec; \
 TYPE(fghost_ ## _PREFIX_ ## vec_t), intent(out) :: fvec; \
 fvec%c_object => cvec; \
 call C_F_POINTER(cvec%val,fvec%val); \
END SUBROUTINE c2f_ ## _PREFIX_ ## vec \

INSTANTIATE_C2F_VEC(s,real,c_float)
INSTANTIATE_C2F_VEC(d,real,c_double)
INSTANTIATE_C2F_VEC(c,complex,c_float)
INSTANTIATE_C2F_VEC(z,complex,c_double)

END MODULE fghost


