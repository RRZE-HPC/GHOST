!! we have to put the explicit interface to global GHOST
!! functions in a separate module from the types/structs
!! because the compiler won't allow derived types in ex-
!! plicit interfaces in the same module where they are  
!! declared.                                            
MODULE ghost_fun

IMPLICIT NONE

!-------------------------------------------------------!
! S1 interfaces to functions                            !
!-------------------------------------------------------!

  ! note: TYPE(c_ptr), value <- void*
  !  TYPE(c_ptr) <- void**
  ! (the default in fortran is call-by-reference,
  ! e.g. each arg is a pointer).

  ! If you want to e.g. use the dot-product, do something like this
  !  REGISTER_DT_D(vecdt)
  !  TYPE(ghost_vec_t) :: v,w
  !  REAL(vecdt_t) :: s
  ! ...
  !  call ghost_dotProduct(v,w,c_loc(s))
 
INTERFACE
! note - if we skip the "name=" clause, the compiler will 
!        probably make it all lower case and not find the 
!        function during link time.
SUBROUTINE ghost_normalizeVec(v) BIND(C,name='ghost_normalizeVec')
USE, INTRINSIC :: iso_c_binding
USE ghost_types
IMPLICIT NONE
TYPE(ghost_vec_t), intent(inout) :: v
END SUBROUTINE ghost_normalizeVec
 
SUBROUTINE ghost_dotProduct(v, w, s) BIND(C,name='ghost_dotProduct')
USE, INTRINSIC :: iso_c_binding
USE ghost_types
IMPLICIT NONE
TYPE(ghost_vec_t), intent(in) :: v,w
TYPE(c_ptr), value :: s
END SUBROUTINE ghost_dotProduct

! the filename here should be the location of an array of
! C_CHARs, not a fortran char(len=) string. For example: 
!                                                        
! CHARACTER(c_char), dimension(256) :: c_filename
! call f2c('vector.dat',c_filename)
! call ghost_vecToFile(v,c_loc(c_filename),ctx) 
!
! The length of the buffer c_filename is not that important
! as long as it is large enough because a \0 character is
! added to indicate the end of the char* array in C.
SUBROUTINE ghost_vecToFile(v, fname, ctx) BIND(C,name='ghost_vecToFile')
USE, INTRINSIC :: iso_c_binding
USE ghost_types
IMPLICIT NONE
TYPE(ghost_vec_t), intent(in) :: v
TYPE(c_ptr), value :: fname
TYPE(ghost_context_t),intent(in) :: ctx
END SUBROUTINE ghost_vecToFile

SUBROUTINE ghost_vecFromFile(v, fname, ctx) BIND(C,name='ghost_vecFromFile')
USE, INTRINSIC :: iso_c_binding
USE ghost_types
IMPLICIT NONE
TYPE(ghost_vec_t), intent(inout) :: v
TYPE(c_ptr), value :: fname
TYPE(ghost_context_t), intent(in) :: ctx
END SUBROUTINE ghost_vecFromFile

SUBROUTINE ghost_matFromFile(m, c, p) BIND(C,name='ghost_matFromFile')
USE, INTRINSIC :: iso_c_binding
USE ghost_types
IMPLICIT NONE
TYPE(ghost_mat_t), intent(inout) :: m
TYPE(ghost_context_t), intent(in) :: c
TYPE(c_ptr), value :: p
END SUBROUTINE ghost_matFromFile

SUBROUTINE ghost_vecFromScalar(v,c,s) BIND(C,name='ghost_vecFromScalar')
USE, INTRINSIC :: iso_c_binding
USE ghost_types
IMPLICIT NONE
TYPE(ghost_vec_t), intent(inout) :: v
TYPE(ghost_context_t),intent(in) :: c
TYPE(c_ptr), value :: s
END SUBROUTINE ghost_vecFromScalar

SUBROUTINE ghost_vecFromFunc(v,c,f) BIND(C,name='ghost_vecFromFunc')
USE, INTRINSIC :: iso_c_binding
USE ghost_types
IMPLICIT NONE
TYPE(ghost_vec_t), intent(inout) :: v
TYPE(ghost_context_t),intent(in) :: c
TYPE(c_funptr), value :: f
END SUBROUTINE ghost_vecFromFunc
! note: it is OK to call this with argc=0 argv=C_NULL_PTR
INTEGER FUNCTION ghost_init(argc, argv) BIND(C,name='ghost_init')
  USE, INTRINSIC :: iso_c_binding
  USE ghost_types
  IMPLICIT NONE
  INTEGER(c_int), value :: argc
  TYPE(C_PTR) :: argv
END FUNCTION ghost_init
 
SUBROUTINE ghost_finish() BIND(C,name='ghost_finish')
  IMPLICIT NONE
END SUBROUTINE ghost_finish

SUBROUTINE ghost_readMatFileHeader(path, header) BIND(C, name='ghost_readMatFileHeader')
  USE, INTRINSIC :: iso_c_binding
  USE ghost_types
  IMPLICIT NONE
  TYPE(c_ptr), value :: path
  TYPE(ghost_matfile_header_t), intent(in) :: header
END SUBROUTINE ghost_readMatFileHeader

TYPE(c_ptr) FUNCTION ghost_createMatrix(traits, nTraits) BIND(C, name='ghost_createMatrix')
  USE, INTRINSIC :: iso_c_binding
  USE ghost_types
  IMPLICIT NONE
  TYPE(ghost_mtraits_t), intent(in) :: traits
  INTEGER(c_int), value :: nTraits
END FUNCTION ghost_createMatrix

TYPE(c_ptr) FUNCTION ghost_createVector(traits) BIND(C, name='ghost_createVector')
  USE, INTRINSIC :: iso_c_binding
  USE ghost_types
  IMPLICIT NONE
  TYPE(ghost_vtraits_t), intent(in) :: traits
END FUNCTION ghost_createVector
 
TYPE(c_ptr) FUNCTION ghost_createContext(nrows, opts) &
        BIND(C, name='ghost_createContext')
  USE, INTRINSIC :: iso_c_binding
  USE ghost_types
  IMPLICIT NONE
  INTEGER(c_long), value :: nrows
  INTEGER(c_int), value :: opts ! note: this is an unsigned int in C, but that does not 
                                ! matter asthe two types occupy the same space
END FUNCTION ghost_createContext

SUBROUTINE ghost_freeContext(ctx) BIND(C,name='ghost_freeContext')
USE ghost_types
TYPE(ghost_context_t) :: ctx
END SUBROUTINE ghost_freeContext

SUBROUTINE ghost_freeVec(v) BIND(C,name='ghost_freeVec')
USE ghost_types
TYPE(ghost_vec_t) :: v
END SUBROUTINE ghost_freeVec
 
 ! functions we don't need for now
 /*
 ghost_comm_t * ghost_createCRS (char *matrixPath, void *deviceFormats);
 int ghost_spmvm(ghost_vec_t *res, ghost_context_t *context, ghost_vec_t *invec, int *spmvmOptions);
ghost_mat_t * ghost_initMatrix(ghost_mtraits_t *traits); 
ghost_vec_t * ghost_initVector(ghost_vtraits_t *traits); 
*/
END INTERFACE
 
END MODULE ghost_fun


