MODULE ghost_types

USE, INTRINSIC :: iso_c_binding
IMPLICIT NONE

!!                                                                      
!! S1 interfaces to ghost structs.                                      
!!                                                                      
!! cf. README for an overview of the Fortran interface implementation.  
!!                                                              


!! these should be consistent with the definitions in ghost_types.h:
INTEGER, PARAMETER :: ghost_vidx_t = c_long
INTEGER, PARAMETER :: ghost_midx_t = c_long
INTEGER, PARAMETER :: ghost_mnnz_t = c_long

! these strings are used to define a matrix format in a
! more or less convenient way (TODO - add others)
CHARACTER(c_char), dimension(4), parameter :: str_CRS = (/'C', 'R','S',C_NULL_CHAR/)
CHARACTER(c_char), dimension(5), parameter :: str_SELL = (/'B','J','D','S',C_NULL_CHAR/)

!---------------------------------------------------------!
! Stage 1 interfaces (have the same name as in C)         !
!---------------------------------------------------------!
TYPE, BIND(C) :: ghost_mat_t
  TYPE(c_ptr) :: traits
  TYPE(c_ptr) :: so
  TYPE(c_ptr) :: rowPerm
  TYPE(c_ptr) :: invRowPerm
  TYPE(c_ptr) :: localPart
  TYPE(c_ptr) :: remotePart
  TYPE(c_ptr) :: context
  TYPE(c_ptr) :: name
  TYPE(c_ptr) :: data
  ! void functions
  TYPE(c_funptr) :: fromFile !(ghost_mat_t *, ghost_context_t *, char *path);
END TYPE ghost_mat_t

TYPE, BIND(C) :: ghost_vec_t
  TYPE(c_ptr) :: traits
  TYPE(c_ptr) :: val
  ! void functions
  TYPE(c_funptr) :: fromFunc !(ghost_vec_t *, void (*fp)(int,int,void *));
  TYPE(c_funptr) :: fromVec !(ghost_vec_t *, ghost_vec_t *, int, int, int *);
  TYPE(c_funptr) :: fromFile !(ghost_vec_t *, char *path, off_t);
  TYPE(c_funptr) :: fromRand !(ghost_vec_t *);
  TYPE(c_funptr) :: fromScalar !(ghost_vec_t *, void *);
  TYPE(c_funptr) :: zero !    !(ghost_vec_t *);
  TYPE(c_funptr) :: distribute !(ghost_vec_t *, ghost_vec_t **, ghost_comm_t *comm);
  TYPE(c_funptr) :: collect !(ghost_vec_t *, ghost_vec_t *, ghost_context_t *);
  TYPE(c_funptr) :: swap !(ghost_vec_t *, ghost_vec_t *);
  TYPE(c_funptr) :: normalize !(ghost_vec_t *);
  TYPE(c_funptr) :: destroy !(ghost_vec_t *);
  TYPE(c_funptr) :: permute !(ghost_vec_t *, ghost_vidx_t *); 
  ! int function 
  TYPE(c_funptr) :: equals !(ghost_vec_t *, ghost_vec_t *); 
  TYPE(c_funptr) :: dotProduct !(ghost_vec_t *, ghost_vec_t *, void *); 
  TYPE(c_funptr) :: scale !(ghost_vec_t *, void *); 
  TYPE(c_funptr) :: axpy !(ghost_vec_t *, ghost_vec_t *, void*)
  TYPE(c_funptr) :: print !(ghost_vec_t *); 
  TYPE(c_funptr) :: toFile !(ghost_vec_t *, char *, off_t, int); 
  TYPE(c_funptr) :: entry !(ghost_vec_t *, int, void *);

  ! functions returning ghost_vec_t* 
  TYPE(c_funptr) :: clone !(ghost_vec_t *); 
  TYPE(c_funptr) :: extract !(ghost_vec_t *, int, int); 
  TYPE(c_funptr) :: view !(ghost_vec_t *, int, int);

TYPE(c_ptr) :: so

INTEGER(c_int) :: isView

#ifdef GHOST_HAVE_CUDA
!#error "not implemented in Fortran interface"
!void * CU_val;
#endif
END TYPE ghost_vec_t

! STAGE 1 interface. This is fairly usable already as it mainly
! contains primitive data types.
TYPE, BIND(C) :: ghost_matfile_header_t
  INTEGER(c_int) :: endianess
  INTEGER(c_int) :: version
  INTEGER(c_int) :: base
  INTEGER(c_int) :: symmetry
  INTEGER(c_int) :: datatype
  INTEGER(c_long) :: nrows
  INTEGER(c_long) :: ncols
  INTEGER(c_long) :: nnz
END TYPE ghost_matfile_header_t


TYPE, BIND(C) :: ghost_vtraits_t
  INTEGER(c_int)  :: flags 
  TYPE(c_ptr)     :: aux
  INTEGER(c_int)     :: datatype
  INTEGER(c_int)     :: nrows
  INTEGER(c_int)     :: nrowspadded
  INTEGER(c_int)     :: nvecs
END TYPE ghost_vtraits_t

TYPE, BIND(C) :: ghost_context_t
  TYPE(c_ptr) :: solvers
  TYPE(c_ptr) :: communicator
  TYPE(c_ptr) :: fullMatrix
  TYPE(c_ptr) :: localMatrix
  TYPE(c_ptr) :: remoteMatrix
  
  TYPE(c_funptr) gnnz !(ghost_context_t *);
  TYPE(c_funptr) gnrows !(ghost_context_t *);
  TYPE(c_funptr) gncols !(ghost_context_t *);
  TYPE(c_funptr) lnnz !(ghost_context_t *);
  TYPE(c_funptr) lnrows !(ghost_context_t *)
  TYPE(c_funptr) lncols !(ghost_context_t *);

  TYPE(c_ptr) matrixName
  INTEGER(c_int) :: flags
END TYPE ghost_context_t

TYPE, BIND(C) :: ghost_mtraits_t 
  INTEGER(c_int) :: format
  INTEGER(c_int) :: flags
  INTEGER(c_int) :: nAux
  INTEGER(c_int) :: datatype
  TYPE(c_ptr) :: aux
  TYPE(c_ptr) :: shift
END TYPE ghost_mtraits_t



! structs we don't need in the first place
! in the Fortran interface
#if 0

struct ghost_comm_t {
    ghost_midx_t halo_elements; // number of nonlocal RHS vector elements
    ghost_mnnz_t* lnEnts;
    ghost_mnnz_t* lfEnt;
    ghost_midx_t* lnrows;
    ghost_midx_t* lfRow;
    ghost_mnnz_t* wishes;
    int* wishlist_mem; // TODO delete
    int** wishlist; // TODO delete
    ghost_mnnz_t* dues;
    int* duelist_mem; // TODO delete
    int** duelist;
    int* due_displ;
    int* wish_displ; // TODO delete
    int* hput_pos;
};

struct ghost_mat_t {
    ghost_mtraits_t *traits; // TODO rename

    // access functions
    void (*destroy) (ghost_mat_t *);
    void (*printInfo) (ghost_mat_t *);
    ghost_mnnz_t (*nnz) (ghost_mat_t *);
    ghost_midx_t (*nrows) (ghost_mat_t *);
    ghost_midx_t (*ncols) (ghost_mat_t *);
    ghost_midx_t (*rowLen) (ghost_mat_t *, ghost_midx_t i); // ghost_mdat_t (*entry) (ghost_mat_t *, ghost_midx_t i, ghost_midx_t j);
    char * (*formatName) (ghost_mat_t *);
    void (*fromBin)(ghost_mat_t *, char *matrixPath, ghost_context_t *ctx, int options);
    void (*fromMM)(ghost_mat_t *, char *matrixPath);
    void (*CUupload)(ghost_mat_t *);
    size_t (*byteSize)(ghost_mat_t *);
    void (*fromCRS)(ghost_mat_t *, void *);
    void (*split)(ghost_mat_t *, int options, ghost_context_t *, ghost_mtraits_t *traits);
    ghost_dummyfun_t *extraFun;
    // TODO MPI-IO
    ghost_kernel_t kernel;
    void *so;

    ghost_midx_t *rowPerm; // may be NULL
    ghost_midx_t *invRowPerm; // may be NULL

    int symmetry;

    void *data;
};

#endif !structs not needed for now

interface f2c
module procedure f2c_string
end interface f2c

CONTAINS

!! copies a fortran string (char(len=*)) into
!! a C compatible array of chars. The length 
!! of the cbuf array should be at least len(fstring)+1,
!! but it is OK to pass it to C even if it is longer. If
!! it is too short, the string is truncated without further
!! warning.
SUBROUTINE f2c_string(fstring,cstring)
USE, INTRINSIC :: iso_c_binding
IMPLICIT NONE
 
character(len=*), intent(in) :: fstring
character(c_char), dimension(:) :: cstring
integer :: i,n

n=min(len(trim(fstring)),size(cstring)-1)
do i=1,n
  cstring(i) = fstring(i:i)
end do
cstring(n+1)=C_NULL_CHAR

END SUBROUTINE f2c_string

END MODULE ghost_types


