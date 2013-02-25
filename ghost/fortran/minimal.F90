#include "fghost.h"

program minimal

use ghost_types
use ghost_fun
use ghost_util
! we need some functions from the structs
! that can't be accessed via global wrappers.
! We use the fortran types in fghost to interface
! these (S2 interface).
use fghost

use, intrinsic :: iso_c_binding

implicit none

GHOST_REGISTER_DT_D(vecdt)

integer :: nIter
real(c_double) :: time
real(c_double), parameter :: zero = 0.
real(c_double), parameter :: FlOPS_PER_ENTRY = 2.0

integer(c_int) :: ghostOptions
integer(c_int) :: spmvmOptions


TYPE(ghost_mtraits_t) :: mtraits
TYPE(ghost_vtraits_t) :: lvtraits, rvtraits
TYPE(ghost_matfile_header_t) :: fileheader

TYPE(c_ptr), target :: ctxPtr,lhsPtr,rhsPtr,matPtr      ! what C gives us when we call createXYZ

! in case we want to actually work with the stuff, we need to cast the xyzPtr using
! C_F_POINTER:
TYPE(ghost_context_t), pointer :: ctx
TYPE(ghost_vec_t), pointer :: lhs
TYPE(ghost_vec_t), pointer :: rhs
TYPE(ghost_mat_t), pointer :: mat


TYPE(fghost_context_t) :: fctx

character(len=256) :: matname
character(c_char), dimension(256) :: cmatname

integer :: ierr, me

#ifdef MPI
call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD,me,ierr)
#else
me=0
#endif

spmvmOptions = GHOST_SPMVM_AXPY 

ierr=ghost_init(0,C_NULL_PTR)
CALL ghost_printSysInfo()
CALL ghost_printGhostInfo()
nIter=1
! TODO - we should have a function that creates the
!        traits in fortran, something like
! mtraits=ghost_createMtraits('CRS',GHOST_SPM_DEFAULT,GHOST_BINCRS_DT_FLOAT)
! Alternatively, we could initialize the most common traits in a module so  
! that they can just be used.
mtraits%format = C_LOC(str_CRS) 
mtraits%flags = GHOST_SPM_DEFAULT
mtraits%datatype = vecdt

lvtraits%flags = GHOST_VEC_LHS
lvtraits%datatype = vecdt
lvtraits%nvecs = 1

rvtraits%flags = GHOST_VEC_RHS
rvtraits%datatype = vecdt
rvtraits%nvecs = 1

! NOTE: it is not clear how this behaves in an MPI 
! context (at least not to me). The function is part
! of the F2003 standard and should otherwise be fine.
call get_command_argument(1,matname,status=ierr)
if (ierr>0) then
  if (me==0) then
    write(*,*) 'USAGE: <progname> <matname>'
  end if
  stop
else if (ierr==-1) then
  if (me==0) then
    write(*,*) 'the matrix name you passed in is too long'
  end if
  stop
end if
call f2c(TRIM(matname),cmatname)

CALL ghost_readMatFileHeader(C_LOC(cmatname),fileheader)
ctxPtr = ghost_createContext(fileheader%nrows,GHOST_CONTEXT_DEFAULT)
! set the pointer in Fortran to point to the new object:
call C_F_POINTER(ctxPtr,ctx)
CALL ghost_printContextInfo(ctx)
! get a 'Stage 2' (S2) interface as well. We need this to
! access member function gnnz further down.
call c2f(ctx, fctx)

matPtr = ghost_createMatrix(mtraits,1)
call C_F_POINTER(matPtr,mat)

rhsPtr = ghost_createVector(rvtraits)
call C_F_POINTER(rhsPtr,rhs)

lhsPtr = ghost_createVector(lvtraits)
call C_F_POINTER(lhsPtr,lhs)

call ghost_matFromFile(mat,ctx,C_LOC(cmatname))
call ghost_vecFromScalar(lhs,ctx,C_LOC(zero))
call ghost_vecFromFunc(rhs,ctx,C_FUNLOC(rhsVal))

CALL ghost_printMatrixInfo(mat)
time = ghost_bench_spmvm(lhs,ctx,rhs,C_LOC(spmvmOptions),nIter)

if (time > 0.) then
  if (me==0) then
    ! TODO - make output look like in C, this is a bit
    !        annoying because we don't have access to 
    !        ctx->gnnz and strings are hard to pass to
    !        C
    write(*,*) 'Performance'
    !ghost_modeName(spmvmOptions),
    write(*,*) 'GF/s',FLOPS_PER_ENTRY*1.e-9*ghost_getMatNnz(mat)/time
    write(*,*) 'run time (s): ',time
  end if
end if

CALL ghost_printFooter()

CALL ghost_freeVec(lhs)
CALL ghost_freeVec(rhs)
CALL ghost_freeContext(ctx)

CALL ghost_finish()

#ifdef MPI
  CALL MPI_Finalize(ierr)
#endif

CONTAINS

subroutine rhsVal (i, v, val)
use, intrinsic :: iso_c_binding
implicit none
integer(c_int), value :: i
integer(c_int), value :: v
TYPE(c_ptr), value :: val

!local
REAL(vecdt_t), pointer :: f_val

CALL C_F_POINTER(val,f_val)

f_val = 1 !+I*1; !i + (vecdt_t)1.0 + I*i;

end subroutine rhsVal


end program minimal

