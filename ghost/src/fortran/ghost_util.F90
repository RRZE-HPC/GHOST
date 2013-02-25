MODULE ghost_util

INTERFACE

SUBROUTINE ghost_printMatrixInfo(matrix) BIND(C, name='ghost_printMatrixInfo')
  USE ghost_types
  IMPLICIT NONE
  TYPE(ghost_mat_t), intent(in) :: matrix
END SUBROUTINE ghost_printMatrixInfo

SUBROUTINE ghost_printSysInfo() BIND(C, name='ghost_printSysInfo')
  USE, INTRINSIC :: iso_c_binding
  IMPLICIT NONE
END SUBROUTINE ghost_printSysInfo
 
SUBROUTINE ghost_printGhostInfo() BIND(C, name='ghost_printGhostInfo')
  USE, INTRINSIC :: iso_c_binding 
  IMPLICIT NONE
END SUBROUTINE ghost_printGhostInfo

SUBROUTINE ghost_printContextInfo(ctx) BIND(C,name='ghost_printContextInfo')
  USE ghost_types
  IMPLICIT NONE
  TYPE(ghost_context_t), intent(in) :: ctx
END SUBROUTINE ghost_printContextInfo

INTEGER(c_int) FUNCTION ghost_getMatNnz(mat) BIND(C, name='ghost_getMatNnz')
  USE ghost_types
  USE, INTRINSIC :: iso_c_binding
  IMPLICIT NONE
  TYPE(ghost_mat_t), intent(in) :: mat
END FUNCTION ghost_getMatNnz

!SUBROUTINE ghost_printHeader(string)
REAL(c_double) FUNCTION ghost_bench_spmvm(lhs,ctx,rhs,spmvmOptions,nIter) &
        BIND(C,name='ghost_bench_spmvm')
  USE ghost_types
  USE, INTRINSIC :: iso_c_binding
  IMPLICIT NONE
  TYPE(ghost_vec_t), intent(in) :: rhs
  TYPE(ghost_vec_t), intent(out) :: lhs
  TYPE(ghost_context_t), intent(in) :: ctx
  TYPE(c_ptr), value :: spmvmOptions
  INTEGER(c_int), value :: nIter
END FUNCTION ghost_bench_spmvm

!TODO - we skipped the ghost_printLine and ghost_printHeader
!       functions because of the variable input args, maybe 
!       an equivalent Fortran implementation could be done to
!       create the same 'look-and-feel'.

SUBROUTINE ghost_printFooter() BIND(C, name='ghost_printFooter')
IMPLICIT NONE
END SUBROUTINE ghost_printFooter

END INTERFACE

END MODULE ghost_util
