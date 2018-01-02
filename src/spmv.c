#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/math.h"
#include "ghost/instr.h"
#include "ghost/util.h"
#include "ghost/spmv_solvers.h"
#include "ghost/compatibility_check.h"

ghost_error ghost_spmv(ghost_densemat *res, ghost_sparsemat *mat, ghost_densemat *invec, ghost_spmv_opts traits)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_lidx ncolsbackup = res->traits.ncols, remcols = res->traits.ncols, donecols = 0;
    GHOST_DEBUG_LOG(1,"Performing SpMV");
    ghost_spmv_kernel solver = NULL;
    if (traits.flags & GHOST_SPMV_MODE_OVERLAP) {
        solver = &ghost_spmv_goodfaith;
    } else if (traits.flags & GHOST_SPMV_MODE_TASK) {
        solver = &ghost_spmv_taskmode; 
    } else if (traits.flags & GHOST_SPMV_MODE_NOCOMM) {
        solver = &ghost_spmv_nocomm; 
    } else if (traits.flags & GHOST_SPMV_MODE_PIPELINED) {
        solver = &ghost_spmv_pipelined; 
    } else {
#ifdef GHOST_HAVE_MPI
        solver = &ghost_spmv_vectormode;
#else
        solver = &ghost_spmv_nocomm; 
#endif
    }

    if (!solver) {
        GHOST_ERROR_LOG("The SpMV solver as specified in options cannot be found.");
        return GHOST_ERR_INVALID_ARG;
    }
    
    //////////////// check compatibility /////////////
    ghost_compatible_mat_vec check = GHOST_COMPATIBLE_MAT_VEC_INITIALIZER;
    check.mat = mat;
    check.right1 = invec;
    check.left1 = res;
    
    GHOST_CALL_RETURN(ghost_check_mat_vec_compatibility(&check,mat->context));
    ///////////////////////////////////////////////////

    if (traits.blocksz < res->traits.ncols) {
       if (res->traits.flags & GHOST_DENSEMAT_SCATTERED ||
               invec->traits.flags & GHOST_DENSEMAT_SCATTERED ||
               (traits.z && traits.z->traits.ncols & GHOST_DENSEMAT_SCATTERED) ||
               (res->traits.ncols > 1 && res->traits.storage == GHOST_DENSEMAT_COLMAJOR) ||
               (invec->traits.ncols > 1 && invec->traits.storage == GHOST_DENSEMAT_COLMAJOR) ||
               (traits.z && traits.z->traits.ncols > 1 && traits.z->traits.storage == GHOST_DENSEMAT_COLMAJOR)) {
           GHOST_WARNING_LOG("Cannot apply desired SpMV blocking! Fallback to no blocking!");
           traits.blocksz = res->traits.ncols;
       }
    }
            


    // TODO only if densemats are compact!
    while (remcols > traits.blocksz) {
        res->traits.ncols = traits.blocksz;
        invec->traits.ncols = traits.blocksz;
        if (traits.z) {
            traits.z->traits.ncols = traits.blocksz;
        }
        
        GHOST_CALL_RETURN(solver(res,mat,invec,traits));
        
        invec->val += traits.blocksz*invec->elSize;
        res->val += traits.blocksz*res->elSize;
        if (traits.z) {
            traits.z->val += traits.blocksz*traits.z->elSize;
        }

        donecols += traits.blocksz;
        remcols -= traits.blocksz;
    }
    res->traits.ncols = remcols;
    invec->traits.ncols = remcols;
    
    if (traits.z) {
        traits.z->traits.ncols = remcols;
    }
    
    GHOST_CALL_RETURN(solver(res,mat,invec,traits));

    res->val -= donecols*res->elSize;
    invec->val -= donecols*invec->elSize;
    
    res->traits.ncols = ncolsbackup;
    invec->traits.ncols = ncolsbackup;
    if (traits.z) {
        traits.z->traits.ncols = ncolsbackup;
        traits.z->val -= donecols*traits.z->elSize;
    }

    if (!(traits.flags & GHOST_SPMV_NOT_REDUCE) && (traits.flags & GHOST_SPMV_DOT)) {
#ifdef GHOST_HAVE_MPI
        ghost_mpi_op op;
        ghost_mpi_datatype dt;
        ghost_mpi_op_sum(&op,res->traits.datatype);
        ghost_mpi_datatype_get(&dt,res->traits.datatype);

        MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, traits.dot, 3*invec->traits.ncols, dt, op, mat->context->mpicomm));
        GHOST_INSTR_STOP("dot_reduce");
#endif
    }

    if (traits.flags & GHOST_SPMV_BARRIER) {
        ghost_barrier();
    }


#ifdef GHOST_INSTR_TIMING
    ghost_gidx nnz;
    ghost_gidx nrow;
    
    ghost_sparsemat_nnz(&nnz,mat);
    ghost_sparsemat_nrows(&nrow,mat);

    ghost_spmv_perf_args spmv_perfargs;
    spmv_perfargs.vecncols = invec->traits.ncols;
    spmv_perfargs.globalnnz = nnz;
    spmv_perfargs.globalrows = nrow;
    spmv_perfargs.dt = invec->traits.datatype;
    spmv_perfargs.flags = traits.flags;
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_spmv_perf,(void *)&spmv_perfargs,sizeof(spmv_perfargs),GHOST_SPMV_PERF_UNIT);
#endif 

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return GHOST_SUCCESS;


}


ghost_error ghost_spmv_nocomm(ghost_densemat *res, ghost_sparsemat *mat, ghost_densemat *invec, ghost_spmv_opts traits)
{
    ghost_type ghost_type;
    GHOST_CALL_RETURN(ghost_type_get(&ghost_type));

    #ifdef GHOST_HAVE_CUDA
    if ((ghost_type == GHOST_TYPE_CUDA) && mat->traits.flags & GHOST_SPARSEMAT_DEVICE) {
        if (res->traits.ncols == 1 && res->traits.storage == GHOST_DENSEMAT_COLMAJOR && mat->traits.C == 1 && !(traits.flags & (GHOST_SPMV_SHIFT|GHOST_SPMV_VSHIFT)) && !(traits.flags & GHOST_SPMV_CHAIN_AXPBY)) {
            return ghost_cu_sell1_spmv_selector(res,mat,invec,traits); // use a function which calls CUSPARSE as they can do better for CRS matrices
        } else {
            return ghost_cu_sell_spmv_selector(res,mat,invec,traits);
        }
    }
    #endif

    return ghost_sell_spmv_selector(res,mat,invec,traits);

}
