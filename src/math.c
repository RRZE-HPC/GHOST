#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/densemat.h"
#include "ghost/locality.h"
#include "ghost/instr.h"
#include "ghost/log.h"
#include "ghost/spmv_solvers.h"
#include <math.h>
#include <complex.h>

static ghost_mpi_op_t GHOST_MPI_OP_SUM_C = MPI_OP_NULL;
static ghost_mpi_op_t GHOST_MPI_OP_SUM_Z = MPI_OP_NULL;

static void ghost_spmv_selectMode(ghost_context_t * context, ghost_spmv_flags_t *flags);

ghost_error_t ghost_dot(void *res, ghost_densemat_t *vec, ghost_densemat_t *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH)
    vec->dot(vec,res,vec2);
#ifdef GHOST_HAVE_MPI
    if (vec->context) {
        GHOST_INSTR_START("reduce")
        ghost_mpi_op_t sumOp;
        ghost_mpi_datatype_t mpiDt;
        ghost_mpi_op_sum(&sumOp,vec->traits.datatype);
        ghost_mpi_datatype(&mpiDt,vec->traits.datatype);
        int v;
        if (!(vec->traits.flags & GHOST_DENSEMAT_GLOBAL)) {
            for (v=0; v<MIN(vec->traits.ncols,vec2->traits.ncols); v++) {
                MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, (char *)res+vec->elSize*v, 1, mpiDt, sumOp, vec->context->mpicomm));
            }
        }
        GHOST_INSTR_STOP("reduce")
    }
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH)
    return GHOST_SUCCESS;

}

ghost_error_t ghost_normalize(ghost_densemat_t *vec)
{
    ghost_lidx_t ncols = vec->traits.ncols;
    ghost_lidx_t c;

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    if (vec->traits.datatype & GHOST_DT_FLOAT) {
        if (vec->traits.datatype & GHOST_DT_COMPLEX) {
            complex float res[ncols];
            GHOST_CALL_RETURN(ghost_dot(res,vec,vec));
            for (c=0; c<ncols; c++) {
                res[c] = 1.f/csqrtf(res[c]);
            }
            GHOST_CALL_RETURN(vec->vscale(vec,res));
        } else {
            float res[ncols];
            GHOST_CALL_RETURN(ghost_dot(res,vec,vec));
            for (c=0; c<ncols; c++) {
                res[c] = 1.f/sqrtf(res[c]);
            }
            GHOST_CALL_RETURN(vec->vscale(vec,res));
        }
    } else {
        if (vec->traits.datatype & GHOST_DT_COMPLEX) {
            complex double res[ncols];
            GHOST_CALL_RETURN(ghost_dot(res,vec,vec));
            for (c=0; c<ncols; c++) {
                res[c] = 1./csqrt(res[c]);
            }
            GHOST_CALL_RETURN(vec->vscale(vec,res));
        } else {
            double res[ncols];
            GHOST_CALL_RETURN(ghost_dot(res,vec,vec));
            for (c=0; c<ncols; c++) {
                res[c] = 1./sqrt(res[c]);
            }
            GHOST_CALL_RETURN(vec->vscale(vec,res));
        }
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_vspmv(ghost_densemat_t *res, ghost_sparsemat_t *mat, ghost_densemat_t *invec, ghost_spmv_flags_t *flags, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    va_list argp_backup;
    va_copy(argp_backup,argp);
    DEBUG_LOG(1,"Performing SpMV");
    ghost_spmvsolver_t solver = NULL;
    ghost_spmv_selectMode(mat->context,flags);
    if (*flags & GHOST_SPMV_MODE_VECTOR) {
        solver = &ghost_spmv_vectormode;
    } else if (*flags & GHOST_SPMV_MODE_OVERLAP) {
        solver = &ghost_spmv_goodfaith;
    } else if (*flags & GHOST_SPMV_MODE_TASK) {
        solver = &ghost_spmv_taskmode; 
    } else if (*flags & GHOST_SPMV_MODE_NOMPI) {
        solver = &ghost_spmv_nompi; 
    }

    if (!solver) {
        ERROR_LOG("The SpMV solver as specified in options cannot be found.");
        return GHOST_ERR_INVALID_ARG;
    }

    solver(res,mat,invec,*flags,argp);

#ifdef GHOST_HAVE_MPI

    if (!(*flags & GHOST_SPMV_NOT_REDUCE) && (*flags & GHOST_SPMV_DOT_ANY)) {
        GHOST_INSTR_START("dot_reduce");
        void *dot = NULL;
        if (*flags & GHOST_SPMV_SCALE) {
            dot = va_arg(argp_backup,void *);
        }
        if (*flags & GHOST_SPMV_AXPBY) {
            dot = va_arg(argp_backup,void *);
        }
        if ((*flags & GHOST_SPMV_SHIFT) || (*flags & GHOST_SPMV_VSHIFT)) {
            dot = va_arg(argp_backup,void *);
        }
        if (*flags & GHOST_SPMV_DOT_ANY) {
            dot = va_arg(argp_backup,void *);
        }
        ghost_mpi_op_t op;
        ghost_mpi_datatype_t dt;
        ghost_mpi_op_sum(&op,res->traits.datatype);
        ghost_mpi_datatype(&dt,res->traits.datatype);

        MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, dot, 3*invec->traits.ncols, dt, op, mat->context->mpicomm));
        GHOST_INSTR_STOP("dot_reduce");
    }
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return GHOST_SUCCESS;


}
ghost_error_t ghost_spmv(ghost_densemat_t *res, ghost_sparsemat_t *mat, ghost_densemat_t *invec, ghost_spmv_flags_t *flags, ...) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);

    ghost_error_t ret = GHOST_SUCCESS;

    if (*flags & GHOST_SPMV_DOT) {
        *flags |= (ghost_spmv_flags_t)(GHOST_SPMV_DOT_YY|GHOST_SPMV_DOT_XY|GHOST_SPMV_DOT_XX);
    }
    va_list argp;
    va_start(argp, flags);
    ret = ghost_vspmv(res,mat,invec,flags,argp);
    va_end(argp);

#ifdef GHOST_HAVE_INSTR_TIMING
    ghost_gidx_t nnz;
    ghost_gidx_t nrow;
    
    ghost_sparsemat_nnz(&nnz,mat);
    ghost_sparsemat_nrows(&nrow,mat);

    ghost_spmv_perf_args_t spmv_perfargs;
    spmv_perfargs.vecncols = invec->traits.ncols;
    spmv_perfargs.globalnnz = nnz;
    spmv_perfargs.globalrows = nrow;
    spmv_perfargs.dt = invec->traits.datatype;
    spmv_perfargs.flags = *flags;
    ghost_timing_set_perfFunc(GHOST_SPMV_PERF_TAG,ghost_spmv_perf,(void *)&spmv_perfargs,sizeof(spmv_perfargs),GHOST_SPMV_PERF_UNIT);
#endif 
    
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);

    return ret;
}

#ifdef GHOST_HAVE_MPI

static void ghost_mpi_add_c(ghost_mpi_c *invec, ghost_mpi_c *inoutvec, int *len)
{
    int i;
    ghost_mpi_c c;

    for (i=0; i<*len; i++, invec++, inoutvec++){
        c.x = invec->x + inoutvec->x;
        c.y = invec->y + inoutvec->y;
        *inoutvec = c;
    }
}

static void ghost_mpi_add_z(ghost_mpi_z *invec, ghost_mpi_z *inoutvec, int *len)
{
    int i;
    ghost_mpi_z c;

    for (i=0; i<*len; i++, invec++, inoutvec++){
        c.x = invec->x + inoutvec->x;
        c.y = invec->y + inoutvec->y;
        *inoutvec = c;
    }
}

#endif

static void ghost_spmv_selectMode(ghost_context_t * context, ghost_spmv_flags_t *flags)
{
    if (!(*flags & GHOST_SPMV_MODES_ALL)) { // no mode specified
#ifdef GHOST_HAVE_MPI
        if (context->flags & GHOST_CONTEXT_REDUNDANT) {
            *flags |= (ghost_spmv_flags_t)GHOST_SPMV_MODE_NOMPI;
        } else {
            *flags |= (ghost_spmv_flags_t)GHOST_SPMV_MODE_OVERLAP;
        }
#else
        UNUSED(context);
        *flags |= (ghost_spmv_flags_t)GHOST_SPMV_MODE_NOMPI;
#endif
        DEBUG_LOG(1,"No spMVM mode has been specified, selecting a sensible default, namely %s",ghost_spmv_mode_string(*flags));
    } else {
#ifndef GHOST_HAVE_MPI
        if ((*flags & GHOST_SPMV_MODES_MPI)) {
            WARNING_LOG("Forcing non-MPI SpMV!");
            *flags &= ~(ghost_spmv_flags_t)GHOST_SPMV_MODES_MPI;
            *flags |= (ghost_spmv_flags_t)GHOST_SPMV_MODE_NOMPI;
        }
#endif
    }
}

ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, int datatype)
{
    if (!op) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
#ifdef GHOST_HAVE_MPI
    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_COMPLEX) {
            *op = GHOST_MPI_OP_SUM_C;
        } else {
            *op = MPI_SUM;
        }
    } else {
        if (datatype & GHOST_DT_COMPLEX) {
            *op = GHOST_MPI_OP_SUM_Z;
        } else {
            *op = MPI_SUM;
        }
    }
#else
    UNUSED(datatype);
    *op = MPI_OP_NULL;
#endif

    return GHOST_SUCCESS;

}

ghost_error_t ghost_mpi_operations_create()
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_c,1,&GHOST_MPI_OP_SUM_C));
    MPI_CALL_RETURN(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_z,1,&GHOST_MPI_OP_SUM_Z));
#else
    UNUSED(GHOST_MPI_OP_SUM_C);
    UNUSED(GHOST_MPI_OP_SUM_Z);
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_mpi_operations_destroy()
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_C));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_Z));
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_spmv_nflops(int *nFlops, ghost_datatype_t m_t, ghost_datatype_t v_t)
{
    if (!ghost_datatype_valid(m_t)) {
        ERROR_LOG("Invalid matrix data type: %d",(int)m_t);
        return GHOST_ERR_INVALID_ARG;
    }
    if (!ghost_datatype_valid(v_t)) {
        ERROR_LOG("Invalid vector data type: %d",(int)v_t);
        return GHOST_ERR_INVALID_ARG;
    }
    if (!nFlops) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *nFlops = 2;

    if (m_t & GHOST_DT_COMPLEX) {
        if (v_t & GHOST_DT_COMPLEX) {
            *nFlops = 8;
        }
    } else {
        if (v_t & GHOST_DT_COMPLEX) {
            *nFlops = 4;
        }
    }

    return GHOST_SUCCESS;
}

char * ghost_spmv_mode_string(ghost_spmv_flags_t flags) 
{
    if (flags & GHOST_SPMV_MODE_NOMPI) {
        return "Non-MPI";
    }
    if (flags & GHOST_SPMV_MODE_VECTOR) {
        return "Vector mode";
    }
    if (flags & GHOST_SPMV_MODE_OVERLAP) {
        return "Naïve overlap mode";
    }
    if (flags & GHOST_SPMV_MODE_TASK) {
        return "Task mode";
    }
    return "Invalid";

}

int ghost_spmv_perf(double *perf, double time, void *varg)
{
    ghost_spmv_perf_args_t arg = *(ghost_spmv_perf_args_t *)varg;

    ghost_lidx_t ncol = arg.vecncols;
    double flops;
    ghost_gidx_t nnz = arg.globalnnz;
    ghost_gidx_t nrow = arg.globalrows;
    int spmvFlopsPerMatrixEntry = 0;
    int flopsPerAdd = 1;
    int flopsPerMul = 1;
    int flopsPerMulSame = 1;

    if (arg.dt & GHOST_DT_COMPLEX) {
        flopsPerAdd = 2;
        flopsPerMul = 6;
        flopsPerMulSame = 3;
    }


    ghost_spmv_nflops(&spmvFlopsPerMatrixEntry,arg.dt,arg.dt);

    flops = (double)spmvFlopsPerMatrixEntry*(double)nnz*(double)ncol;

    if (arg.flags & GHOST_SPMV_AXPY) {
        flops += flopsPerAdd*nrow*ncol;
    }
    if (arg.flags & GHOST_SPMV_AXPBY) {
        flops += (flopsPerMul+flopsPerAdd)*nrow*ncol;
    }
    if (arg.flags & GHOST_SPMV_SHIFT) {
        flops += (flopsPerMul+flopsPerAdd)*nrow*ncol;
    } else if (arg.flags & GHOST_SPMV_VSHIFT) {
        flops += (flopsPerMul+flopsPerAdd)*nrow*ncol;
    }
    if (arg.flags & GHOST_SPMV_SCALE) {
        flops += flopsPerMul*nrow*ncol;
    }
    if (arg.flags & GHOST_SPMV_DOT_YY) {
        flops += (flopsPerMulSame+1)*nrow*ncol;
    }
    if (arg.flags & GHOST_SPMV_DOT_XY) {
        flops += (flopsPerMul+flopsPerAdd)*nrow*ncol;
    }
    if (arg.flags & GHOST_SPMV_DOT_XX) {
        flops += (flopsPerMulSame+1)*nrow*ncol;
    }

    *perf = flops/1.e9/time;

    return 0;

}

int ghost_axpy_perf(double *perf, double time, void *varg)
{
    ghost_axpy_perf_args_t arg = *(ghost_axpy_perf_args_t *)varg;

    ghost_lidx_t ncol = arg.ncols;
    ghost_lidx_t nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*3*ncol*nrow/1.e9/time;

    return 0;
}

int ghost_dot_perf(double *perf, double time, void *varg)
{
    ghost_dot_perf_args_t arg = *(ghost_dot_perf_args_t *)varg;

    ghost_lidx_t ncol = arg.ncols;
    ghost_lidx_t nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*ncol*nrow/1.e9/time;

    if (arg.samevec) {
        *perf *= 2;
    }

    return 0;
}

int ghost_scale_perf(double *perf, double time, void *varg)
{
    ghost_scale_perf_args_t arg = *(ghost_scale_perf_args_t *)varg;
    
    ghost_lidx_t ncol = arg.ncols;
    ghost_lidx_t nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*2*ncol*nrow/1.e9/time;

    return 0;
}
