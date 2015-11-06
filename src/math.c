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
#include <float.h>

#define GHOST_MAX_SPMMV_WIDTH INT_MAX

static ghost_mpi_op_t GHOST_MPI_OP_SUM_C = MPI_OP_NULL;
static ghost_mpi_op_t GHOST_MPI_OP_SUM_Z = MPI_OP_NULL;
static ghost_mpi_op_t GHOST_MPI_OP_SUM_DENSEMAT_S = MPI_OP_NULL;
static ghost_mpi_op_t GHOST_MPI_OP_SUM_DENSEMAT_D = MPI_OP_NULL;
static ghost_mpi_op_t GHOST_MPI_OP_SUM_DENSEMAT_C = MPI_OP_NULL;
static ghost_mpi_op_t GHOST_MPI_OP_SUM_DENSEMAT_Z = MPI_OP_NULL;

static void ghost_spmv_selectMode(ghost_context_t * context, ghost_spmv_flags_t *flags, ghost_sparsemat_flags_t matflags);

ghost_error_t ghost_dot(void *res, ghost_densemat_t *vec, ghost_densemat_t *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH)
    GHOST_CALL_RETURN(vec->dot(vec,res,vec2));
#ifdef GHOST_HAVE_MPI
    if (vec->context) {
        GHOST_INSTR_START("reduce")
        ghost_mpi_op_t sumOp;
        ghost_mpi_datatype_t mpiDt;
        ghost_mpi_op_sum(&sumOp,vec->traits.datatype);
        ghost_mpi_datatype(&mpiDt,vec->traits.datatype);
        int v;
        if (vec->context) {
            for (v=0; v<MIN(vec->traits.ncols,vec2->traits.ncols); v++) {
                MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, (char *)res+vec->elSize*v, 1, mpiDt, sumOp, vec->context->mpicomm));
            }
        }
        GHOST_INSTR_STOP("reduce")
    }
#endif

#ifdef GHOST_HAVE_INSTR_TIMING
    ghost_dot_perf_args_t pargs;
    pargs.ncols = vec->traits.ncols;
    if (vec->context) {
        pargs.globnrows = vec->context->gnrows;
    } else {
        pargs.globnrows = vec->traits.nrows;
    }
    pargs.dt = vec->traits.datatype;
    pargs.samevec = vec==vec2;
    
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_dot_perf,(void *)&pargs,sizeof(pargs),GHOST_DOT_PERF_UNIT);
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
    ghost_lidx_t ncolsbackup = res->traits.ncols, remcols = res->traits.ncols, donecols = 0;
    va_list argp_backup;
    va_copy(argp_backup,argp);
    DEBUG_LOG(1,"Performing SpMV");
    ghost_spmvsolver_t solver = NULL;
    ghost_spmv_selectMode(mat->context,flags,mat->traits->flags);
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

    void *alpha = NULL, *beta = NULL, *gamma = NULL, *dot = NULL, *delta = NULL, *eta = NULL;
    ghost_densemat_t *z = NULL;

    // need to process varargs if CHAIN_AXPBY or DOT
    if ((!(*flags & GHOST_SPMV_NOT_REDUCE) && (*flags & GHOST_SPMV_DOT_ANY)) || (*flags & GHOST_SPMV_CHAIN_AXPBY)) {
        GHOST_INSTR_START("dot_reduce");
        if (*flags & GHOST_SPMV_SCALE) {
            alpha = va_arg(argp_backup,void *);
        }
        if (*flags & GHOST_SPMV_AXPBY) {
            beta = va_arg(argp_backup,void *);
        }
        if ((*flags & GHOST_SPMV_SHIFT) || (*flags & GHOST_SPMV_VSHIFT)) {
            gamma = va_arg(argp_backup,void *);
        }
        if (*flags & GHOST_SPMV_DOT_ANY) {
            dot = va_arg(argp_backup,void *);
        }
        if (*flags & GHOST_SPMV_CHAIN_AXPBY) {
            z = va_arg(argp_backup,ghost_densemat_t *);
            delta = va_arg(argp_backup,void *);
            eta = va_arg(argp_backup,void *);
        }

    }
    UNUSED(alpha);
    UNUSED(beta);
    UNUSED(gamma);
    UNUSED(delta);
    UNUSED(eta);

    // TODO only of densemats are compact!
    while (remcols > GHOST_MAX_SPMMV_WIDTH) {

        INFO_LOG("Restricting vector block width!");

        res->traits.ncols = GHOST_MAX_SPMMV_WIDTH;
        invec->traits.ncols = GHOST_MAX_SPMMV_WIDTH;
        
        invec->val += donecols*invec->elSize;
        res->val += donecols*res->elSize;
        
        if (z) {
            z->val += donecols*z->elSize;
            z->traits.ncols = GHOST_MAX_SPMMV_WIDTH;
        }
        GHOST_CALL_RETURN(solver(res,mat,invec,*flags,argp));

        donecols += GHOST_MAX_SPMMV_WIDTH;
        remcols -= donecols;
    }
    res->traits.ncols = remcols;
    invec->traits.ncols = remcols;
    invec->val += donecols*invec->elSize;
    res->val += donecols*res->elSize;
    
    if (z) {
        z->val += donecols*z->elSize;
        z->traits.ncols = remcols;
    }
    
    GHOST_CALL_RETURN(solver(res,mat,invec,*flags,argp));

    res->val -= (ncolsbackup-remcols)*res->elSize;
    invec->val -= (ncolsbackup-remcols)*invec->elSize;
    
    res->traits.ncols = ncolsbackup;
    invec->traits.ncols = ncolsbackup;
    if (z) {
        z->traits.ncols = ncolsbackup;
        z->val -= (ncolsbackup-remcols)*z->elSize;
    }

    if (!(*flags & GHOST_SPMV_NOT_REDUCE) && (*flags & GHOST_SPMV_DOT_ANY)) {
#ifdef GHOST_HAVE_MPI
        ghost_mpi_op_t op;
        ghost_mpi_datatype_t dt;
        ghost_mpi_op_sum(&op,res->traits.datatype);
        ghost_mpi_datatype(&dt,res->traits.datatype);

        MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE, dot, 3*invec->traits.ncols, dt, op, mat->context->mpicomm));
        GHOST_INSTR_STOP("dot_reduce");
#endif
    }

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
    ghost_timing_set_perfFunc(NULL,__ghost_functag,ghost_spmv_perf,(void *)&spmv_perfargs,sizeof(spmv_perfargs),GHOST_SPMV_PERF_UNIT);
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

// taken from stackoverflow.com/questions/29285883
static void ghost_mpi_add_densemat_d(void *in, void *inout, int *len, MPI_Datatype *dtype)
{
    double *invec = in;
    double *inoutvec = inout;
    int nints, naddresses, ntypes;
    int combiner;

    if (*len != 1) {
        ERROR_LOG("Only len==1 supported at the moment!");
        return;
    } 

    MPI_Type_get_envelope(*dtype, &nints, &naddresses, &ntypes, &combiner); 
    if (combiner != MPI_COMBINER_VECTOR) {
        ERROR_LOG("Do not understand composite datatype!");
        return;
    } 

    int vecargs [nints];
    MPI_Aint vecaddrs[naddresses];
    MPI_Datatype vectypes[ntypes];

    MPI_Type_get_contents(*dtype, nints, naddresses, ntypes, 
            vecargs, vecaddrs, vectypes);

    if (vectypes[0] != MPI_DOUBLE) {
        ERROR_LOG("Not a densemat of doubles!");
        return;
    }

    int count    = vecargs[0];
    int blocklen = vecargs[1];
    int stride   = vecargs[2];

    for ( int i=0; i<count; i++ ) {
        for ( int j=0; j<blocklen; j++) {
            inoutvec[i*stride+j] += invec[i*stride+j]; 
        } 
    }
}

static void ghost_mpi_add_densemat_s(void *in, void *inout, int *len, MPI_Datatype *dtype)
{
    float *invec = in;
    float *inoutvec = inout;
    int nints, naddresses, ntypes;
    int combiner;

    if (*len != 1) {
        ERROR_LOG("Only len==1 supported at the moment!");
        return;
    } 

    MPI_Type_get_envelope(*dtype, &nints, &naddresses, &ntypes, &combiner); 
    if (combiner != MPI_COMBINER_VECTOR) {
        ERROR_LOG("Do not understand composite datatype!");
        return;
    } 

    int vecargs [nints];
    MPI_Aint vecaddrs[naddresses];
    MPI_Datatype vectypes[ntypes];

    MPI_Type_get_contents(*dtype, nints, naddresses, ntypes, 
            vecargs, vecaddrs, vectypes);

    if (vectypes[0] != MPI_FLOAT) {
        ERROR_LOG("Not a densemat of floats!");
        return;
    }

    int count    = vecargs[0];
    int blocklen = vecargs[1];
    int stride   = vecargs[2];

    for ( int i=0; i<count; i++ ) {
        for ( int j=0; j<blocklen; j++) {
            inoutvec[i*stride+j] += invec[i*stride+j]; 
        } 
    }
}

static void ghost_mpi_add_densemat_z(void *in, void *inout, int *len, MPI_Datatype *dtype)
{
    complex double *invec = in;
    complex double *inoutvec = inout;
    int nints, naddresses, ntypes;
    int combiner;

    if (*len != 1) {
        ERROR_LOG("Only len==1 supported at the moment!");
        return;
    } 

    MPI_Type_get_envelope(*dtype, &nints, &naddresses, &ntypes, &combiner); 
    if (combiner != MPI_COMBINER_VECTOR) {
        ERROR_LOG("Do not understand composite datatype!");
        return;
    } 

    int vecargs [nints];
    MPI_Aint vecaddrs[naddresses];
    MPI_Datatype vectypes[ntypes];

    MPI_Type_get_contents(*dtype, nints, naddresses, ntypes, 
            vecargs, vecaddrs, vectypes);

    ghost_mpi_datatype_t dt_z;
    ghost_mpi_datatype(&dt_z,(ghost_datatype_t)(GHOST_DT_DOUBLE|GHOST_DT_COMPLEX));
    // the complex double MPI datatype is derived and must be free'd
    MPI_Type_free(&vectypes[0]);

    int count    = vecargs[0];
    int blocklen = vecargs[1];
    int stride   = vecargs[2];

    for ( int i=0; i<count; i++ ) {
        for ( int j=0; j<blocklen; j++) {
            inoutvec[i*stride+j] += invec[i*stride+j]; 
        } 
    }
}

static void ghost_mpi_add_densemat_c(void *in, void *inout, int *len, MPI_Datatype *dtype)
{
    complex float *invec = in;
    complex float *inoutvec = inout;
    int nints, naddresses, ntypes;
    int combiner;

    if (*len != 1) {
        ERROR_LOG("Only len==1 supported at the moment!");
        return;
    } 

    MPI_Type_get_envelope(*dtype, &nints, &naddresses, &ntypes, &combiner); 
    if (combiner != MPI_COMBINER_VECTOR) {
        ERROR_LOG("Do not understand composite datatype!");
        return;
    } 

    int vecargs [nints];
    MPI_Aint vecaddrs[naddresses];
    MPI_Datatype vectypes[ntypes];

    MPI_Type_get_contents(*dtype, nints, naddresses, ntypes, 
            vecargs, vecaddrs, vectypes);

    ghost_mpi_datatype_t dt_c;
    ghost_mpi_datatype(&dt_c,(ghost_datatype_t)(GHOST_DT_FLOAT|GHOST_DT_COMPLEX));
    // the complex double MPI datatype is derived and must be free'd
    MPI_Type_free(&vectypes[0]);

    int count    = vecargs[0];
    int blocklen = vecargs[1];
    int stride   = vecargs[2];

    for ( int i=0; i<count; i++ ) {
        for ( int j=0; j<blocklen; j++) {
            inoutvec[i*stride+j] += invec[i*stride+j]; 
        } 
    }
}
#endif

static void ghost_spmv_selectMode(ghost_context_t * context, ghost_spmv_flags_t *flags, ghost_sparsemat_flags_t matflags)
{
    int nranks;
    ghost_nrank(&nranks,context->mpicomm);
    if (!(*flags & GHOST_SPMV_MODES_ALL)) { // no mode specified
#ifdef GHOST_HAVE_MPI
        if (nranks == 1) {
            *flags |= (ghost_spmv_flags_t)GHOST_SPMV_MODE_NOMPI;
        } else {
            if (matflags & GHOST_SPARSEMAT_NOT_STORE_SPLIT) {
                *flags |= (ghost_spmv_flags_t)GHOST_SPMV_MODE_VECTOR;
            } else {
                *flags |= (ghost_spmv_flags_t)GHOST_SPMV_MODE_OVERLAP;
            }
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

ghost_error_t ghost_mpi_op_sum(ghost_mpi_op_t * op, ghost_datatype_t datatype)
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

ghost_error_t ghost_mpi_op_densemat_sum(ghost_mpi_op_t * op, ghost_datatype_t datatype)
{
    if (!op) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
#ifdef GHOST_HAVE_MPI
    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_COMPLEX) {
            *op = GHOST_MPI_OP_SUM_DENSEMAT_C;
        } else {
            *op = GHOST_MPI_OP_SUM_DENSEMAT_S;
        }
    } else {
        if (datatype & GHOST_DT_COMPLEX) {
            *op = GHOST_MPI_OP_SUM_DENSEMAT_Z;
        } else {
            *op = GHOST_MPI_OP_SUM_DENSEMAT_D;
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
    MPI_CALL_RETURN(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_densemat_d,1,&GHOST_MPI_OP_SUM_DENSEMAT_D));
    MPI_CALL_RETURN(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_densemat_s,1,&GHOST_MPI_OP_SUM_DENSEMAT_S));
    MPI_CALL_RETURN(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_densemat_c,1,&GHOST_MPI_OP_SUM_DENSEMAT_C));
    MPI_CALL_RETURN(MPI_Op_create((MPI_User_function *)&ghost_mpi_add_densemat_z,1,&GHOST_MPI_OP_SUM_DENSEMAT_Z));
#else
    UNUSED(GHOST_MPI_OP_SUM_C);
    UNUSED(GHOST_MPI_OP_SUM_Z);
    UNUSED(GHOST_MPI_OP_SUM_DENSEMAT_S);
    UNUSED(GHOST_MPI_OP_SUM_DENSEMAT_D);
    UNUSED(GHOST_MPI_OP_SUM_DENSEMAT_C);
    UNUSED(GHOST_MPI_OP_SUM_DENSEMAT_Z);
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_mpi_operations_destroy()
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_C));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_Z));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_DENSEMAT_S));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_DENSEMAT_D));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_DENSEMAT_C));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_DENSEMAT_Z));
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
    if (arg.flags & GHOST_SPMV_CHAIN_AXPBY) {
        flops += (2*flopsPerMul+flopsPerAdd)*nrow*ncol;
    }

    *perf = flops/1.e9/time;

    return 0;

}

int ghost_axpbypcz_perf(double *perf, double time, void *varg)
{
    ghost_axpbypcz_perf_args_t arg = *(ghost_axpbypcz_perf_args_t *)varg;

    ghost_lidx_t ncol = arg.ncols;
    ghost_lidx_t nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*4*ncol*nrow/1.e9/time;

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

    if (!arg.samevec) {
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

bool ghost_iszero(void *vnumber, ghost_datatype_t dt) 
{
    if (dt & GHOST_DT_COMPLEX) {
        if (dt & GHOST_DT_FLOAT) {
            complex float number = *(complex float *)vnumber;
            return fabsf(crealf(number)) < FLT_MIN && fabsf(cimagf(number)) < FLT_MIN;
        } else {
            complex double number = *(complex double *)vnumber;
            return fabs(creal(number)) < DBL_MIN && fabs(cimag(number)) < DBL_MIN;
        }
    } else {
        if (dt & GHOST_DT_FLOAT) {
            float number = *(float *)vnumber;
            return number < FLT_MIN;
        } else {
            double number = *(double *)vnumber;
            return number < DBL_MIN;
        }
    }

}

bool ghost_isone(void *vnumber, ghost_datatype_t dt) 
{
    if (dt & GHOST_DT_COMPLEX) {
        if (dt & GHOST_DT_FLOAT) {
            complex float number = *(complex float *)vnumber;
            return fabsf(crealf(number)-1.f) < FLT_MIN && fabsf(cimagf(number)) < FLT_MIN;
        } else {
            complex double number = *(complex double *)vnumber;
            return fabs(creal(number)-1.) < DBL_MIN && fabs(cimag(number)) < DBL_MIN;
        }
    } else {
        if (dt & GHOST_DT_FLOAT) {
            float number = *(float *)vnumber;
            return (number-1.f) < FLT_MIN;
        } else {
            double number = *(double *)vnumber;
            return (number-1.) < DBL_MIN;
        }
    }
}
