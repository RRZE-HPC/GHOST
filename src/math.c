#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/densemat.h"
#include "ghost/locality.h"
#include "ghost/instr.h"
#include "ghost/log.h"
#include <math.h>
#include <complex.h>
#include <float.h>

static ghost_mpi_op GHOST_MPI_OP_SUM_C = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_Z = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_DENSEMAT_S = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_DENSEMAT_D = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_DENSEMAT_C = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_DENSEMAT_Z = MPI_OP_NULL;

/*
ghost_error ghost_dot(void *res, ghost_densemat *vec, ghost_densemat *vec2)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH)
    GHOST_CALL_RETURN(vec->dot(vec,res,vec2));
#ifdef GHOST_HAVE_MPI
    if (vec->context) {
        GHOST_INSTR_START("reduce")
        ghost_mpi_op sumOp;
        ghost_mpi_datatype mpiDt;
        ghost_mpi_op_sum(&sumOp,vec->traits.datatype);
        ghost_mpi_datatype_get(&mpiDt,vec->traits.datatype);
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
    ghost_dot_perf_args pargs;
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

}*/

ghost_error ghost_normalize(ghost_densemat *vec)
{
    ghost_lidx ncols = vec->traits.ncols;
    ghost_lidx c;

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

    ghost_mpi_datatype dt_z;
    ghost_mpi_datatype_get(&dt_z,(ghost_datatype)(GHOST_DT_DOUBLE|GHOST_DT_COMPLEX));
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

    ghost_mpi_datatype dt_c;
    ghost_mpi_datatype_get(&dt_c,(ghost_datatype)(GHOST_DT_FLOAT|GHOST_DT_COMPLEX));
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

ghost_error ghost_mpi_op_sum(ghost_mpi_op * op, ghost_datatype datatype)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;

}

ghost_error ghost_mpi_op_densemat_sum(ghost_mpi_op * op, ghost_datatype datatype)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}


ghost_error ghost_mpi_operations_create()
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return GHOST_SUCCESS;
}

ghost_error ghost_mpi_operations_destroy()
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TEARDOWN);
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_C));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_Z));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_DENSEMAT_S));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_DENSEMAT_D));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_DENSEMAT_C));
    MPI_CALL_RETURN(MPI_Op_free(&GHOST_MPI_OP_SUM_DENSEMAT_Z));
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TEARDOWN);
    return GHOST_SUCCESS;
}

ghost_error ghost_spmv_nflops(int *nFlops, ghost_datatype m_t, ghost_datatype v_t)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

int ghost_spmv_perf(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    ghost_spmv_perf_args arg = *(ghost_spmv_perf_args *)varg;

    ghost_lidx ncol = arg.vecncols;
    double flops;
    ghost_gidx nnz = arg.globalnnz;
    ghost_gidx nrow = arg.globalrows;
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    return 0;

}

int ghost_axpbypcz_perf(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    ghost_axpbypcz_perf_args arg = *(ghost_axpbypcz_perf_args *)varg;

    ghost_lidx ncol = arg.ncols;
    ghost_lidx nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*4*ncol*nrow/1.e9/time;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    return 0;
}

int ghost_axpby_perf(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    ghost_axpby_perf_args arg = *(ghost_axpby_perf_args *)varg;

    ghost_lidx ncol = arg.ncols;
    ghost_lidx nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*3*ncol*nrow/1.e9/time;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    return 0;
}

int ghost_axpy_perf(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    ghost_axpy_perf_args arg = *(ghost_axpy_perf_args *)varg;

    ghost_lidx ncol = arg.ncols;
    ghost_lidx nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*3*ncol*nrow/1.e9/time;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    return 0;
}

int ghost_dot_perf(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    ghost_dot_perf_args arg = *(ghost_dot_perf_args *)varg;

    ghost_lidx ncol = arg.ncols;
    ghost_lidx nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*ncol*nrow/1.e9/time;

    if (!arg.samevec) {
        *perf *= 2;
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    return 0;
}

int ghost_scale_perf(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    ghost_scale_perf_args arg = *(ghost_scale_perf_args *)varg;
    
    ghost_lidx ncol = arg.ncols;
    ghost_lidx nrow = arg.globnrows;

    size_t size;
    ghost_datatype_size(&size,arg.dt);

    *perf = size*2*ncol*nrow/1.e9/time;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    return 0;
}

bool ghost_iszero(void *vnumber, ghost_datatype dt) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    bool ret = false;

    if (dt & GHOST_DT_COMPLEX) {
        if (dt & GHOST_DT_FLOAT) {
            complex float number = *(complex float *)vnumber;
            ret = fabsf(crealf(number)) < FLT_MIN && fabsf(cimagf(number)) < FLT_MIN;
        } else {
            complex double number = *(complex double *)vnumber;
            ret = fabs(creal(number)) < DBL_MIN && fabs(cimag(number)) < DBL_MIN;
        }
    } else {
        if (dt & GHOST_DT_FLOAT) {
            float number = *(float *)vnumber;
            ret = number < FLT_MIN;
        } else {
            double number = *(double *)vnumber;
            ret = number < DBL_MIN;
        }
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return ret;
}

bool ghost_isone(void *vnumber, ghost_datatype dt) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    bool ret = false;
    
    if (dt & GHOST_DT_COMPLEX) {
        if (dt & GHOST_DT_FLOAT) {
            complex float number = *(complex float *)vnumber;
            ret = fabsf(crealf(number)-1.f) < FLT_MIN && fabsf(cimagf(number)) < FLT_MIN;
        } else {
            complex double number = *(complex double *)vnumber;
            ret = fabs(creal(number)-1.) < DBL_MIN && fabs(cimag(number)) < DBL_MIN;
        }
    } else {
        if (dt & GHOST_DT_FLOAT) {
            float number = *(float *)vnumber;
            ret = (number-1.f) < FLT_MIN;
        } else {
            double number = *(double *)vnumber;
            ret = (number-1.) < DBL_MIN;
        }
    }
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return ret;
}
