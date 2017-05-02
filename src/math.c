#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/densemat.h"
#include "ghost/densemat_cm.h"
#include "ghost/densemat_rm.h"
#include "ghost/locality.h"
#include "ghost/instr.h"
#include "ghost/log.h"
#include <math.h>
#include <complex.h>
#include <float.h>
#include "ghost/compatibility_check.h"

static ghost_mpi_op GHOST_MPI_OP_SUM_C = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_Z = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_DENSEMAT_S = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_DENSEMAT_D = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_DENSEMAT_C = MPI_OP_NULL;
static ghost_mpi_op GHOST_MPI_OP_SUM_DENSEMAT_Z = MPI_OP_NULL;

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

int ghost_kacz_perf(double *perf, double time, void *varg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);

    UNUSED(time);
    ghost_kacz_perf_args arg = *(ghost_kacz_perf_args *)varg;
    UNUSED(arg);
    *perf = 0;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_BENCH);
    return 0;

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

    if (!(arg.flags & GHOST_SPMV_AXPY) && !(arg.flags & GHOST_SPMV_AXPBY)) { // y=Ax case, we have one ADD less in each row as tmp=0 before the inner loop
        flops -= flopsPerAdd*nrow*ncol;
    }
    if (arg.flags & GHOST_SPMV_AXPBY) {
        flops += flopsPerMul*nrow*ncol;
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
            ret = fabsf(number) < FLT_MIN;
        } else {
            double number = *(double *)vnumber;
            ret = fabs(number) < DBL_MIN;
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


ghost_error ghost_axpy(ghost_densemat *y, ghost_densemat *x, void *a)
{
    ghost_error ret;

    //////////////// check compatibility /////////////
    ghost_compatible_vec_vec check = GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;
    check.OUT = y;
    check.C = x;
    ret = ghost_check_vec_vec_compatibility(&check);
    ///////////////////////////////////////////////////

    GHOST_DENSEMAT_CHECK_SIMILARITY(y,x);
    ghost_location commonlocation = (ghost_location)(y->traits.location & x->traits.location);

    typedef ghost_error (*ghost_axpy_kernel)(ghost_densemat*, ghost_densemat*, void*);
    ghost_axpy_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_axpy;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_axpy;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_axpy;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_axpy;
#endif

    SELECT_BLAS1_KERNEL(kernels,commonlocation,y->traits.compute_at,y->traits.storage,ret,y,x,a);

    return ret;
}

ghost_error ghost_vaxpy(ghost_densemat *y, ghost_densemat *x, void *a)
{
    ghost_error ret;

    //////////////// check compatibility /////////////
    ghost_compatible_vec_vec check = GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;
    check.OUT = y;
    check.C = x;
    ret = ghost_check_vec_vec_compatibility(&check);
    ///////////////////////////////////////////////////

    GHOST_DENSEMAT_CHECK_SIMILARITY(y,x);

    ghost_location commonlocation = (ghost_location)(y->traits.location & x->traits.location);

    typedef ghost_error (*ghost_vaxpy_kernel)(ghost_densemat*, ghost_densemat*, void*);
    ghost_vaxpy_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_vaxpy_selector;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_vaxpy_selector;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_vaxpy;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_vaxpy;
#endif

    SELECT_BLAS1_KERNEL(kernels,commonlocation,y->traits.compute_at,y->traits.storage,ret,y,x,a);

    return ret;
}

ghost_error ghost_axpby(ghost_densemat *y, ghost_densemat *x, void *a, void *b)
{
    ghost_error ret;

    //////////////// check compatibility /////////////
    ghost_compatible_vec_vec check = GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;
    check.OUT = y;
    check.C = x;
    ret = ghost_check_vec_vec_compatibility(&check);
    ///////////////////////////////////////////////////

    GHOST_DENSEMAT_CHECK_SIMILARITY(y,x);

    ghost_location commonlocation = (ghost_location)(y->traits.location & x->traits.location);

    typedef ghost_error (*ghost_axpby_kernel)(ghost_densemat*, ghost_densemat*, void*, void*);
    ghost_axpby_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_axpby;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_axpby;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_axpby;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_axpby;
#endif

    SELECT_BLAS1_KERNEL(kernels,commonlocation,y->traits.compute_at,y->traits.storage,ret,y,x,a,b);

    return ret;
}

ghost_error ghost_vaxpby(ghost_densemat *y, ghost_densemat *x, void *a, void *b)
{
    ghost_error ret;

    //////////////// check compatibility /////////////
    ghost_compatible_vec_vec check = GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;
    check.OUT = y;
    check.C = x;
    ret = ghost_check_vec_vec_compatibility(&check);
    ///////////////////////////////////////////////////

    GHOST_DENSEMAT_CHECK_SIMILARITY(y,x);

    ghost_location commonlocation = (ghost_location)(y->traits.location & x->traits.location);

     typedef ghost_error (*ghost_vaxpby_kernel)(ghost_densemat*, ghost_densemat*, void*, void*);
    ghost_vaxpby_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_vaxpby_selector;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_vaxpby_selector;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_vaxpby;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_vaxpby;
#endif

    SELECT_BLAS1_KERNEL(kernels,commonlocation,y->traits.compute_at,y->traits.storage,ret,y,x,a,b);

    return ret;
}

ghost_error ghost_axpbypcz(ghost_densemat *y, ghost_densemat *x, void *a, void *b, ghost_densemat *z, void *c)
{

    ghost_error ret;

    //////////////// check compatibility /////////////
    ghost_compatible_vec_vec check = GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;
    check.OUT = y;
    check.C = x;
    check.D = z;
    ret = ghost_check_vec_vec_compatibility(&check);
    ///////////////////////////////////////////////////

    GHOST_DENSEMAT_CHECK_SIMILARITY(y,x);
    GHOST_DENSEMAT_CHECK_SIMILARITY(y,z);

    ghost_location commonlocation = (ghost_location)(y->traits.location & x->traits.location);

    typedef ghost_error (*ghost_axpbypcz_kernel)(ghost_densemat*, ghost_densemat*, void*, void*, ghost_densemat*, void*);
    ghost_axpbypcz_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_axpbypcz;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_axpbypcz;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_axpbypcz;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_axpbypcz;
#endif

    SELECT_BLAS1_KERNEL(kernels,commonlocation,y->traits.compute_at,y->traits.storage,ret,y,x,a,b,z,c);

    return ret;
}

ghost_error ghost_vaxpbypcz(ghost_densemat *y, ghost_densemat *x, void *a, void *b, ghost_densemat *z, void *c)
{

    ghost_error ret;

    //////////////// check compatibility /////////////
    ghost_compatible_vec_vec check = GHOST_COMPATIBLE_VEC_VEC_INITIALIZER;
    check.OUT = y;
    check.C = x;
    check.D = z;
    ret = ghost_check_vec_vec_compatibility(&check);
    ///////////////////////////////////////////////////

    GHOST_DENSEMAT_CHECK_SIMILARITY(y,x);
    GHOST_DENSEMAT_CHECK_SIMILARITY(y,z);

    ghost_location commonlocation = (ghost_location)(y->traits.location & x->traits.location);

    typedef ghost_error (*ghost_vaxpbypcz_kernel)(ghost_densemat*, ghost_densemat*, void*, void*, ghost_densemat*, void*);
    ghost_vaxpbypcz_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_vaxpbypcz_selector;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_vaxpbypcz_selector;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_vaxpbypcz;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_vaxpbypcz;
#endif

    SELECT_BLAS1_KERNEL(kernels,commonlocation,y->traits.compute_at,y->traits.storage,ret,y,x,a,b,z,c);

    return ret;
}

ghost_error ghost_scale(ghost_densemat *x, void *a)
{
    ghost_error ret;

    typedef ghost_error (*ghost_scale_kernel)(ghost_densemat*, void*);
    ghost_scale_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_scale;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_scale;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_scale;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_scale;
#endif

    SELECT_BLAS1_KERNEL(kernels,x->traits.location,x->traits.compute_at,x->traits.storage,ret,x,a);

    return ret;
}

ghost_error ghost_vscale(ghost_densemat *x, void *a)
{
    ghost_error ret;

    typedef ghost_error (*ghost_vscale_kernel)(ghost_densemat*, void*);
    ghost_vscale_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_vscale_selector;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_vscale_selector;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_vscale;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_vscale;
#endif

    SELECT_BLAS1_KERNEL(kernels,x->traits.location,x->traits.compute_at,x->traits.storage,ret,x,a);

    return ret;
}

ghost_error ghost_normalize(ghost_densemat *x)
{
    ghost_error ret;

    typedef ghost_error (*ghost_normalize_kernel)(ghost_densemat*);
    ghost_normalize_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_normalize_selector;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_normalize_selector;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_normalize_selector;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_normalize_selector;
#endif

    SELECT_BLAS1_KERNEL(kernels,x->traits.location,x->traits.compute_at,x->traits.storage,ret,x);

    return ret;
}

ghost_error ghost_conj(ghost_densemat *x)
{
    ghost_error ret;

    typedef ghost_error (*ghost_conj_kernel)(ghost_densemat*);
    ghost_conj_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_conj_selector;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_conj_selector;
#ifdef GHOST_HAVE_CUDA
    kernels[GHOST_DEVICE_IDX][GHOST_RM_IDX] = &ghost_densemat_cu_rm_conj;
    kernels[GHOST_DEVICE_IDX][GHOST_CM_IDX] = &ghost_densemat_cu_cm_conj;
#endif

    SELECT_BLAS1_KERNEL(kernels,x->traits.location,x->traits.compute_at,x->traits.storage,ret,x);

    return ret;
}

ghost_error ghost_norm(void *norm, ghost_densemat *x, void *pow)
{
    ghost_error ret;

    typedef ghost_error (*ghost_norm_kernel)(ghost_densemat*, void*, void*);
    ghost_norm_kernel kernels[2][2] = {{NULL,NULL},{NULL,NULL}};
    kernels[GHOST_HOST_IDX][GHOST_RM_IDX] = &ghost_densemat_rm_norm_selector;
    kernels[GHOST_HOST_IDX][GHOST_CM_IDX] = &ghost_densemat_cm_norm_selector;
    // TODO GPU implementation

    SELECT_BLAS1_KERNEL(kernels,x->traits.location,x->traits.compute_at,x->traits.storage,ret,x,norm,pow);

    return ret;
}

ghost_error ghost_nrm2(void *norm, ghost_densemat *x)
{
    GHOST_CALL_RETURN(ghost_dot(norm,x,x));
    if (x->traits.datatype & GHOST_DT_COMPLEX) {
        if (x->traits.datatype & GHOST_DT_DOUBLE) {
            (*(double complex *)norm) = csqrt(*(double complex *)norm);
        } else {
            (*(float complex *)norm) = csqrtf(*(float complex *)norm);
        }
    } else {
        if (x->traits.datatype & GHOST_DT_DOUBLE) {
            (*(double *)norm) = sqrt(*(double *)norm);
        } else {
            (*(float *)norm) = sqrtf(*(float *)norm);
        }
    }

    return GHOST_SUCCESS;
}

