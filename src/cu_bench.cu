#include "ghost/util.h"
#include "ghost/timing.h"
#include "ghost/cu_util.h"
#include "ghost/bench.h"


#define THREADSPERBLOCK 256
#define N (ghost_lidx_t)1e8 

static void dummy(double *a) {
    if (a[(ghost_lidx_t)N>>1] < 0) {
        WARNING_LOG("dummy");
    }
}

#ifdef GHOST_HAVE_CUDA
__global__ static void cu_copy_kernel(double * __restrict__ a, const double * __restrict__ b)
{
    ghost_lidx_t row = blockIdx.x*blockDim.x+threadIdx.x;
    for (; row<N; row+=gridDim.x*blockDim.x) {
        a[row] = b[row];
    }
} 
#endif

extern "C" ghost_error_t ghost_cu_bench_stream(ghost_bench_stream_test_t test, double *bw)
{

    ghost_error_t ret = GHOST_SUCCESS;
    if (test != GHOST_BENCH_STREAM_COPY) {
        WARNING_LOG("Currently only STREAM copy implemented!");
    }

    int i;
    double start,stop;

    double *a = NULL;
    double *b = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&a,N*sizeof(double)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&b,N*sizeof(double)),err,ret);

#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = 0;
        b[i] = i;
    }
    
    ghost_type_t mytype;
    ghost_type_get(&mytype);

    
    if (mytype == GHOST_TYPE_CUDA) {
#ifdef GHOST_HAVE_CUDA
        double *da = NULL;
        double *db = NULL;
        ghost_cu_malloc((void **)&da,N*sizeof(double));
        ghost_cu_malloc((void **)&db,N*sizeof(double));
        ghost_cu_upload(da,a,N*sizeof(double));
        ghost_cu_upload(db,b,N*sizeof(double));

        ghost_timing_wc(&start);
        cu_copy_kernel<<<CEILDIV(N,THREADSPERBLOCK),THREADSPERBLOCK>>> (da,db);
        cudaDeviceSynchronize();
        ghost_timing_wc(&stop);
        ghost_cu_download(a,da,N*sizeof(double));
        ghost_cu_free(da);
        ghost_cu_free(db);
#endif
    }

    *bw = 2*N/1.e9*sizeof(double)/(stop-start);

    dummy(a);

    goto out;

err:
out:
    free(a); a = NULL;
    free(b); b = NULL;

    return ret;
}

