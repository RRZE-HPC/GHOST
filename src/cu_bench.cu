#include "ghost/util.h"
#include "ghost/timing.h"
#include "ghost/cu_util.h"
#include "ghost/cu_sell_kernel.h"
#include "ghost/bench.h"
#include "ghost/densemat.h"
#include "ghost/funcptr_wrappers.h"
#include "ghost/dot.h"

#define NITER 20
#define THREADSPERBLOCK 256
#define N (ghost_lidx)1e8 

static void dummy(double *a) {
    if (a[(ghost_lidx)N>>1] < 0) {
        WARNING_LOG("dummy");
    }
}

#ifdef GHOST_HAVE_CUDA
__global__ static void cu_copy_kernel(double * __restrict__ a, const double * __restrict__ b)
{
    ghost_lidx row = blockIdx.x*blockDim.x+threadIdx.x;
    for (; row<N; row+=gridDim.x*blockDim.x) {
        a[row] = b[row];
    }
} 

__global__ static void cu_triad_kernel(double * __restrict__ a, const double * __restrict__ b, const double * __restrict__ c, const double s)
{
    ghost_lidx row = blockIdx.x*blockDim.x+threadIdx.x;
    for (; row<N; row+=gridDim.x*blockDim.x) {
        a[row] = b[row]+s*c[row];
    }
} 

__global__ static void cu_store_kernel(double * __restrict__ a, const double s)
{
    ghost_lidx row = blockIdx.x*blockDim.x+threadIdx.x;
    for (; row<N; row+=gridDim.x*blockDim.x) {
        a[row] = s;
    }
} 
#endif

extern "C" ghost_error ghost_cu_bench_stream(ghost_bench_stream_test_t test, double *mean_bw, double *max_bw)
{

    ghost_error ret = GHOST_SUCCESS;

    int i;
    double start,stop,start1,stop1,tmin;
    tmin=1e99;
    double *a = NULL;
    double *b = NULL;
    double *c = NULL;
    double s = 2.2;
    GHOST_CALL_GOTO(ghost_malloc((void **)&a,N*sizeof(double)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&b,N*sizeof(double)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&c,N*sizeof(double)),err,ret);

#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = 0;
        b[i] = i;
        c[i] = i+1;
    }
    
    ghost_type mytype;
    ghost_type_get(&mytype);

    
    if (mytype == GHOST_TYPE_CUDA) {
#ifdef GHOST_HAVE_CUDA
        double *da = NULL;
        double *db = NULL;
        double *dc = NULL;
        ghost_cu_malloc((void **)&da,N*sizeof(double));
        ghost_cu_malloc((void **)&db,N*sizeof(double));
        ghost_cu_malloc((void **)&dc,N*sizeof(double));
        ghost_cu_upload(da,a,N*sizeof(double));
        ghost_cu_upload(db,b,N*sizeof(double));
        ghost_cu_upload(dc,c,N*sizeof(double));

        ghost_timing_wc(&start);
        for (int iter=0; iter<NITER; iter++) {
        ghost_timing_wc(&start1);
        switch (test) {
            case GHOST_BENCH_STREAM_COPY:
                cu_copy_kernel<<<CEILDIV(N,THREADSPERBLOCK),THREADSPERBLOCK>>> (da,db);
                break;
            case GHOST_BENCH_STREAM_TRIAD:
                cu_triad_kernel<<<CEILDIV(N,THREADSPERBLOCK),THREADSPERBLOCK>>> (da,db,dc,s);
                break;
            case GHOST_BENCH_STREAM_STORE:
                cu_store_kernel<<<CEILDIV(N,THREADSPERBLOCK),THREADSPERBLOCK>>> (da,s);
                break;
            case GHOST_BENCH_STREAM_LOAD:
                ghost_densemat *a_dm, *b_dm;
                ghost_densemat_traits dmtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
                dmtraits.nrows = N;
                ghost_densemat_create(&a_dm,NULL,dmtraits);
                ghost_densemat_create(&b_dm,NULL,dmtraits);
                ghost_densemat_view_plain(a_dm,da,1);
                ghost_densemat_view_plain(b_dm,db,1);
                ghost_dot(&s,a_dm,b_dm,MPI_COMM_NULL);
                ghost_densemat_destroy(a_dm);
                ghost_densemat_destroy(b_dm);
                break;
        }
        cudaDeviceSynchronize();
        ghost_timing_wc(&stop1);
        tmin=MIN(tmin,stop1-start1);
        }
        ghost_timing_wc(&stop);
        GHOST_CALL_GOTO(ghost_cu_download(a,da,N*sizeof(double)),err,ret);
        ghost_cu_free(da);
        ghost_cu_free(db);
        ghost_cu_free(dc);
#endif
    }

    switch (test) {
        case GHOST_BENCH_STREAM_COPY:
            *mean_bw = 2*N/1.e9*NITER*sizeof(double)/(stop-start);
            *max_bw = 2*N/1.e9*sizeof(double)/tmin;
            break;
        case GHOST_BENCH_STREAM_TRIAD:
            *mean_bw = 3*N/1.e9*NITER*sizeof(double)/(stop-start);
            *max_bw = 3*N/1.e9*sizeof(double)/tmin;
            break;
        case GHOST_BENCH_STREAM_STORE:
            *mean_bw = N/1.e9*NITER*sizeof(double)/(stop-start);
            *max_bw = N/1.e9*sizeof(double)/tmin;
            break;
        case GHOST_BENCH_STREAM_LOAD:
            *mean_bw = 2*N/1.e9*NITER*sizeof(double)/(stop-start);
            *max_bw = 2*N/1.e9*sizeof(double)/tmin;
            break;
    }

    dummy(a);

    goto out;

err:
out:
    free(a); a = NULL;
    free(b); b = NULL;
    free(c); c = NULL;

    return ret;
}

