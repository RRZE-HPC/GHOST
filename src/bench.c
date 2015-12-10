#include "ghost/util.h"
#include "ghost/cu_util.h"
#include "ghost/bench.h"
#include "ghost/cu_bench.h"

#define N (ghost_lidx_t)1e8 

static void dummy(double *a) {
    if (a[(ghost_lidx_t)N>>1] < 0) {
        WARNING_LOG("dummy");
    }
}

static void copy_kernel(double * __restrict__ a, const double * __restrict__ b)
{
    ghost_lidx_t i;
#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = b[i];
    }

}

ghost_error_t ghost_bench_stream(ghost_bench_stream_test_t test, double *bw)
{
    ghost_type_t mytype;
    ghost_type_get(&mytype);
    
    if (mytype == GHOST_TYPE_CUDA) {
#if GHOST_HAVE_CUDA
        return ghost_cu_bench_stream(test,bw);
#endif
    }

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
    
    ghost_timing_wc(&start);
    copy_kernel(a,b);
    ghost_timing_wc(&stop);

    *bw = 2*N/1.e9*sizeof(double)/(stop-start);

    dummy(a);

    goto out;

err:
out:
    free(a); a = NULL;
    free(b); b = NULL;

    return ret;
}

ghost_error_t ghost_bench_pingpong(double *bw)
{

    UNUSED(bw);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_bench_peakperformance(double *gf)
{

    UNUSED(gf);

    return GHOST_SUCCESS;
}
