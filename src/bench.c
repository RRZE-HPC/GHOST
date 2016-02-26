#include "ghost/util.h"
#include "ghost/machine.h"
#include "ghost/bench.h"
#include "ghost/task.h"
#include "ghost/timing.h"
#include "ghost/cu_bench.h"

#include <immintrin.h>

#define N PAD((ghost_lidx)1e8,16)
#define NITER 40

static void dummy(double *a) {
    if (a[(ghost_lidx)N>>1] < 0) {
        WARNING_LOG("dummy");
    }
}

static void ghost_copy_kernel(double * __restrict__ a, const double * __restrict__ b)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_KERNEL|GHOST_FUNCTYPE_BENCH);
    ghost_lidx i;

#ifdef GHOST_BUILD_MIC
    __m512d bv;
#pragma omp parallel for private(bv)
    for (i=0; i<N; i+=8) {
        bv = _mm512_load_pd(&b[i]);
        _mm512_storenrngo_pd(&a[i],bv);
    }
#elif defined(GHOST_BUILD_AVX)
    __m256d bv;
#pragma omp parallel for private(bv)
    for (i=0; i<N; i+=4) {
        bv = _mm256_load_pd(&b[i]);
        _mm256_stream_pd(&a[i],bv);
    }
#elif defined(GHOST_BUILD_SSE)
    __m128d bv;
#pragma omp parallel for private(bv)
    for (i=0; i<N; i+=2) {
        bv = _mm_load_pd(&b[i]);
        _mm_stream_pd(&a[i],bv);
    }
#else
    PERFWARNING_LOG("Cannot guarantee streaming stores for copy benchmark!");
#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = b[i];
    }
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_KERNEL|GHOST_FUNCTYPE_BENCH);
}

ghost_error ghost_bench_stream(ghost_bench_stream_test_t test, double *bw)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_BENCH);
    ghost_type mytype;
    ghost_type_get(&mytype);
    
    if (mytype == GHOST_TYPE_CUDA) {
#ifdef GHOST_HAVE_CUDA
        return ghost_cu_bench_stream(test,bw);
#endif
    }
        
    ghost_task *cur = NULL;
    ghost_task_cur(&cur);
    if (!cur) {
        WARNING_LOG("STREAM benchmark called outside a GHOST task. The reported bandwidth will probably not be meaningful!");
    }

    ghost_error ret = GHOST_SUCCESS;
    if (test != GHOST_BENCH_STREAM_COPY) {
        WARNING_LOG("Currently only STREAM copy implemented!");
    }

    int i;
    double start,stop;

    double *a = NULL;
    double *b = NULL;
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&a,N*sizeof(double),ghost_machine_alignment()),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&b,N*sizeof(double),ghost_machine_alignment()),err,ret);

#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = 0;
        b[i] = i;
    }
   
    // warm up 
    ghost_copy_kernel(a,b);

    ghost_timing_wc(&start);
    for (i=0; i<NITER; i++) {
        ghost_copy_kernel(a,b);
    }
    ghost_timing_wc(&stop);

    *bw = 2*N/1.e9*NITER*sizeof(double)/(stop-start);

    dummy(a);

    goto out;

err:
out:
    free(a); a = NULL;
    free(b); b = NULL;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_BENCH);
    return ret;
}

ghost_error ghost_bench_pingpong(double *bw)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_BENCH);

    UNUSED(bw);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_BENCH);

    return GHOST_SUCCESS;
}

ghost_error ghost_bench_peakperformance(double *gf)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_BENCH);

    UNUSED(gf);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_BENCH);

    return GHOST_SUCCESS;
}
