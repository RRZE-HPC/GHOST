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

static void ghost_load_kernel(const double * __restrict__ a, double * s)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_KERNEL|GHOST_FUNCTYPE_BENCH);
    ghost_lidx i;

#ifdef GHOST_BUILD_MIC
    __m512d av;
    __m512d sv_priv, sv;
    sv_priv = _mm512_setzero_pd();
    sv = _mm512_setzero_pd();
#pragma omp parallel private(av,sv_priv)
    {
#pragma omp for
        for (i=0; i<N; i+=8) {
            av = _mm512_load_pd(&a[i]);
            sv_priv = _mm512_add_pd(sv_priv,av);
        }
#pragma omp critical
        sv = _mm512_add_pd(sv,sv_priv);
    }
    *s = ((double *)&sv)[0]+((double *)&sv)[1]+((double *)&sv)[2]+((double *)&sv)[3]+((double *)&sv)[4]+((double *)&sv)[5]+((double *)&sv)[6]+((double *)&sv)[7];
#elif defined(GHOST_BUILD_AVX)
    __m256d av;
    __m256d sv_priv, sv;
    sv_priv = _mm256_setzero_pd();
    sv = _mm256_setzero_pd();
#pragma omp parallel private(av,sv_priv)
    {
#pragma omp for
        for (i=0; i<N; i+=8) {
            av = _mm256_load_pd(&a[i]);
            sv_priv = _mm256_add_pd(sv_priv,av);
        }
#pragma omp critical
        sv = _mm256_add_pd(sv,sv_priv);
    }
    *s = ((double *)&sv)[0]+((double *)&sv)[1]+((double *)&sv)[2]+((double *)&sv)[3]+((double *)&sv)[4];
#elif defined(GHOST_BUILD_SSE)
    __m128d av;
    __m128d sv_priv, sv;
    sv_priv = _mm_setzero_pd();
    sv = _mm_setzero_pd();
#pragma omp parallel private(av,sv_priv)
    {
#pragma omp for
        for (i=0; i<N; i+=8) {
            av = _mm_load_pd(&a[i]);
            sv_priv = _mm_add_pd(sv_priv,av);
        }
#pragma omp critical
        sv = _mm_add_pd(sv,sv_priv);
    }
    *s = ((double *)&sv)[0]+((double *)&sv)[1];
#else
    PERFWARNING_LOG("Cannot guarantee streaming stores for triad benchmark!");
    double res;
#pragma omp parallel for reduction(+:res)
    for (i=0; i<N; i++) {
        res += a[i];
    }
    *s = res;
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_KERNEL|GHOST_FUNCTYPE_BENCH);
}

static void ghost_triad_kernel(double * __restrict__ a, const double * __restrict__ b, const double * __restrict__ c, const double s)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_KERNEL|GHOST_FUNCTYPE_BENCH);
    ghost_lidx i;

#ifdef GHOST_BUILD_MIC
    __m512d bv;
    __m512d cv;
    __m512d sv = _mm512_set1_pd(s);
#pragma omp parallel for private(bv,cv)
    for (i=0; i<N; i+=8) {
        bv = _mm512_load_pd(&b[i]);
        cv = _mm512_load_pd(&c[i]);
        _mm512_storenrngo_pd(&a[i],_mm512_add_pd(bv,_mm512_mul_pd(cv,sv)));
    }
#elif defined(GHOST_BUILD_AVX)
    __m256d bv;
    __m256d cv;
    __m256d sv = _mm256_set1_pd(s);
#pragma omp parallel for private(bv,cv)
    for (i=0; i<N; i+=4) {
        bv = _mm256_load_pd(&b[i]);
        cv = _mm256_load_pd(&c[i]);
        _mm256_stream_pd(&a[i],_mm256_add_pd(bv,_mm256_mul_pd(cv,sv)));
    }
#elif defined(GHOST_BUILD_SSE)
    __m128d bv;
    __m128d cv;
    __m128d sv = _mm_set1_pd(s);
#pragma omp parallel for private(bv,cv)
    for (i=0; i<N; i+=2) {
        bv = _mm_load_pd(&b[i]);
        cv = _mm_load_pd(&c[i]);
        _mm_stream_pd(&a[i],_mm_add_pd(bv,_mm_mul_pd(cv,sv)));
    }
#else
    PERFWARNING_LOG("Cannot guarantee streaming stores for triad benchmark!");
#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = b[i] + s*c[i];
    }
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_KERNEL|GHOST_FUNCTYPE_BENCH);
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

static void ghost_store_kernel(double * __restrict__ a, const double s)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_KERNEL|GHOST_FUNCTYPE_BENCH);
    ghost_lidx i;

#ifdef GHOST_BUILD_MIC
    __m512d sv = _mm512_set1_pd(s);;
#pragma omp parallel for
    for (i=0; i<N; i+=8) {
        _mm512_storenrngo_pd(&a[i],sv);
    }
#elif defined(GHOST_BUILD_AVX)
    __m256d sv = _mm256_set1_pd(s);
#pragma omp parallel for
    for (i=0; i<N; i+=4) {
        _mm256_stream_pd(&a[i],sv);
    }
#elif defined(GHOST_BUILD_SSE)
    __m128d sv = _mm_set1_pd(s);
#pragma omp parallel for
    for (i=0; i<N; i+=2) {
        _mm_stream_pd(&a[i],sv);
    }
#else
    PERFWARNING_LOG("Cannot guarantee streaming stores for copy benchmark!");
#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = s;
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

    int i;
    double start,stop;

    double *a = NULL;
    double *b = NULL;
    double *c = NULL;
    double s = 2.2;
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&a,N*sizeof(double),ghost_machine_alignment()),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&b,N*sizeof(double),ghost_machine_alignment()),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&c,N*sizeof(double),ghost_machine_alignment()),err,ret);

#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = 0;
        b[i] = i;
        c[i] = i+1;
    }
  
    switch (test) {
        case GHOST_BENCH_STREAM_COPY:
            ghost_copy_kernel(a,b); // warm up
            ghost_timing_wc(&start);
            for (i=0; i<NITER; i++) {
                ghost_copy_kernel(a,b);
            }
            ghost_timing_wc(&stop);
            *bw = 2*N/1.e9*NITER*sizeof(double)/(stop-start);
            break;
        case GHOST_BENCH_STREAM_TRIAD:
            ghost_triad_kernel(a,b,c,s); // warm up
            ghost_timing_wc(&start);
            for (i=0; i<NITER; i++) {
                ghost_triad_kernel(a,b,c,s);
            }
            ghost_timing_wc(&stop);
            *bw = 3*N/1.e9*NITER*sizeof(double)/(stop-start);
            break;
        case GHOST_BENCH_STREAM_LOAD:
            ghost_load_kernel(a,&s); // warm up
            ghost_timing_wc(&start);
            for (i=0; i<NITER; i++) {
                ghost_load_kernel(a,&s);
            }
            ghost_timing_wc(&stop);
            *bw = N/1.e9*NITER*sizeof(double)/(stop-start);
            break;
        case GHOST_BENCH_STREAM_STORE:
            ghost_store_kernel(a,s); // warm up
            ghost_timing_wc(&start);
            for (i=0; i<NITER; i++) {
                ghost_store_kernel(a,s);
            }
            ghost_timing_wc(&stop);
            *bw = N/1.e9*NITER*sizeof(double)/(stop-start);
            break;
    }

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
