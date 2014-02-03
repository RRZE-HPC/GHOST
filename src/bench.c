#include "ghost/ghost_util.h"
#include "ghost/ghost_bench.h"


#define STREAM_BYTES 1e9
#define N (int)(STREAM_BYTES/sizeof(double))

static void dummy(double *a) {
    if (a[N>>1] < 0) {
        WARNING_LOG("dummy");
    }
}

int ghost_stream(int test, double *bw)
{

    ghost_error_t ret = GHOST_SUCCESS;
    UNUSED(test);

    int i;
    double start;

    double *a = NULL;
    double *b = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&a,N*sizeof(double)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&b,N*sizeof(double)),err,ret);

#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = 0;
        b[i] = i;
    }
    
    start = ghost_wctime();
#pragma omp parallel for
    for (i=0; i<N; i++) {
        a[i] = b[i];
    }
    *bw = 2*STREAM_BYTES/(ghost_wctime()-start);

    dummy(a);

    goto out;

err:
out:
    free(a); a = NULL;
    free(b); b = NULL;

    return ret;
}


int ghost_pingpong(double *bw)
{

    UNUSED(bw);

    return GHOST_SUCCESS;
}

int ghost_peakperformance(double *gf)
{

    UNUSED(gf);

    return GHOST_SUCCESS;
}
