#include "ghost_util.h"
#include "ghost_bench.h"


#define STREAM_BYTES 1e9
#define N (int)(STREAM_BYTES/sizeof(double))

static void dummy(double *a) {
    if (a[N>>1] < 0) {
        WARNING_LOG("dummy");
    }
}

int ghost_stream(int test, double *bw)
{

    UNUSED(test);

    int i;
    double start;

    double *a = (double *)ghost_malloc(N*sizeof(double));
    double *b = (double *)ghost_malloc(N*sizeof(double));

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

    return GHOST_SUCCESS;
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
