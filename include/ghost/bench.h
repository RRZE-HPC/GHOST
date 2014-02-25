#ifndef GHOST_BENCH_H
#define GHOST_BENCH_H

ghost_error_t ghost_bench_stream(int test, double *bw);
ghost_error_t ghost_bench_pingpong(double *bw);
ghost_error_t ghost_bench_peakperformance(double *gf);

#endif
