/**
 * @file bench.h
 * @brief Functions for micro-benchmarking.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_BENCH_H
#define GHOST_BENCH_H

typedef enum
{
    GHOST_BENCH_STREAM_COPY,
    GHOST_BENCH_STREAM_TRIAD,
    GHOST_BENCH_LOAD,
    GHOST_BENCH_STORE,
    GHOST_BENCH_UPDATE
} ghost_bench_bw_test;

#ifdef __cplusplus
extern "C" {
#endif

ghost_error ghost_bench_bw(ghost_bench_bw_test, double *mean_bw, double *max_bw);
ghost_error ghost_bench_pingpong(double *bw);
ghost_error ghost_bench_peakperformance(double *gf);

#ifdef __cplusplus
} //extern "C"
#endif

#endif
