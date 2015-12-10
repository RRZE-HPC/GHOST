/**
 * @file bench.h
 * @brief Functions for micro-benchmarking.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_BENCH_H
#define GHOST_BENCH_H

typedef enum
{
    GHOST_BENCH_STREAM_COPY
} ghost_bench_stream_test_t;

ghost_error_t ghost_bench_stream(ghost_bench_stream_test_t, double *bw);
ghost_error_t ghost_bench_pingpong(double *bw);
ghost_error_t ghost_bench_peakperformance(double *gf);

#endif
