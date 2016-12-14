/**
 * @file cu_bench.h
 * @brief Functions for micro-benchmarking on CUDA devices.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_CU_BENCH_H
#define GHOST_CU_BENCH_H

#include "bench.h"

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error ghost_cu_bench_bw(ghost_bench_bw_test, double *mean_bw, double *max_bw);

#ifdef __cplusplus
}
#endif

#endif
