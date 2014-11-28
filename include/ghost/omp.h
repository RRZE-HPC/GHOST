/**
 * @file omp.h
 * @brief Function wrappers for OpenMP functions.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_OMP_H
#define GHOST_OMP_H

#ifdef __cplusplus
extern "C" {
#endif
    
    void ghost_omp_nthread_set(int nthreads);
    int ghost_omp_threadnum();
    int ghost_omp_nthread();
    int ghost_omp_get_nested();
    void ghost_omp_set_nested(int nested);
    int ghost_omp_in_parallel();

#ifdef __cplusplus
}
#endif

#endif
