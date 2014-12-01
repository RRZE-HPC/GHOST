/**
 * @file omp.h
 * @brief Function wrappers for OpenMP functions.
 * If OpenMP ist disabled, the function are still defined but stubs.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_OMP_H
#define GHOST_OMP_H

#ifdef __cplusplus
extern "C" {
#endif
    
    /**
     * @brief Set the number of threads.
     *
     * @param nthreads The number of threads.
     */
    void ghost_omp_nthread_set(int nthreads);
    /**
     * @brief Get the active thread number.
     *
     * @return The thread number.
     */
    int ghost_omp_threadnum();
    /**
     * @brief Get the number of OpenMP threads.
     *
     * @return The number of threads.
     */
    int ghost_omp_nthread();
    /**
     * @brief Check whether nested parallelism is enabled.
     *
     * @return 1 if nested parallelism is enabled, 0 if not.
     */
    int ghost_omp_get_nested();
    /**
     * @brief En- or disable nested parallelism.
     *
     * @param nested 1 if nested parallelism should be en-, 0 if it should be
     * disabled.
     */
    void ghost_omp_set_nested(int nested);
    /**
     * @brief Check whether one is in an active parallel region.
     *
     * @return 1 if in parallel region, 0 otherwise.
     */
    int ghost_omp_in_parallel();

#ifdef __cplusplus
}
#endif

#endif
