/** 
 * @file machine.h
 * @brief Functions to access machine information.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_MACHINE_H
#define GHOST_MACHINE_H

#include <stddef.h>
#include <hwloc.h>
#include "ghost/error.h"

/**
 * @brief This is the alignment size for memory allocated using `ghost_malloc_align()`.
 */
#define GHOST_DATA_ALIGNMENT 1024

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Initialize and load the topology object (of type hwloc_topology_t). 
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * If the topology has been created before, this function returns immediately.
     */
    ghost_error_t ghost_createTopology();
    /**
     * @brief Destroy and free the topology object.
     */
    void ghost_destroyTopology();
    /**
     * @brief Get to topology object 
     *
     * @param topo Where to store the topology.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_getTopology(hwloc_topology_t *topo);
    /**
     * @ingroup machine
     * 
     * @brief Get the size of the last level cache. 
     *
     * @param size Where to store the size.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * This information may be useful in situations where the locality of data (cache or memory) influences things like the OpenMP scheduling.
     */
    ghost_error_t ghost_getSizeOfLLC(uint64_t *size);
    /**
     * @brief Get the cache line siye. 
     *
     * @param size Where to store the size.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     *
     * This information may be useful in situations where false sharing has to be avoided.
     */
    ghost_error_t ghost_getSizeOfCacheLine(unsigned *size);
    /**
     * @brief Get the number of physical cores in the machine.
     *
     * @param nCores Where to store the number of cores.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_getNumberOfPhysicalCores(int *nCores);
    /**
     * @brief Get the number of SMT threads per core in the machine.
     *
     * @param nLevels Where to store the number.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_getSMTlevel(int *nLevels);
    /**
     * @brief Get the number of available hardware threads (= physical cores times SMT level)  in the machine.
     *
     * @param nThreads Where to store the number.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_getNumberOfHwThreads(int *nThreads);
    /**
     * @brief Get the number of NUMA nodes in the machine.
     *
     * @param nNodes Where to store the number.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_getNumberOfNumaNodes(int *nNodes);
    /**
     * @brief Check whether machine is big endian.
     *
     * @return 1 if machine is big endian, 0 if machine is little endian.
     */
    char ghost_machineIsBigEndian();

#ifdef __cplusplus
}
#endif

#endif

