/** 
 * @file machine.h
 * @brief Functions to access machine information.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_MACHINE_H
#define GHOST_MACHINE_H

#include <stddef.h>
#include <stdbool.h>
#include <hwloc.h>
#include "ghost/error.h"

/**
 * @brief This is the alignment size for memory allocated using `ghost_malloc_align()`.
 */
#define GHOST_DATA_ALIGNMENT 1024
#define GHOST_NUMANODE_ANY -1


#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Initialize and load the topology object (of type hwloc_topology_t). 
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * If the topology has been created before, this function returns immediately.
     */
    ghost_error_t ghost_topology_create();
    /**
     * @brief Destroy and free the topology object.
     */
    void ghost_topology_destroy();
    /**
     * @brief Get to topology object 
     *
     * @param topo Where to store the topology.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_topology_get(hwloc_topology_t *topo);
    /**
     * @ingroup machine
     * 
     * @brief Get the size of the first level cache. 
     *
     * @param size Where to store the size.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This information may be useful in situations where the locality of data (cache or memory) influences things like the OpenMP scheduling.
     */
    ghost_error_t ghost_machine_innercache_size(uint64_t *size);
    /**
     * @ingroup machine
     * 
     * @brief Get the size of the last level cache. 
     *
     * @param size Where to store the size.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This information may be useful in situations where the locality of data (cache or memory) influences things like the OpenMP scheduling.
     */
    ghost_error_t ghost_machine_outercache_size(uint64_t *size);
    /**
     * @brief Get the cache line siye. 
     *
     * @param size Where to store the size.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * This information may be useful in situations where false sharing has to be avoided.
     */
    ghost_error_t ghost_machine_cacheline_size(unsigned *size);
    /**
     * @brief Get the number of (physical) cores in the machine.
     *
     * @param nCores Where to store the number of cores.
     * @param numaNode Only look for PUs inside this NUMA node.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_machine_ncore(int *nCores, int numaNode);
    /**
     * @brief Get the number of SMT threads per core in the machine.
     *
     * @param nLevels Where to store the number.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_machine_nsmt(int *nLevels);
    /**
     * @brief Get the number of available hardware threads (= physical cores times SMT level) 
     * or processing units in the machine.
     *
     * @param nPUs Where to store the number.
     * @param numaNode Only look for PUs inside this NUMA node.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_machine_npu(int *nPUs, int numaNode);
    /**
     * @brief Get the number of NUMA nodes in the machine.
     *
     * @param nNodes Where to store the number.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_machine_nnuma(int *nNodes);
    ghost_error_t ghost_machine_numanode(hwloc_obj_t *node, int idx);
    /**
     * @brief Check whether machine is big endian.
     *
     * @return true if machine is big endian, false if machine is little endian.
     */
    bool ghost_machine_bigendian();
    /**
     * @ingroup stringification
     *
     * @brief Get a string of the machine information. 
     *
     * @param[out] str Where to store the string.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_machine_string(char **str);

#ifdef __cplusplus
}
#endif

#endif

