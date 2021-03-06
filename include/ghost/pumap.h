/**
 * @ingroup task @{
 * @file pumap.h
 * @brief Types and functions for the PU (processing unit) map functionality. 
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_PUMAP_H
#define GHOST_PUMAP_H
#include <hwloc.h>

#include "error.h"

/**
 * @brief The PU (processing units) map containing all available processing units inside a given CPU set.
 */
typedef struct ghost_pumap {
    /**
     * @brief Ordered list of processing units (PU).
     *
     * There is a list of PUs for each NUMA domain.
     * Each list contains all avilable PUs in the CPU map.
     * The list is ordered to contain domain-local PUs first and remote PUs afterwards.
     */
    hwloc_obj_t **PUs;
    /**
     * @brief The cpuset for this CPU map. 
     */
    hwloc_cpuset_t cpuset;
    /**
     * @brief A bitmap indicating busy cores.
     *
     * The ordering is the same as in the cpuset.
     */
    hwloc_cpuset_t busy;
    /**
     * @brief The number of NUMA domains covered by the CPU map.
     */
    int nDomains;
} ghost_pumap;

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Create a PU map.
     *
     * @param cpuset The CPU set to be covered by the PU map.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_pumap_create(hwloc_cpuset_t cpuset);
    /**
     * @brief Destroy the PU map.
     */
    void ghost_pumap_destroy();
    /**
     * @brief Get the PU map.
     *
     * @param map Where to store the map.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_pumap_get(ghost_pumap **map);
    /**
     * @brief Set the given CPU set in the PU map to idle.  
     *
     * @param cpuset The CPU set to be set idle.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * If the CPU set is not included in the PU map's CPU set an error is returned.
     */
    ghost_error ghost_pumap_setidle(hwloc_bitmap_t cpuset);
    /**
     * @brief Set the given index in the PU map to idle.  
     *
     * @param idx The index.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * If the index is not included in the PU map's CPU set an error is returned.
     */
    ghost_error ghost_pumap_setidle_idx(int idx);
    /**
     * @brief Set the given CPU set in the PU map to busy.  
     *
     * @param cpuset The CPU set to be set busy.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     *
     * If the CPU set is not included in the PU map's CPU set an error is returned.
     */
    ghost_error ghost_pumap_setbusy(hwloc_bitmap_t cpuset);
    /**
     * @brief Get the number of idle processing units in total or in a given NUMA node.
     *
     * @param nPUs Where to store the number.
     * @param numaNode The NUMA node to consider or GHOST_NUMANODE_ANY.
     *
     * @return ::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_pumap_nidle(int *nPUs, int numaNode);

    /**
     * @brief Get the number of processing units in total or in a given NUMA node.
     *
     * @param nPUs Where to store the number.
     * @param numaNode The NUMA node to consider or GHOST_NUMANODE_ANY.
     *
     * @return ::::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_pumap_npu(int *nPUs, int numaNode);

    /**
     * @ingroup stringification
     *
     * @brief Turn the PU mpa into a string. 
     *
     * @param[out] str Where to store the string.
     *
     * @return ::::GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error ghost_pumap_string(char **str);

    extern pthread_mutex_t ghost_pumap_mutex;

#ifdef __cplusplus
}
#endif

/** @} */
#endif
