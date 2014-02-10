#include <hwloc.h>

#include "error.h"

/**
 * @brief The thread pool consisting of all threads that will ever do some
 * tasking-related work.
 */
typedef struct ghost_cpumap_t {
    /**
     * @brief The PU (Processing Unit) of each GHOST thread.
     */
    hwloc_obj_t **PUs;
    /**
     * @brief The cpuset this thread pool is covering. 
     */
    hwloc_bitmap_t cpuset;
    /**
     * @brief A bitmap with one bit per PU where 1 means that a PU is busy and 0 means that it is
     * idle.
     */
    hwloc_bitmap_t busy;
    /**
     * @brief The number of LDs covered by the pool's threads.
     */
    int nLDs;
} ghost_cpumap_t;

ghost_error_t ghost_createCPUmap();
void ghost_destroyCPUmap();
ghost_error_t ghost_getCPUmap(ghost_cpumap_t **map);
ghost_error_t ghost_setCPUidle(int cpu);
ghost_error_t ghost_setCPUbusy(int cpu);
