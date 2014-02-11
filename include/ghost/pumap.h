#include <hwloc.h>

#include "error.h"

/**
 * @brief The PU (processing units) map containing all available processing units inside a given CPU set.
 */
typedef struct ghost_pumap_t {
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
    hwloc_bitmap_t cpuset;
    /**
     * @brief A bitmap indicating busy cores.
     *
     * The ordering is the same as in the cpuset.
     */
    hwloc_bitmap_t busy;
    /**
     * @brief The number of NUMA domains covered by the CPU map.
     */
    int nDomains;
} ghost_pumap_t;

ghost_error_t ghost_pumap_create(hwloc_cpuset_t cpuset);
void ghost_pumap_destroy();
ghost_error_t ghost_pumap_get(ghost_pumap_t **map);
ghost_error_t ghost_pumap_setIdle(hwloc_bitmap_t cpuset);
ghost_error_t ghost_pumap_setIdleIdx(int idx);
ghost_error_t ghost_pumap_setBusy(hwloc_bitmap_t cpuset);
ghost_error_t ghost_pumap_getNumberOfIdlePUs(int *nPUs, int numaNode);
