#ifndef GHOST_AFFINITY_H
#define GHOST_AFFINITY_H

#include "config.h"
#include "types.h"
#include <hwloc.h>

#if GHOST_HAVE_MPI
#include <mpi.h>
#endif

typedef struct {
    int maxCores;
    int smtLevel;
} ghost_hw_config_t;

typedef enum {
    GHOST_HYBRIDMODE_INVALID, 
    GHOST_HYBRIDMODE_ONEPERNODE, 
    GHOST_HYBRIDMODE_ONEPERNUMA, 
    GHOST_HYBRIDMODE_ONEPERCORE,
    GHOST_HYBRIDMODE_CUSTOM
} ghost_hybridmode_t;

#define GHOST_HW_CONFIG_INVALID -1

extern const ghost_hw_config_t GHOST_HW_CONFIG_INITIALIZER;


//extern hwloc_topology_t topology;


#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_setHybridMode(ghost_hybridmode_t hm);
ghost_error_t ghost_getHybridMode(ghost_hybridmode_t *hm);
ghost_error_t ghost_setHwConfig(ghost_hw_config_t);
ghost_error_t ghost_getHwConfig(ghost_hw_config_t * hwconfig);
ghost_error_t ghost_getRank(ghost_mpi_comm_t, int *rank);
ghost_error_t ghost_getNumberOfNodes(ghost_mpi_comm_t comm, int *nNodes);
ghost_error_t ghost_getNumberOfRanks(ghost_mpi_comm_t comm, int *nRanks);
ghost_error_t ghost_getCore(int *core);
ghost_error_t ghost_setCore(int core);
ghost_error_t ghost_unsetCore();
void ghost_pinThreads(int options, char *procList);
#ifdef __cplusplus
}
#endif

#endif
