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

#define GHOST_HW_CONFIG_INVALID -1

extern const ghost_hw_config_t GHOST_HW_CONFIG_INITIALIZER;


//extern hwloc_topology_t topology;


#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_setHwConfig(ghost_hw_config_t);
ghost_error_t ghost_getHwConfig(ghost_hw_config_t * hwconfig);
int ghost_getRank(ghost_mpi_comm_t);
//int ghost_getLocalRank(MPI_Comm);
//int ghost_getNumberOfLocalRanks(MPI_Comm);
int ghost_getNumberOfThreads();
int ghost_getNumberOfNodes();
int ghost_getNumberOfRanks(ghost_mpi_comm_t);
int ghost_getCore();
void ghost_setCore(int core);
void ghost_unsetCore();
void ghost_pinThreads(int options, char *procList);
#ifdef __cplusplus
}
#endif

#endif
