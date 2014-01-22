#ifndef __GHOST_AFFINITY_H__
#define __GHOST_AFFINITY_H__

#include "config.h"
#include "types.h"
#include <hwloc.h>

#if GHOST_HAVE_MPI
#include <mpi.h>
#endif

extern hwloc_topology_t topology;
#ifdef __cplusplus
extern "C" {
#endif

int ghost_getRank(ghost_mpi_comm_t);
//int ghost_getLocalRank(MPI_Comm);
//int ghost_getNumberOfLocalRanks(MPI_Comm);
int ghost_getNumberOfHwThreads();
int ghost_getNumberOfNumaNodes();
int ghost_getNumberOfThreads();
int ghost_getNumberOfNodes();
int ghost_getNumberOfRanks(ghost_mpi_comm_t);
int ghost_getNumberOfPhysicalCores();
int ghost_getCore();
int ghost_getSMTlevel();
void ghost_setCore(int core);
void ghost_unsetCore();
void ghost_pinThreads(int options, char *procList);
#ifdef __cplusplus
}
#endif

#endif
