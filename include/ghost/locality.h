/**
 * @file locality.h
 * @brief Types and functions for gathering locality information.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_LOCALITY_H
#define GHOST_LOCALITY_H

#include "error.h"
#include <hwloc.h>

#ifdef GHOST_HAVE_MPI
#include <mpi.h>
#endif

typedef struct {
    int nCores;
    int nSmt;
} ghost_hwconfig_t;

typedef enum {
    GHOST_HYBRIDMODE_INVALID, 
    GHOST_HYBRIDMODE_ONEPERNODE, 
    GHOST_HYBRIDMODE_ONEPERNUMA, 
    GHOST_HYBRIDMODE_ONEPERCORE,
    GHOST_HYBRIDMODE_CUSTOM
} ghost_hybridmode_t;

#define GHOST_HWCONFIG_INVALID -1
#define GHOST_HWCONFIG_INITIALIZER (ghost_hwconfig_t) {.nCores = GHOST_HWCONFIG_INVALID, .nSmt = GHOST_HWCONFIG_INVALID}


//extern hwloc_topology_t topology;


#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_hybridmode_set(ghost_hybridmode_t hm);
ghost_error_t ghost_hybridmode_get(ghost_hybridmode_t *hm);
ghost_error_t ghost_hwconfig_set(ghost_hwconfig_t);
ghost_error_t ghost_hwconfig_get(ghost_hwconfig_t * hwconfig);
ghost_error_t ghost_getRank(ghost_mpi_comm_t comm, int *rank);
ghost_error_t ghost_getNumberOfNodes(ghost_mpi_comm_t comm, int *nNodes);
ghost_error_t ghost_getNumberOfRanks(ghost_mpi_comm_t comm, int *nRanks);
ghost_error_t ghost_getCore(int *core);
ghost_error_t ghost_setCore(int core);
ghost_error_t ghost_unsetCore();
ghost_error_t ghost_getNodeComm(ghost_mpi_comm_t *comm);
ghost_error_t ghost_setupNodeMPI(ghost_mpi_comm_t comm);
#ifdef __cplusplus
}
#endif

#endif
