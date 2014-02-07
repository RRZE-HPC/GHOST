#ifndef GHOST_MACHINE_H
#define GHOST_MACHINE_H

#include <stddef.h>
#include <hwloc.h>
#include "ghost/error.h"

#ifdef __cplusplus
extern "C" {
#endif

ghost_error_t ghost_createTopology();
ghost_error_t ghost_destroyTopology();
ghost_error_t ghost_getTopology(hwloc_topology_t *topo);
ghost_error_t ghost_getSizeOfLLC(uint64_t *size);
ghost_error_t ghost_getSizeOfCacheLine(unsigned *size);
ghost_error_t ghost_getNumberOfPhysicalCores(int *nCores);
ghost_error_t ghost_getSMTlevel(int *nLevels);
ghost_error_t ghost_getNumberOfHwThreads(int *nThreads);
ghost_error_t ghost_getNumberOfNumaNodes(int *nNodes);
int ghost_machineIsBigEndian();

#ifdef __cplusplus
}
#endif

#endif
