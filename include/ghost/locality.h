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
    int ncore;
    int nsmt;
    int cudevice;
} ghost_hwconfig;

#define GHOST_HWCONFIG_INVALID -1
#define GHOST_HWCONFIG_INITIALIZER {.ncore = GHOST_HWCONFIG_INVALID, .nsmt = GHOST_HWCONFIG_INVALID, .cudevice = GHOST_HWCONFIG_INVALID}

#ifdef __cplusplus
extern "C" {
#endif

    ghost_error ghost_hwconfig_set(ghost_hwconfig);
    ghost_error ghost_hwconfig_get(ghost_hwconfig * hwconfig);
    ghost_error ghost_rank(int *rank, ghost_mpi_comm comm);
    ghost_error ghost_nnode(int *nnode, ghost_mpi_comm comm);
    ghost_error ghost_nrank(int *nrank, ghost_mpi_comm comm);
    ghost_error ghost_cpu(int *core);
    ghost_error ghost_thread_pin(int core);
    ghost_error ghost_thread_unpin();
    ghost_error ghost_nodecomm_get(ghost_mpi_comm *comm);
    ghost_error ghost_nodecomm_setup(ghost_mpi_comm comm);

#ifdef __cplusplus
}
#endif

#endif
