#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include "ghost.h"
#include "matricks.h"

#include <mpi.h>

void       setupSingleNodeComm();
void ghost_createDistributedSetupSerial(ghost_setup_t *, CR_TYPE* const, int, ghost_mtraits_t *);
void ghost_createDistributedSetup(ghost_setup_t *, CR_TYPE* const, char *, int, ghost_mtraits_t *);
void ghost_createDistribution(CR_TYPE *cr, int options, ghost_comm_t *lcrp);
void ghost_createCommunication(CR_TYPE *cr, int options, ghost_setup_t *setup);
MPI_Comm getSingleNodeComm();

#endif
