#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include "ghost.h"
#include "matricks.h"

#include <mpi.h>

void       setupSingleNodeComm();
void ghost_createDistributedContextSerial(ghost_context_t *, CR_TYPE* const, int, ghost_mtraits_t *);
void ghost_createDistributedContext(ghost_context_t *, char *, int, ghost_mtraits_t *);
void ghost_createDistribution(CR_TYPE *cr, int options, ghost_comm_t *lcrp);
void ghost_createCommunication(CR_TYPE *cr, int options, ghost_context_t *context);
MPI_Comm getSingleNodeComm();

#endif
