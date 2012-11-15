#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include "spmvm.h"
#include "matricks.h"

#include <mpi.h>

void       setupSingleNodeComm();
void SpMVM_createDistributedSetupSerial(ghost_setup_t *, CR_TYPE* const, int, mat_trait_t *);
void SpMVM_createDistributedSetup(ghost_setup_t *, CR_TYPE* const, char *, int, mat_trait_t *);
void SpMVM_createDistribution(CR_TYPE *cr, int options, ghost_comm_t *lcrp);
void SpMVM_createCommunication(CR_TYPE *cr, int options, ghost_setup_t *setup);
MPI_Comm getSingleNodeComm();

#endif
