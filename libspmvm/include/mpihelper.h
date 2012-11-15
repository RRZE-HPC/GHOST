#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include "spmvm.h"

#include <mpi.h>

void       setupSingleNodeComm();
void SpMVM_createDistributedSetupSerial(SETUP_TYPE *, CR_TYPE* const, int);
void SpMVM_createDistributedSetup(SETUP_TYPE *, CR_TYPE* const, char *, int);
void SpMVM_createDistribution(CR_TYPE *cr, int options, LCRP_TYPE *lcrp);
void SpMVM_createCommunication(CR_TYPE *cr, int options, SETUP_TYPE *setup);
MPI_Comm getSingleNodeComm();

#endif
