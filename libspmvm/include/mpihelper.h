#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include "spmvm.h"

#include <mpi.h>

void       setupSingleNodeComm();
void SpMVM_createDistributedSetupSerial(SETUP_TYPE *, CR_TYPE* const, int);
void SpMVM_createDistributedSetup(SETUP_TYPE *, CR_TYPE* const, char *, int);
LCRP_TYPE *SpMVM_createCommunicator(CR_TYPE *cr, int options);
MPI_Comm getSingleNodeComm();

#endif
