#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include "spmvm.h"

#include <mpi.h>

void       setupSingleNodeComm();
LCRP_TYPE* setup_communication(CR_TYPE* const, int);
LCRP_TYPE* setup_communication_parallel(CR_TYPE* cr, char *matrixPath, int options);
MPI_Comm getSingleNodeComm();

#endif
