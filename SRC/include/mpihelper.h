#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include <mpi.h>

void       setupSingleNodeComm( char*, MPI_Comm*, int* );
LCRP_TYPE* setup_communication(CR_TYPE* const, int);

#endif
