#ifndef _MPI_HELPER_H_
#define _MPI_HELPER_H_

#include <mpi.h>
#include "oclfun.h"

void setupSingleNodeComm( char*, MPI_Comm*, int* );
void selectNodalGPUs( MPI_Comm*, const char* );
void CL_selectNodalGPUs( MPI_Comm*, const char* );

#endif
