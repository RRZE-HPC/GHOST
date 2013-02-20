#ifndef __GHOST_MPI_UTIL_H__
#define __GHOST_MPI_UTIL_H__

#include "ghost.h"
#include "ghost_mat.h"

#include <mpi.h>

void       setupSingleNodeComm();
//void ghost_createDistributedContextSerial(ghost_context_t *, CR_TYPE* const, int, ghost_mtraits_t *);
void ghost_createDistributedContext(ghost_context_t *, char *, int, ghost_mtraits_t *);
//void ghost_createDistribution(CR_TYPE *cr, int options, ghost_comm_t *lcrp);
//void ghost_createCommunication(CR_TYPE *, CR_TYPE **, CR_TYPE **, int options, ghost_context_t *context);
MPI_Comm getSingleNodeComm();
int ghost_mpi_dataType(int datatype);

void hybrid_kernel_I(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
void hybrid_kernel_II(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
void hybrid_kernel_III(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);

#endif
