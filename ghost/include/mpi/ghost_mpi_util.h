#ifndef __GHOST_MPI_UTIL_H__
#define __GHOST_MPI_UTIL_H__

#include "ghost.h"
#include "ghost_mat.h"

void       setupSingleNodeComm();
//void ghost_createDistributedContextSerial(ghost_context_t *, CR_TYPE* const, int, ghost_mtraits_t *);
void ghost_createDistributedContext(ghost_context_t *, char *, int, ghost_mtraits_t *);
//void ghost_createDistribution(CR_TYPE *cr, int options, ghost_comm_t *lcrp);
//void ghost_createCommunication(CR_TYPE *, CR_TYPE **, CR_TYPE **, int options, ghost_context_t *context);
MPI_Comm getSingleNodeComm();
MPI_Datatype ghost_mpi_dataType(int datatype);
MPI_Op ghost_mpi_op_sum(int datatype);
void ghost_scatterv(void *sendbuf, int *sendcnts, ghost_midx_t *displs, MPI_Datatype sendtype, void *recvbuv, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm);

void hybrid_kernel_I(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
void hybrid_kernel_II(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
void hybrid_kernel_III(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);

#endif
