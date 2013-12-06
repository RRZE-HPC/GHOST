#ifndef __GHOST_HAVE_MPI_UTIL_H__
#define __GHOST_HAVE_MPI_UTIL_H__

#include <ghost_config.h>
#include <ghost_types.h>

#if GHOST_HAVE_MPI
#include <mpi.h>
#endif

extern MPI_Comm ghost_node_comm;
extern MPI_Comm ghost_node_rank;
#ifdef __cplusplus
extern "C" {
#endif
void       setupSingleNodeComm();
//void ghost_createDistributedContextSerial(ghost_context_t *, CR_TYPE* const, int, ghost_mtraits_t *);
void ghost_createDistributedContext(ghost_context_t *, char *, int, ghost_mtraits_t *);
//void ghost_createDistribution(CR_TYPE *cr, int options, ghost_comm_t *lcrp);
//void ghost_createCommunication(CR_TYPE *, CR_TYPE **, CR_TYPE **, int options, ghost_context_t *context);
MPI_Comm getSingleNodeComm();
int ghost_setupNodeMPI(MPI_Comm comm);
int ghost_setupLocalMPIcomm(MPI_Comm mpicomm); 
MPI_Datatype ghost_mpi_dataType(int datatype);
MPI_Op ghost_mpi_op_sum(int datatype);
void ghost_scatterv(void *sendbuf, int *sendcnts, ghost_midx_t *displs, MPI_Datatype sendtype, void *recvbuv, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm);


void hybrid_kernel_I(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
void hybrid_kernel_II(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);
void hybrid_kernel_III(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions);

#ifdef __cplusplus
} //extern "C"
#endif
#endif
