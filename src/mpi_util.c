#define _GNU_SOURCE
#include "ghost_mpi_util.h"
#include "ghost.h"
#include "ghost_util.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/syscall.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <fcntl.h>
#include <errno.h>

#include <math.h>
#include <complex.h>
#include <dlfcn.h>


static MPI_Comm single_node_comm;
MPI_Comm *ghost_mpi_comms;


MPI_Datatype ghost_mpi_dataType(int datatype)
{
	if (datatype & GHOST_BINCRS_DT_FLOAT) {
		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			return GHOST_MPI_DT_C;
		else
			return MPI_FLOAT;
	} else {
		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			return GHOST_MPI_DT_Z;
		else
			return MPI_DOUBLE;
	}
}

MPI_Op ghost_mpi_op_sum(int datatype)
{
	if (datatype & GHOST_BINCRS_DT_FLOAT) {
		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			return GHOST_MPI_OP_SUM_C;
		else
			return MPI_SUM;
	} else {
		if (datatype & GHOST_BINCRS_DT_COMPLEX)
			return GHOST_MPI_OP_SUM_Z;
		else
			return MPI_SUM;
	}

}

void ghost_scatterv(void *sendbuf, int *sendcnts, ghost_midx_t *displs, MPI_Datatype sendtype, void *recvbuv, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
#ifdef LONGIDX

	UNUSED(sendbuf);
	UNUSED(sendcnts);
	UNUSED(displs);
	UNUSED(sendtype);
	UNUSED(recvbuv);
	UNUSED(recvcnt);
	UNUSED(recvtype);
	UNUSED(root);
	UNUSED(comm);
#else
MPI_safecall(MPI_Scatterv(sendbuf,sendcnts,displs,sendtype,recvbuv,recvcnt,recvtype,root,comm));
#endif

}



