#define _GNU_SOURCE

#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/mpi_util.h"
#include "ghost/util.h"
#include "ghost/constants.h"
#include "ghost/affinity.h"

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

#define LOCAL_HOSTNAME_MAX 	256


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

static uint32_t adler32(const void * buf, size_t buflength)
{
	// Trace();

	const uint8_t * buffer = (const uint8_t *)buf;

	uint32_t s1 = 1;
	uint32_t s2 = 0;

	for (size_t n = 0; n < buflength; n++) {
		s1 = (s1 + buffer[n]) % 65521;
		s2 = (s2 + s1) % 65521;
	}

	return (s2 << 16) | s1;
}

static ghost_error_t ghost_hostname(char ** hostnamePtr, size_t * hostnameLength)
{
	// Trace();

	char * hostname = NULL;
	size_t nHostname = 0;

	int allocateMore = 0;

	*hostnamePtr = NULL;
	*hostnameLength = 0;

	do {
		nHostname += MAX(HOST_NAME_MAX, LOCAL_HOSTNAME_MAX);

		hostname = (char *)malloc(sizeof(char) * nHostname);

		if (hostname == NULL) {
			WARNING_LOG("Allocating %zu bytes of memory for hostname failed: %s",
				sizeof(char) * nHostname, strerror(errno));
			return GHOST_ERR_INTERNAL;
		}

		int error;

		error = gethostname(hostname, nHostname);

		if (error == -1) {
			if (errno == ENAMETOOLONG) {
				allocateMore = 1;
				free(hostname); hostname = NULL;
			}
			else {
				free(hostname);
				hostname = NULL;

				WARNING_LOG("gethostname failed with error %d: %s", errno, strerror(errno));
				return GHOST_ERR_INTERNAL;
			}

		}
		else {
			allocateMore = 0;
		}

	} while (allocateMore);

	// Make sure hostname is \x00 terminated.
	hostname[nHostname - 1] = 0x00;

	*hostnameLength = strnlen(hostname, nHostname) + 1;
	*hostnamePtr = hostname;

	return 0;
}

ghost_error_t ghost_setupNodeMPI(MPI_Comm comm)
{
    int mpiRank = ghost_getRank(comm);
	int error;
	char * hostname = NULL;
	size_t hostnameLength = 0;

	error = ghost_hostname(&hostname, &hostnameLength);
	if (error != 0) {
		return -1;
	}

	uint32_t checkSum = adler32(hostname, hostnameLength);

	int commRank = -1;
	MPI_safecall(MPI_Comm_rank(comm, &commRank));

	MPI_Comm nodeComm = MPI_COMM_NULL;

	DEBUG_LOG(2," comm_split:  color:  %u  rank:  %d   hostnameLength: %zu", checkSum, mpiRank, hostnameLength);

	MPI_safecall(MPI_Comm_split(comm, checkSum, mpiRank, &nodeComm));

	int nodeRank;
	MPI_safecall(MPI_Comm_rank(nodeComm, &nodeRank));

	int nodeSize;
	MPI_safecall(MPI_Comm_size(nodeComm, &nodeSize));

	// Determine if collisions of the hashed hostname occured.

	int nSend = MAX(HOST_NAME_MAX, LOCAL_HOSTNAME_MAX);
	char * send = (char *)ghost_malloc(sizeof(char) * nSend);

	strncpy(send, hostname, nSend);

	// Ensure terminating \x00 at the end, this may not be
	// garanteed if if len(send) = nSend.
	send[nSend - 1] = 0x00;

	char * recv = (char *)malloc(sizeof(char) * nSend * nodeSize);
	MPI_safecall(MPI_Allgather(send, nSend, MPI_CHAR, recv, nSend, MPI_CHAR, nodeComm));

	char * neighbor = recv;
	int localNodeRank = 0;

	#define STREQ(a, b)  (strcmp((a), (b)) == 0)

	// recv contains now an array of hostnames from all MPI ranks of
	// this communicator. They are sorted ascending by the MPI rank.
	// Also if collisions occur these are handled here.

	for (int i = 0; i < nodeSize; ++i) {

		if (STREQ(send, neighbor)) {
			if (i < nodeRank) {
				// Compared neighbors still have lower rank than we have.
				++localNodeRank;
			}
			else {
				break;
			}
		}
		else {
			// Collision of the hash.
		}

		neighbor += nSend;
	}

	#undef streq


	if (nodeRank != localNodeRank) {
		WARNING_LOG("[%5d] collisions occured during node rank determinaton: "
			  "node rank:  %5d, local node rank:  %5d, host: %s",
			  commRank, nodeRank, localNodeRank, send);
		nodeRank = localNodeRank;
	}


	// Clean up.

	free(send); send = NULL;
	free(recv); recv = NULL;

	ghost_node_comm = nodeComm;
    ghost_node_rank = nodeRank;

	free(hostname); hostname = NULL;

	return GHOST_SUCCESS;
}
