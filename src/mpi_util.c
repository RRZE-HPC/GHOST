#define _GNU_SOURCE

#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_mpi_util.h>
#include <ghost_util.h>
#include <ghost_constants.h>
#include <ghost_affinity.h>

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

/*int ghost_setupLocalMPIcomm(MPI_Comm mpicomm) 
{
    int i, coreId, me, n_nodes, me_node;
    char **all_hostnames;
    char *all_hn_mem;
    char hostname[MAXHOSTNAMELEN];
    gethostname(hostname,MAXHOSTNAMELEN);

    size_t size_ahnm, size_ahn, size_nint;
    int *mymate, *acc_mates;

    MPI_safecall(MPI_Comm_size ( mpicomm, &n_nodes ));
    MPI_safecall(MPI_Comm_rank ( mpicomm, &me ));

//    coreId = getProcessorId();

    size_ahnm = (size_t)( MAXHOSTNAMELEN*n_nodes * sizeof(char) );
    size_ahn  = (size_t)( n_nodes    * sizeof(char*) );
    size_nint = (size_t)( n_nodes    * sizeof(int) );

    mymate        = (int*)      malloc( size_nint);
    acc_mates     = (int*)      malloc( size_nint );
    all_hn_mem    = (char*)     malloc( size_ahnm );
    all_hostnames = (char**)    malloc( size_ahn );

    for (i=0; i<n_nodes; i++){
        all_hostnames[i] = &all_hn_mem[i*MAXHOSTNAMELEN];
        mymate[i] = 0;
    }

    MPI_safecall(MPI_Allgather ( hostname, MAXHOSTNAMELEN, MPI_CHAR, 
                &all_hostnames[0][0], MAXHOSTNAMELEN, MPI_CHAR, mpicomm ));

    coreId=ghost_getCore();
    if (coreId==0){
        for (i=0; i<n_nodes; i++){
            if ( strcmp (hostname, all_hostnames[i]) == 0) mymate[i]=me;
        }
    }
    for (i=0; i<n_nodes; i++) {
      INFO_LOG("mymate[%d] = %d",i,mymate[i]);
    }  

    MPI_safecall(MPI_Allreduce( mymate, acc_mates, n_nodes, MPI_INT, MPI_SUM, mpicomm)); 
    MPI_safecall(MPI_Comm_split ( mpicomm, acc_mates[me], me, &ghost_node_mpicomm ));
    MPI_safecall(MPI_Comm_rank ( ghost_node_mpicomm, &me_node));

    INFO_LOG("local ranks: %d",ghost_getNumberOfRanks(ghost_node_mpicomm));
    DEBUG_LOG(1,"Rank in single node comm: %d", me_node);

    free( mymate );
    free( acc_mates );
    free( all_hn_mem );
    free( all_hostnames );

    return GHOST_SUCCESS;
}*/

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

static int ghost_hostname(char ** hostnamePtr, size_t * hostnameLength)
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
			WARNING_LOG("Allocating %lu bytes of memory for hostname failed: %s",
				sizeof(char) * nHostname, strerror(errno));
			return GHOST_FAILURE;
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
				return GHOST_FAILURE;
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

int ghost_setupNodeMPI(MPI_Comm comm)
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

	DEBUG_LOG(2," comm_split:  color:  %u  rank:  %d   hostnameLength: %lu", checkSum, mpiRank, hostnameLength);

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
