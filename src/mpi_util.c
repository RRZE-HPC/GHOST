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
#include <sched.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <err.h>
#include <fcntl.h>
#include <errno.h>

#include <math.h>
#include <omp.h>
#include <complex.h>
#include <dlfcn.h>

#define MAX_NUM_THREADS 128
#define GHOST_RANK_INVALID -1

#define GHOST_SHMEM_NAME "/ghost-shmem"

#ifndef PAGE_SIZE
	#define PAGE_SIZE	4096
#endif

static MPI_Comm single_node_comm;
MPI_Comm *ghost_mpi_comms;
static int localRank = GHOST_RANK_INVALID;
static int nLocalRanks = GHOST_RANK_INVALID;

static inline int AtomicFetchAndAdd(int * variable, int value);
static void * shared_mem_allocate();
static void shared_mem_deallocate(void * shmRegion);

static int getProcessorId() 
{

	cpu_set_t  cpu_set;
	int processorId;

	CPU_ZERO(&cpu_set);
	sched_getaffinity((pid_t)0,sizeof(cpu_set_t), &cpu_set);

	for (processorId=0;processorId<MAX_NUM_THREADS;processorId++){
		if (CPU_ISSET(processorId,&cpu_set))
		{  
			break;
		}
	}
	return processorId;
}

MPI_Comm getSingleNodeComm()
{

	return single_node_comm;
}

void setupSingleNodeComm() 
{

	/* return MPI communicator between nodal MPI processes single_node_comm
	 * and process rank me_node on local node */

	int i, coreId, me, n_nodes, me_node;
	char **all_hostnames;
	char *all_hn_mem;
	char hostname[MAXHOSTNAMELEN];
	gethostname(hostname,MAXHOSTNAMELEN);

	size_t size_ahnm, size_ahn, size_nint;
	int *mymate, *acc_mates;

	MPI_safecall(MPI_Comm_size ( MPI_COMM_WORLD, &n_nodes ));
	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));

	coreId = getProcessorId();

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

	/* write local hostname to all_hostnames and share */
	MPI_safecall(MPI_Allgather ( hostname, MAXHOSTNAMELEN, MPI_CHAR, 
				&all_hostnames[0][0], MAXHOSTNAMELEN, MPI_CHAR, MPI_COMM_WORLD ));

	/* one process per node writes its global id to all its mates' fields */ 
	if (coreId==0){
		for (i=0; i<n_nodes; i++){
			if ( strcmp (hostname, all_hostnames[i]) == 0) mymate[i]=me;
		}
	}  

	MPI_safecall(MPI_Allreduce( mymate, acc_mates, n_nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD)); 
	/* all processes should now have the rank of their coreId 0 process in their acc_mate field;
	 * split into comm groups with this rank as communicator ID */
	MPI_safecall(MPI_Comm_split ( MPI_COMM_WORLD, acc_mates[me], me, &single_node_comm ));
	MPI_safecall(MPI_Comm_rank ( single_node_comm, &me_node));

	DEBUG_LOG(1,"Rank in single node comm: %d", me_node);

	free( mymate );
	free( acc_mates );
	free( all_hn_mem );
	free( all_hostnames );
}

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

static inline int AtomicFetchAndAdd(int * variable, int value)
{
	__asm__ volatile (
		"lock;"
		"xaddl %%eax, %2;"
		: "=a" (value)					// output
		: "a" (value), "m" (*variable)	// input
		:"memory" 						// cloppered
	);

	return value;
}


static void NodeRankByShm(MPI_Comm comm)
{
	int * nodeRankPtr = (int *)shared_mem_allocate();

	int nodeRank;
	nodeRank = AtomicFetchAndAdd(nodeRankPtr, 1);

	shared_mem_deallocate((void *)nodeRankPtr);

	localRank = nodeRank;

	MPI_safecall(MPI_Barrier(comm));
	nLocalRanks = nodeRank+1;
}

static int g_shm_fd = -1;

static void * shared_mem_allocate()
{
	int err;
	int shm_fd;

	// Try to create the shared memory object.
	shm_fd = shm_open(GHOST_SHMEM_NAME, O_RDWR | O_CREAT | O_EXCL, 0600);

	if (shm_fd < 0) {
		if (errno == EEXIST) {
			// SMO already exists, just open it.
			shm_fd = shm_open(GHOST_SHMEM_NAME, O_RDWR, 0600);
			if (shm_fd < 0) {
				ABORT("shm_open failed");
			}
		}
		else if (errno < 0) {
			ABORT("shm_open failed");
		}
	}
	else {
		// g_shm_creator = 1;
	}

	g_shm_fd = shm_fd;

	// TODO: is it really safe to call this from every process?
	err = ftruncate(shm_fd, PAGE_SIZE);
	if (err < 0) {
		shm_unlink(GHOST_SHMEM_NAME);
		ABORT("ftruncate failed");
	}

	void * shm_region;
	shm_region = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

	if (shm_region == MAP_FAILED) {
		shm_unlink(GHOST_SHMEM_NAME);
		ABORT("mmap failed");
	}

	return shm_region;
}


static void shared_mem_deallocate(void * shmRegion)
{
	int shouldExit = 0;
	int err;

	if (shmRegion == NULL) {
		ABORT("shared_mem_deallocate was called, but memory was not initialized");
	}

	err = munmap(shmRegion, PAGE_SIZE);
	if (err < 0) {
		perror("region");
		shouldExit = 1;
	}

	err = shm_unlink(GHOST_SHMEM_NAME);
	if (err < 0) {
		// This error occurs if another process has already called shm_unlink.
		if (errno != ENOENT) {
			perror("shm_unlink");
			shouldExit = 1;
		}
	}

	err = close(g_shm_fd);

	if (shouldExit) {
		ABORT("shared_mem_deallocate");
	}

	return;
}


