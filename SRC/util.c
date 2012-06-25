#include "matricks.h"
#include <sys/param.h>
#include <libgen.h>
#include "oclfun.h"


void SpMVM_init (int argc, char **argv, char *matrixPath, MATRIX_FORMATS *matrixFormats, int jobmask) {

	int ierr;
	int me;
	int me_node;
	char hostname[MAXHOSTNAMELEN];
	int i;

	thishost(hostname);


	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );


	setupSingleNodeComm( hostname, &single_node_comm, &me_node);
#ifdef OCLKERNEL
	int node_rank, node_size;

	ierr = MPI_Comm_size( single_node_comm, &node_size);
	ierr = MPI_Comm_rank( single_node_comm, &node_rank);
	CL_init( node_rank, node_size, hostname, matrixFormats);
#endif

}

