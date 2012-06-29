#include "matricks.h"
#include <sys/param.h>
#include "oclfun.h"
#include <libgen.h>


CR_TYPE * SpMVM_createCRS (char *matrixPath) {

	int ierr;
	int me;
	int i;

	CR_TYPE *cr;
	MM_TYPE *mm;


	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );

	
	if (me == 0){
		if (!isMMfile(matrixPath)){
			cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
			bin_read_cr(cr, matrixPath);
		} else{
			mm = readMMFile( matrixPath, 0.0 );
			cr = convertMMToCRMatrix( mm );
			bin_write_cr(cr, basename(matrixPath));
			freeMMMatrix(mm);
		}

	} else{

		/* Allokiere minimalen Speicher fuer Dummyversion der globalen Matrix */
		cr            = (CR_TYPE*) allocateMemory( sizeof(CR_TYPE), "cr" );
		cr->nRows     = 0;
		cr->nEnts     = 1;
		cr->rowOffset = (int*)     allocateMemory( sizeof(int),     "rowOffset" );
		cr->col       = (int*)     allocateMemory( sizeof(int),     "col" );
		cr->val       = (double*)  allocateMemory( sizeof(double),  "val" );
	}
	return cr;
	
}

VECTOR_TYPE * SpMVM_distributeVector(LCRP_TYPE *lcrp, HOSTVECTOR_TYPE *vec) {
	int ierr;
	int me;
	int i;


	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );
	int pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;


	VECTOR_TYPE *nodeVec = newVector( pseudo_ldim ); 

	/* Placement of RHS Vector */
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < pseudo_ldim; i++ ) 
		nodeVec->val[i] = 0.0;
	
		/* Fill up halo with some markers */
	for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) 
		nodeVec->val[i] = 77.0;

	/* Scatter the input vector from the master node to all others */
	ierr = MPI_Scatterv ( vec->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE,nodeVec->val, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

	return nodeVec;
}


LCRP_TYPE * SpMVM_init (CR_TYPE *cr, MATRIX_FORMATS *matrixFormats) {

	int ierr;
	int me;
	int i;
	char hostname[MAXHOSTNAMELEN];
	int me_node;



	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );
	setupSingleNodeComm( hostname, &single_node_comm, &me_node);
	thishost(hostname);

#ifdef OCLKERNEL
	int node_rank, node_size;

	ierr = MPI_Comm_size( single_node_comm, &node_size);
	ierr = MPI_Comm_rank( single_node_comm, &node_rank);
	CL_init( node_rank, node_size, hostname, matrixFormats);
#endif

	LCRP_TYPE *lcrp = setup_communication(cr, 1,matrixFormats);

#ifdef OCLKERNEL
	if( jobmask & 503 ) { // only if jobtype requires combined computation
		CL_bindMatrixToKernel(lcrp->fullMatrix,lcrp->fullFormat,matrixFormats->T[SPM_KERNEL_FULL],SPM_KERNEL_FULL);
	}
	
	if( jobmask & 261640 ) { // only if jobtype requires split computation
		CL_bindMatrixToKernel(lcrp->localMatrix,lcrp->localFormat,matrixFormats->T[SPM_KERNEL_LOCAL],SPM_KERNEL_LOCAL);
		CL_bindMatrixToKernel(lcrp->remoteMatrix,lcrp->remoteFormat,matrixFormats->T[SPM_KERNEL_REMOTE],SPM_KERNEL_REMOTE);
	}
#endif

	return lcrp;

}

void printMatrixInfo(LCRP_TYPE *lcrp, char *matrixName) {

	int me;
	size_t ws;
	int ierr;

	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );

#ifdef OCLKERNEL	
	size_t fullMemSize, localMemSize, remoteMemSize, 
		   totalFullMemSize = 0, totalLocalMemSize = 0, totalRemoteMemSize = 0;

	if( jobmask & 503 ) { 
		fullMemSize = getBytesize(lcrp->fullMatrix,lcrp->fullFormat)/(1024*1024);
		MPI_Reduce(&fullMemSize, &totalFullMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);

	} 
	if( jobmask & 261640 ) { // only if jobtype requires split computation
		localMemSize = getBytesize(lcrp->localMatrix,lcrp->localFormat)/(1024*1024);
		remoteMemSize = getBytesize(lcrp->remoteMatrix,lcrp->remoteFormat)/(1024*1024);
		MPI_Reduce(&localMemSize, &totalLocalMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&remoteMemSize, &totalRemoteMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
	}
#endif	

	if(me==0){
		ws = ((lcrp->nRows+1)*sizeof(int) + lcrp->nEnts*(sizeof(double)+sizeof(int)))/(1024*1024);
		printf("-----------------------------------------------------\n");
		printf("-------         Statistics about matrix       -------\n");
		printf("-----------------------------------------------------\n");
		printf("Investigated matrix         : %12s\n", matrixName); 
		printf("Dimension of matrix         : %12.0f\n", (float)lcrp->nRows); 
		printf("Non-zero elements           : %12.0f\n", (float)lcrp->nEnts); 
		printf("Average elements per row    : %12.3f\n", (float)lcrp->nEnts/(float)lcrp->nRows); 
		printf("Working set             [MB]: %12lu\n", ws);
#ifdef OCLKERNEL	
		if( jobmask & 503 ) 
			printf("Device matrix (combin.) [MB]: %12lu\n", totalFullMemSize); 
		if( jobmask & 261640 ) {
			printf("Device matrix (local)   [MB]: %12lu\n", totalLocalMemSize); 
			printf("Device matrix (remote)  [MB]: %12lu\n", totalRemoteMemSize); 
			printf("Device matrix (loc+rem) [MB]: %12lu\n", totalLocalMemSize+totalRemoteMemSize); 
		}
#endif
		printf("-----------------------------------------------------\n");
		fflush(stdout);
	}
}


	
