#include "matricks.h"
#include <sys/param.h>
#include <libgen.h>
#include "oclfun.h"


LCRP_TYPE * SpMVM_init (char *matrixPath, MATRIX_FORMATS *matrixFormats, VECTOR_TYPE **hlpvec_out, VECTOR_TYPE **hlpvec_in, VECTOR_TYPE **resCR) {

	int ierr;
	int me;
	int me_node;
	char hostname[MAXHOSTNAMELEN];
	int i;

	VECTOR_TYPE* rhsVec = NULL;
	thishost(hostname);
	CR_TYPE *cr;
	MM_TYPE *mm;


	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );


	setupSingleNodeComm( hostname, &single_node_comm, &me_node);
#ifdef OCLKERNEL
	int node_rank, node_size;

	ierr = MPI_Comm_size( single_node_comm, &node_size);
	ierr = MPI_Comm_rank( single_node_comm, &node_rank);
	CL_init( node_rank, node_size, hostname, matrixFormats);
#endif
	
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
		/* convert column indices in CRS format to FORTRAN-numbering, required for CPU kernel */
		crColIdToFortran(cr);


		rhsVec = newVector( cr->nCols );
		*resCR = newVector( cr->nCols );

		for (i=0; i<cr->nCols; i++) 
			rhsVec->val[i] = i+1;

		/* Serial CRS-multiplication to get reference result */
		fortrancrs_(&(cr->nRows), &(cr->nEnts), 
				(*resCR)->val, rhsVec->val, cr->val , cr->col, cr->rowOffset);

	} else{

		/* Allokiere minimalen Speicher fuer Dummyversion der globalen Matrix */
		mm            = (MM_TYPE*) allocateMemory( sizeof(MM_TYPE), "mm" );
		cr            = (CR_TYPE*) allocateMemory( sizeof(CR_TYPE), "cr" );
		cr->nRows     = 0;
		cr->nEnts     = 1;
		cr->rowOffset = (int*)     allocateMemory( sizeof(int),     "rowOffset" );
		cr->col       = (int*)     allocateMemory( sizeof(int),     "col" );
		cr->val       = (double*)  allocateMemory( sizeof(double),  "val" );
		rhsVec = newVector( 1 );
		*resCR  = newVector( 1 );
	}
	
	LCRP_TYPE *lcrp = setup_communication(cr, 1,matrixFormats);

#ifdef OCLKERNEL
	if( jobmask & 503 ) { 
		CL_bindMatrixToKernel(lcrp->fullMatrix,lcrp->fullFormat,matrixFormats->T[SPM_KERNEL_FULL],SPM_KERNEL_FULL);
	} 
	if( jobmask & 261640 ) { // only if jobtype requires split computation
		CL_bindMatrixToKernel(lcrp->localMatrix,lcrp->localFormat,matrixFormats->T[SPM_KERNEL_LOCAL],SPM_KERNEL_LOCAL);
		CL_bindMatrixToKernel(lcrp->remoteMatrix,lcrp->remoteFormat,matrixFormats->T[SPM_KERNEL_REMOTE],SPM_KERNEL_REMOTE);
	}
#endif

	int pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;


	*hlpvec_out = newVector( lcrp->lnRows[me] );
	*hlpvec_in = newVector( pseudo_ldim );  

#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnRows[me]; i++) 
		(*hlpvec_out)->val[i] = -63.5;


	/* Placement of RHS Vector */
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < pseudo_ldim; i++ ) 
		(*hlpvec_in)->val[i] = 0.0;
	
		/* Fill up halo with some markers */
	for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) 
		(*hlpvec_in)->val[i] = 77.0;

	/* Scatter the input vector from the master node to all others */
	ierr = MPI_Scatterv ( rhsVec->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
			(*hlpvec_in)->val, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

	freeVector(rhsVec);
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


	
