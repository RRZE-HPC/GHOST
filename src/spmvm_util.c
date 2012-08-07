#include "spmvm_util.h"
#include "matricks.h"
#include <sys/param.h>
#include <libgen.h>
#include <unistd.h>
#include <mpihelper.h>

#ifdef OPENCL
#include "oclfun.h"
#include "my_ellpack.h"
#endif

#ifdef LIKWID
#include <likwid.h>
#endif

#include <omp.h>



#ifdef COMPLEX
typedef struct {
#ifdef DOUBLE
	double x;
	double y;
#endif
#ifdef SINGLE
	float x;
	float y;
#endif

} MPI_complex;

void complAdd(MPI_complex *invec, MPI_complex *inoutvec, int *len, 
		MPI_Datatype *datatype)
{

	int i;
	MPI_complex c;

	for (i=0; i<*len; i++, invec++, inoutvec++){
		c.x = invec->x + inoutvec->x;
		c.y = invec->y + inoutvec->y;
		*inoutvec = c;
	}
}
#endif

int SpMVM_init(int argc, char **argv)
{

	int me, req, prov;
	req = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, req, &prov );

	if (req != prov)
		fprintf(stderr, "Required MPI threading level (%d) is not \
				provided (%d)!\n",req,prov);
	MPI_Comm_rank ( MPI_COMM_WORLD, &me );


#ifdef COMPLEX
#ifdef DOUBLE
	MPI_Type_contiguous(2,MPI_DOUBLE,&MPI_MYDATATYPE);
#endif
#ifdef SINGLE
	MPI_Type_contiguous(2,MPI_FLOAT,&MPI_MYDATATYPE);
#endif
	MPI_Type_commit(&MPI_MYDATATYPE);
	MPI_Op_create((MPI_User_function *)&complAdd,TRUE,&MPI_MYSUM);
#endif

#ifdef LIKWID_MARKER
	likwid_markerInit();
#endif

	return me;
}

void SpMVM_finish()
{

#ifdef LIKWID_MARKER
	likwid_markerClose();
#endif

	MPI_Finalize();

#ifdef OPENCL
	CL_finish();
#endif




}

CR_TYPE * SpMVM_createCRS (char *matrixPath)
{

	
	int me;
	CR_TYPE *cr;
	MM_TYPE *mm;

	MPI_Comm_rank ( MPI_COMM_WORLD, &me );

	if (me == 0){
		if (!isMMfile(matrixPath)){
			cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
			readCRbinFile(cr, matrixPath);
		} else{
			mm = readMMFile( matrixPath);
			cr = convertMMToCRMatrix( mm );
			freeMMMatrix(mm);
		}
		crColIdToFortran(cr);

	} else{

		/* Allokiere minimalen Speicher fuer Dummyversion der globalen Matrix */
		cr            = (CR_TYPE*) allocateMemory( sizeof(CR_TYPE), "cr");
		cr->nRows     = 0;
		cr->nEnts     = 1;
		cr->rowOffset = (int*)     allocateMemory(sizeof(int), "rowOffset");
		cr->col       = (int*)     allocateMemory(sizeof(int), "col");
		cr->val       = (real*)  allocateMemory(sizeof(real), "val");
	}
	return cr;

}

VECTOR_TYPE * SpMVM_distributeVector(LCRP_TYPE *lcrp, HOSTVECTOR_TYPE *vec)
{
	
	int me;
	int i;


	MPI_Comm_rank ( MPI_COMM_WORLD, &me );
	int pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;


	VECTOR_TYPE *nodeVec = newVector( pseudo_ldim ); 

	/* Placement of RHS Vector */
#pragma omp parallel for
	for( i = 0; i < pseudo_ldim; i++ ) 
		nodeVec->val[i] = 0.0;

	/* Fill up halo with some markers */
	for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) 
		nodeVec->val[i] = 77.0;

	/* Scatter the input vector from the master node to all others */
	MPI_Scatterv ( vec->val, lcrp->lnRows, lcrp->lfRow, MPI_MYDATATYPE,
			nodeVec->val, lcrp->lnRows[me], MPI_MYDATATYPE, 0, MPI_COMM_WORLD );

	return nodeVec;
}

void SpMVM_collectVectors(LCRP_TYPE *lcrp, VECTOR_TYPE *vec, 
		HOSTVECTOR_TYPE *totalVec) {
	
	int me;


	MPI_Comm_rank ( MPI_COMM_WORLD, &me );
	MPI_Gatherv(vec->val,lcrp->lnRows[me],MPI_MYDATATYPE,totalVec->val,
			lcrp->lnRows,lcrp->lfRow,MPI_MYDATATYPE,0,MPI_COMM_WORLD);
}

LCRP_TYPE * SpMVM_distributeCRS (CR_TYPE *cr)
{
	int me;
	char hostname[MAXHOSTNAMELEN];
	int me_node;

	MPI_Comm_rank ( MPI_COMM_WORLD, &me );
	gethostname(hostname,MAXHOSTNAMELEN);
	setupSingleNodeComm( hostname, &single_node_comm, &me_node);

	LCRP_TYPE *lcrp = setup_communication(cr, 1);


	return lcrp;

}

void SpMVM_printMatrixInfo(LCRP_TYPE *lcrp, char *matrixName)
{

	int me;
	size_t ws;
	

	MPI_Comm_rank ( MPI_COMM_WORLD, &me );

#ifdef OPENCL	
	size_t fullMemSize, localMemSize, remoteMemSize, 
		   totalFullMemSize = 0, totalLocalMemSize = 0, totalRemoteMemSize = 0;

	if (SPMVM_KERNELS_SELECTED & SPMVM_KERNELS_COMBINED) { // combined computation
		fullMemSize = getBytesize(lcrp->fullMatrix, lcrp->fullFormat)/
			(1024*1024);
		MPI_Reduce(&fullMemSize, &totalFullMemSize,1,MPI_LONG,MPI_SUM,0,
				MPI_COMM_WORLD);
	} 
	if (SPMVM_KERNELS_SELECTED & SPMVM_KERNELS_SPLIT) { // split computation
		localMemSize = getBytesize(lcrp->localMatrix,lcrp->localFormat)/
			(1024*1024);
		remoteMemSize = getBytesize(lcrp->remoteMatrix,lcrp->remoteFormat)/
			(1024*1024);
		MPI_Reduce(&localMemSize, &totalLocalMemSize,1,MPI_LONG,MPI_SUM,0,
				MPI_COMM_WORLD);
		MPI_Reduce(&remoteMemSize, &totalRemoteMemSize,1,MPI_LONG,MPI_SUM,0,
				MPI_COMM_WORLD);
	}
#endif	

	if(me==0){
		ws = ((lcrp->nRows+1)*sizeof(int) + 
				lcrp->nEnts*(sizeof(real)+sizeof(int)))/(1024*1024);
		printf("-----------------------------------------------------\n");
		printf("-------         Statistics about matrix       -------\n");
		printf("-----------------------------------------------------\n");
		printf("Investigated matrix         : %12s\n", matrixName); 
		printf("Dimension of matrix         : %12.0f\n", (double)lcrp->nRows); 
		printf("Non-zero elements           : %12.0f\n", (double)lcrp->nEnts); 
		printf("Average elements per row    : %12.3f\n", (double)lcrp->nEnts/
				(double)lcrp->nRows); 
		printf("CRS matrix              [MB]: %12lu\n", ws);
#ifdef OPENCL	
		if (SPMVM_KERNELS_SELECTED & SPMVM_KERNELS_COMBINED) { // combined computation
			printf("Device matrix (combin.) [MB]: %12lu\n", totalFullMemSize);
		}	
		if (SPMVM_KERNELS_SELECTED & SPMVM_KERNELS_SPLIT) { // split computation
			printf("Device matrix (local)   [MB]: %12lu\n", totalLocalMemSize); 
			printf("Device matrix (remote)  [MB]: %12lu\n", totalRemoteMemSize);
			printf("Device matrix (loc+rem) [MB]: %12lu\n", totalLocalMemSize+
					totalRemoteMemSize); 
		}
#endif
		printf("-----------------------------------------------------\n\n");
		fflush(stdout);
	}
}

void SpMVM_printEnvInfo() {

	int me;
	MPI_Comm_rank ( MPI_COMM_WORLD, &me );

	if (me==0) {
	int nproc;
	int nthreads;
	MPI_Comm_size ( MPI_COMM_WORLD, &nproc );

#pragma omp parallel
#pragma omp master
	nthreads = omp_get_num_threads();

		printf("-----------------------------------------------------\n");
		printf("-------       Statistics about environment    -------\n");
		printf("-----------------------------------------------------\n");
		printf("MPI processes               : %12d\n", nproc); 
		printf("OpenMP threads per process  : %12d\n", nthreads);
		printf("Data type                   : %12s\n", DATATYPE_NAMES[DATATYPE_DESIRED]);
#ifdef OPENCL
		printf("OpenCL                      :      enabled\n");

#endif
	   //TODO gpu info	
		printf("-----------------------------------------------------\n\n");
		fflush(stdout);

	}

}

HOSTVECTOR_TYPE * SpMVM_createGlobalHostVector(int nRows, real (*fp)(int))
{
	
	int me;
	MPI_Comm_rank ( MPI_COMM_WORLD, &me );

	if (me==0) {
		return newHostVector( nRows,fp );
	} else {
		return newHostVector(0,NULL);
	}
}

void SpMVM_referenceSolver(CR_TYPE *cr, real *rhs, real *lhs, int nIter) 
{

	int iteration;
	if (SPMVM_OPTIONS & SPMVM_OPTION_AXPY) {

		for (iteration=0; iteration<nIter; iteration++) {
#ifdef DOUBLE
#ifdef COMPLEX
			fortrancrsaxpyc_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val ,
				   	cr->col, cr->rowOffset);
#else
			fortrancrsaxpy_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val ,
				   	cr->col, cr->rowOffset);
#endif
#endif
#ifdef SINGLE
#ifdef COMPLEX
			fortrancrsaxpycf_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val,
				   	cr->col, cr->rowOffset);
#else
			fortrancrsaxpyf_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val,
				   	cr->col, cr->rowOffset);
#endif
#endif
		}
	} else {
#ifdef DOUBLE
#ifdef COMPLEX
		fortrancrsc_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val, cr->col,
			   	cr->rowOffset);
#else
		fortrancrs_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val, cr->col,
			   	cr->rowOffset);
#endif
#endif
#ifdef SINGLE
#ifdef COMPLEX
		fortrancrscf_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val, cr->col,
			   	cr->rowOffset);
#else
		fortrancrsf_(&(cr->nRows), &(cr->nEnts), lhs, rhs, cr->val , cr->col,
			   	cr->rowOffset);
#endif
#endif
	}
}

int SpMVM_kernelValid(int kernel, LCRP_TYPE *lcrp) {
	
		if (!(0x1<<kernel & SPMVM_KERNELS_SELECTED)) 
			return 0; // kernel not selected
		if ((0x1<<kernel & SPMVM_KERNEL_NOMPI)  && lcrp->nodes>1) 
			return 0; // non-MPI kernel
		if ((0x1<<kernel & SPMVM_KERNEL_TASKMODE) &&  lcrp->threads==1) 
			return 0; // not enough threads

		return 1;
}
