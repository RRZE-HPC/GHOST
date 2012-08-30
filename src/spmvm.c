#include "spmvm.h"
#include "spmvm_util.h"
#include "matricks.h"
#include "mpihelper.h"
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>

int SpMVM_init(int argc, char **argv)
{

	int me, req, prov;
	req = MPI_THREAD_MULTIPLE;
	MPI_safecall(MPI_Init_thread(&argc, &argv, req, &prov ));

	if (req != prov) {
		fprintf(stderr, "Required MPI threading level (%d) is not"
				"provided (%d)!\n",req,prov);
	}
	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));


#ifdef COMPLEX
#ifdef DOUBLE
	MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&MPI_MYDATATYPE));
#endif
#ifdef SINGLE
	MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&MPI_MYDATATYPE));
#endif
	MPI_safecall(MPI_Type_commit(&MPI_MYDATATYPE));
	MPI_safecall(MPI_Op_create((MPI_User_function *)&complAdd,1,&MPI_MYSUM));
#endif

#ifdef PIN
#pragma omp parallel
	{
		cpu_set_t cpu_set;
		CPU_ZERO(&cpu_set);
		CPU_SET(coreNumber, &cpu_set);

		error = sched_setaffinity((pid_t)0, sizeof(cpu_set_t), &cpu_set);

		if (error != 0) {
			printf("pinning thread to core %d failed (%d): %s\n", 
					coreNumber, error, strerror(error));
		}
	}
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

	MPI_safecall(MPI_Finalize());

#ifdef OPENCL
	CL_finish();
#endif




}

CR_TYPE * SpMVM_createCRS (char *matrixPath)
{


	int me;
	CR_TYPE *cr;
	MM_TYPE *mm;

	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));

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


	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));
	int pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;


	VECTOR_TYPE *nodeVec = SpMVM_newVector( pseudo_ldim ); 

	/* Placement of RHS Vector */
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < pseudo_ldim; i++ ) 
		nodeVec->val[i] = 0.0;

	/* Fill up halo with some markers */
	for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) 
		nodeVec->val[i] = 77.0;

	/* Scatter the input vector from the master node to all others */
	MPI_safecall(MPI_Scatterv ( vec->val, lcrp->lnRows, lcrp->lfRow, MPI_MYDATATYPE,
				nodeVec->val, lcrp->lnRows[me], MPI_MYDATATYPE, 0, MPI_COMM_WORLD ));

	return nodeVec;
}

void SpMVM_collectVectors(LCRP_TYPE *lcrp, VECTOR_TYPE *vec, 
		HOSTVECTOR_TYPE *totalVec) {

	int me;


	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));
	MPI_safecall(MPI_Gatherv(vec->val,lcrp->lnRows[me],MPI_MYDATATYPE,totalVec->val,
				lcrp->lnRows,lcrp->lfRow,MPI_MYDATATYPE,0,MPI_COMM_WORLD));
}

LCRP_TYPE * SpMVM_distributeCRS (CR_TYPE *cr, void *deviceFormats)
{
	int me;
	char hostname[MAXHOSTNAMELEN];
	int me_node;

	MPI_safecall(MPI_Comm_rank ( MPI_COMM_WORLD, &me ));
	gethostname(hostname,MAXHOSTNAMELEN);
	setupSingleNodeComm( hostname, &single_node_comm, &me_node);

	LCRP_TYPE *lcrp = setup_communication(cr, WORKDIST_DESIRED);

	if (deviceFormats == NULL) {
#ifdef OPENCL
		myabort("Device matrix formats have to be passed to SPMVM_distributeCRS");
#endif
	}
#ifdef OPENCL
	SPM_GPUFORMATS *formats = (SPM_GPUFORMATS *)deviceFormats;
	CL_uploadCRS ( lcrp, formats);
#endif

	return lcrp;

}
