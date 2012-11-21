#include "ghost_vec.h"
#include "ghost_util.h"

#include <stdio.h>

void SpMVM_zeroVector(ghost_vec_t *vec) 
{
	int i;
	for (i=0; i<vec->nrows; i++) {
#ifdef COMPLEX
		vec->val[i] = 0;
#else
		vec->val[i] = 0+I*0;
#endif
	}

#ifdef OPENCL
	CL_uploadVector(vec);
#endif


}

ghost_vec_t* SpMVM_newVector( const int nrows, unsigned int flags ) 
{
	ghost_vec_t* vec;
	size_t size_val;
	int i;

	size_val = (size_t)( nrows * sizeof(mat_data_t) );
	vec = (ghost_vec_t*) allocateMemory( sizeof( ghost_vec_t ), "vec");


	vec->val = (mat_data_t*) allocateMemory( size_val, "vec->val");
	vec->nrows = nrows;
	vec->flags = flags;

#pragma omp parallel for schedule(runtime) 
	for( i = 0; i < nrows; i++ ) 
		vec->val[i] = 0.0;

#ifdef OPENCL
#ifdef CL_IMAGE
	vec->CL_val_gpu = CL_allocDeviceMemoryCached( size_val,vec->val );
#else
	vec->CL_val_gpu = CL_allocDeviceMemoryMapped( size_val,vec->val,CL_MEM_READ_WRITE );
#endif
	//vec->CL_val_gpu = CL_allocDeviceMemory( size_val );
	//printf("before: %p\n",vec->val);
	//vec->val = CL_mapBuffer(vec->CL_val_gpu,size_val);
	//printf("after: %p\n",vec->val);
	//CL_uploadVector(vec);
#endif

	return vec;
}

ghost_vec_t * SpMVM_distributeVector(ghost_comm_t *lcrp, ghost_vec_t *vec)
{
	DEBUG_LOG(1,"Distributing vector");
#ifdef MPI
	int me = SpMVM_getRank();

	int nrows;

	MPI_safecall(MPI_Bcast(&(vec->flags),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));

	if (vec->flags & ghost_vec_t_LHS)
		nrows = lcrp->lnrows[me];
	else if ((vec->flags & ghost_vec_t_RHS) || (vec->flags & ghost_vec_t_BOTH))
		nrows = lcrp->lnrows[me]+lcrp->halo_elements;
	else
		ABORT("No valid type for vector (has to be one of ghost_vec_t_LHS/_RHS/_BOTH");


	DEBUG_LOG(2,"Creating local vector with %d rows",nrows);
	ghost_vec_t *nodeVec = SpMVM_newVector( nrows, vec->flags ); 

	DEBUG_LOG(2,"Scattering global vector to local vectors");
	MPI_safecall(MPI_Scatterv ( vec->val, (int *)lcrp->lnrows, (int *)lcrp->lfRow, MPI_MYDATATYPE,
				nodeVec->val, (int)lcrp->lnrows[me], MPI_MYDATATYPE, 0, MPI_COMM_WORLD ));
#else
	UNUSED(lcrp);
	ghost_vec_t *nodeVec = SpMVM_newVector( vec->nrows, vec->flags ); 
	int i;
	for (i=0; i<vec->nrows; i++) nodeVec->val[i] = vec->val[i];
#endif

#ifdef OPENCL // TODO depending on flag
	CL_uploadVector(nodeVec);
#endif

	DEBUG_LOG(1,"Vector distributed successfully");
	return nodeVec;
}

void SpMVM_collectVectors(ghost_setup_t *setup, ghost_vec_t *vec, ghost_vec_t *totalVec, int kernel) 
{

#ifdef MPI
	// TODO
	//if (matrix->trait.format != GHOST_SPMFORMAT_CRS)
	//	DEBUG_LOG(0,"Cannot handle other matrices than CRS in the MPI case!");

	int me = SpMVM_getRank();
	if ( 0x1<<kernel & GHOST_MODES_COMBINED)  {
		SpMVM_permuteVector(vec->val,setup->fullMatrix->invRowPerm,setup->communicator->lnrows[me]);
	} else if ( 0x1<<kernel & GHOST_MODES_SPLIT ) {
		// one of those must return immediately
		SpMVM_permuteVector(vec->val,setup->localMatrix->invRowPerm,setup->communicator->lnrows[me]);
		SpMVM_permuteVector(vec->val,setup->remoteMatrix->invRowPerm,setup->communicator->lnrows[me]);
	}
	MPI_safecall(MPI_Gatherv(vec->val,(int)setup->communicator->lnrows[me],MPI_MYDATATYPE,totalVec->val,
				(int *)setup->communicator->lnrows,(int *)setup->communicator->lfRow,MPI_MYDATATYPE,0,MPI_COMM_WORLD));
#else
	int i;
	UNUSED(kernel);
	SpMVM_permuteVector(vec->val,setup->fullMatrix->invRowPerm,setup->fullMatrix->nrows(setup->fullMatrix));
	for (i=0; i<totalVec->nrows; i++) 
		totalVec->val[i] = vec->val[i];
#endif
}

void SpMVM_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2) 
{
	mat_data_t *dtmp;

	dtmp = v1->val;
	v1->val = v2->val;
	v2->val = dtmp;
#ifdef OPENCL
	cl_mem tmp;
	tmp = v1->CL_val_gpu;
	v1->CL_val_gpu = v2->CL_val_gpu;
	v2->CL_val_gpu = tmp;
#endif

}

void SpMVM_normalizeVector( ghost_vec_t *vec)
{
	int i;
	mat_data_t sum = 0;

	for (i=0; i<vec->nrows; i++)	
		sum += vec->val[i]*vec->val[i];

	mat_data_t f = (mat_data_t)1/SQRT(ABS(sum));

	for (i=0; i<vec->nrows; i++)	
		vec->val[i] *= f;

#ifdef OPENCL
	CL_uploadVector(vec);
#endif
}

void SpMVM_freeVector( ghost_vec_t* const vec ) 
{
	if( vec ) {
		freeMemory( (size_t)(vec->nrows*sizeof(mat_data_t)), "vec->val",  vec->val );
#ifdef OPENCL
		CL_freeDeviceMemory( vec->CL_val_gpu );
#endif
		free( vec );
	}
}

void SpMVM_permuteVector( mat_data_t* vec, mat_idx_t* perm, mat_idx_t len) 
{
	/* permutes values in vector so that i-th entry is mapped to position perm[i] */
	mat_idx_t i;
	mat_data_t* tmp;

	if (perm == NULL) {
		DEBUG_LOG(1,"Permutation vector is NULL, returning.");
		return;
	} else {
		DEBUG_LOG(1,"Permuting vector");
	}


	tmp = (mat_data_t*)allocateMemory(sizeof(mat_data_t)*len, "permute tmp");

	for(i = 0; i < len; ++i) {
		if( perm[i] >= len ) {
			ABORT("Permutation index out of bounds: %"PRmatIDX" > %"PRmatIDX,perm[i],len);
		}
		tmp[perm[i]] = vec[i];
	}
	for(i=0; i < len; ++i) {
		vec[i] = tmp[i];
	}

	free(tmp);
}
