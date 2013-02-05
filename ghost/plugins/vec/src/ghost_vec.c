#include "ghost_vec.h"
#include "ghost_util.h"

#include <stdio.h>
#include <string.h>

#ifdef CUDA
#include <cuda_runtime.h> // TODO in cu_util
#endif

#ifdef GHOST_VEC_COMPLEX
#ifdef GHOST_VEC_DP
#define VAL(vec) ((double complex *)(vec->val))
#else
#define VAL(vec) ((float complex *)(vec->val))
#endif
#else
#ifdef GHOST_VEC_DP
#define VAL(vec) ((double *)(vec->val))
#else
#define VAL(vec) ((float *)(vec->val))
#endif
#endif


const char name[] = "Vector plugin for ghost";
const char version[] = "0.1a";

static void         ghost_zeroVector(ghost_vec_t *vec);
static ghost_vec_t *ghost_newVector( const int nrows, unsigned int flags );
static void         ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2);
static void         ghost_normalizeVector( ghost_vec_t *vec);
static ghost_vec_t * ghost_distributeVector(ghost_vec_t *vec, ghost_comm_t *comm);
static void ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec, ghost_context_t *context);
static void         ghost_freeVector( ghost_vec_t* const vec );
static void ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm); 
static int ghost_vecEquals(ghost_vec_t *a, ghost_vec_t *b);
static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src);
static ghost_vdat_t *ghost_vecVal(ghost_vec_t *vec);

ghost_vec_t *init(ghost_vtraits_t *traits)
{
	ghost_vec_t *vec = (ghost_vec_t *)allocateMemory(sizeof(ghost_vec_t),"vector");
	vec->traits = traits;

	DEBUG_LOG(1,"Initializing vector");

	vec->zero = &ghost_zeroVector;
	vec->distribute = &ghost_distributeVector;
	vec->collect = &ghost_collectVectors;
	vec->swap = &ghost_swapVectors;
	vec->normalize = &ghost_normalizeVector;
	vec->destroy = &ghost_freeVector;
	vec->permute = &ghost_permuteVector;
	vec->equals = &ghost_vecEquals;
	vec->clone = &ghost_cloneVector;
	vec->val = &ghost_vecVal;

	return vec;
}

static void ghost_zeroVector(ghost_vec_t *vec) 
{
	DEBUG_LOG(1,"Zeroing vector");
	int i;
	for (i=0; i<vec->traits->nrows; i++) {
#ifdef GHOST_VEC_COMPLEX
		VAL(vec)[i] = 0.+I*0.;
#else
		VAL(vec)[i] = 0.;
#endif
	}

#ifdef OPENCL
	CL_uploadVector(vec);
#endif
#ifdef CUDA
	CU_uploadVector(vec);
#endif


}

static ghost_vec_t* ghost_newVector( const int nrows, unsigned int flags ) 
{
	ghost_vec_t* vec;
	size_t size_val;
	int i;

	size_val = (size_t)( ghost_pad(nrows,VEC_PAD) * sizeof(ghost_vdat_t) );
	vec = (ghost_vec_t*) allocateMemory( sizeof( ghost_vec_t ), "vec");


	VAL(vec) = (ghost_vdat_t*) allocateMemory( size_val, "VAL(vec)");
	vec->traits->nrows = nrows;
	vec->traits->flags = flags;

#pragma omp parallel for schedule(runtime) 
	for( i = 0; i < nrows; i++ ) 
		VAL(vec)[i] = 0.0;

#ifdef OPENCL
#ifdef CL_IMAGE
	vec->CL_val_gpu = CL_allocDeviceMemoryCached( size_val,VAL(vec) );
#else
	vec->CL_val_gpu = CL_allocDeviceMemoryMapped( size_val,VAL(vec),CL_MEM_READ_WRITE );
#endif
	//vec->CL_val_gpu = CL_allocDeviceMemory( size_val );
	//printf("before: %p\n",VAL(vec));
	//VAL(vec) = CL_mapBuffer(vec->CL_val_gpu,size_val);
	//printf("after: %p\n",VAL(vec));
	//CL_uploadVector(vec);
#endif
#ifdef CUDA
	vec->CU_val = CU_allocDeviceMemory(size_val);
#endif

	return vec;
}

static ghost_vec_t * ghost_distributeVector(ghost_vec_t *vec, ghost_comm_t *comm)
{
	DEBUG_LOG(1,"Distributing vector");
#ifdef MPI
	int me = ghost_getRank();

	ghost_vidx_t nrows;

	MPI_safecall(MPI_Bcast(&(vec->traits->flags),1,MPI_INT,0,MPI_COMM_WORLD));

	if (vec->traits->flags & GHOST_VEC_RHS)
		nrows = comm->lnrows[me]+comm->halo_elements;
	else if (vec->traits->flags & GHOST_VEC_LHS)
		nrows = comm->lnrows[me];
	else
		ABORT("No valid type for vector (has to be one of GHOST_VEC_LHS/_RHS/_BOTH");


	DEBUG_LOG(2,"Creating local vector with %"PRvecIDX" rows",nrows);
	ghost_vec_t *nodeVec = ghost_newVector( nrows, vec->traits->flags ); 

	DEBUG_LOG(2,"Scattering global vector to local vectors");
	MPI_safecall(MPI_Scatterv ( VAL(vec), (int *)comm->lnrows, (int *)comm->lfRow, ghost_mpi_dt_vdat,
				VAL(nodeVec), (int)comm->lnrows[me], ghost_mpi_dt_vdat, 0, MPI_COMM_WORLD ));
#else
	UNUSED(comm);
	ghost_vec_t *nodeVec = ghost_newVector( vec->traits->nrows, vec->traits->flags ); 
	int i;
	for (i=0; i<vec->traits->nrows; i++) VAL(nodeVec)[i] = VAL(vec)[i];
#endif

#ifdef OPENCL // TODO depending on flag
	CL_uploadVector(nodeVec);
#endif
#ifdef CUDA // TODO depending on flag
	CU_uploadVector(nodeVec);
#endif

	DEBUG_LOG(1,"Vector distributed successfully");
	return nodeVec;
}

static void ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec, ghost_context_t *context) 
{

#ifdef MPI
	// TODO
	//if (matrix->trait.format != GHOST_SPMFORMAT_CRS)
	//	DEBUG_LOG(0,"Cannot handle other matrices than CRS in the MPI case!");

	int me = ghost_getRank();
	//TODO permute
	/*if ( 0x1<<kernel & GHOST_SPMVM_MODES_COMBINED)  {
		ghost_permuteVector(VAL(vec),context->fullMatrix->invRowPerm,context->communicator->lnrows[me]);
	} else if ( 0x1<<kernel & GHOST_SPMVM_MODES_SPLIT ) {
		// one of those must return immediately
		ghost_permuteVector(VAL(vec),context->localMatrix->invRowPerm,context->communicator->lnrows[me]);
		ghost_permuteVector(VAL(vec),context->remoteMatrix->invRowPerm,context->communicator->lnrows[me]);
	}*/
	MPI_safecall(MPI_Gatherv(VAL(vec),(int)context->communicator->lnrows[me],ghost_mpi_dt_vdat,totalVec->val,
				(int *)context->communicator->lnrows,(int *)context->communicator->lfRow,ghost_mpi_dt_vdat,0,MPI_COMM_WORLD));
#else
	int i;
//	UNUSED(kernel);
	vec->permute(vec,context->fullMatrix->invRowPerm);
	for (i=0; i<totalVec->traits->nrows; i++) 
		VAL(totalVec)[i] = VAL(vec)[i];
#endif
}

static void ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2) 
{
	ghost_vdat_t *dtmp;

	dtmp = v1->val;
	v1->val = v2->val;
	v2->val = dtmp;
#ifdef OPENCL
	cl_mem tmp;
	tmp = v1->CL_val_gpu;
	v1->CL_val_gpu = v2->CL_val_gpu;
	v2->CL_val_gpu = tmp;
#endif
#ifdef OPENCL
	dtmp = v1->CU_val;
	v1->CU_val = v2->CU_val;
	v2->CU_val = dtmp;
#endif

}

static void ghost_normalizeVector( ghost_vec_t *vec)
{
	int i;
	ghost_vdat_t sum = 0;

	for (i=0; i<vec->traits->nrows; i++)	
		sum += VAL(vec)[i]*VAL(vec)[i];

	ghost_vdat_t f = (ghost_vdat_t)1./VSQRT(VABS(sum));

	for (i=0; i<vec->traits->nrows; i++)	
		VAL(vec)[i] *= f;

#ifdef OPENCL
	CL_uploadVector(vec);
#endif
#ifdef CUDA
	CU_uploadVector(vec);
#endif
}

static void ghost_freeVector( ghost_vec_t* const vec ) 
{
	if( vec ) {
#ifdef CUDA_PINNEDMEM
		if (vec->traits->flags & GHOST_VEC_DEVICE)
			CU_safecall(cudaFreeHost(VAL(vec)));
#else
		free(VAL(vec));
#endif
//		freeMemory( (size_t)(vec->traits->nrows*sizeof(ghost_mdat_t)), "VAL(vec)",  VAL(vec) );
#ifdef OPENCL
		if (vec->traits->flags & GHOST_VEC_DEVICE)
			CL_freeDeviceMemory( vec->CL_val_gpu );
#endif
#ifdef CUDA
		if (vec->traits->flags & GHOST_VEC_DEVICE)
			CU_freeDeviceMemory( vec->CU_val );
#endif
		free( vec );
	}
}

static void ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm) 
{
	/* permutes values in vector so that i-th entry is mapped to position perm[i] */
	ghost_midx_t i;
	ghost_vidx_t len = vec->traits->nrows;
	ghost_vdat_t* tmp;

	if (perm == NULL) {
		DEBUG_LOG(1,"Permutation vector is NULL, returning.");
		return;
	} else {
		DEBUG_LOG(1,"Permuting vector");
	}


	tmp = (ghost_vdat_t*)allocateMemory(sizeof(ghost_vdat_t)*len, "permute tmp");

	for(i = 0; i < len; ++i) {
		if( perm[i] >= len ) {
			ABORT("Permutation index out of bounds: %"PRmatIDX" > %"PRmatIDX,perm[i],len);
		}
		tmp[perm[i]] = VAL(vec)[i];
	}
	for(i=0; i < len; ++i) {
		VAL(vec)[i] = tmp[i];
	}

	free(tmp);
}

static int ghost_vecEquals(ghost_vec_t *a, ghost_vec_t *b)
{
	double tol = 1e-5;
	int i;
	for (i=0; i<a->traits->nrows; i++) {
		if (VREAL(VABS(VAL(a)[i]-VAL(b)[i])) > tol || VIMAG(VABS(VAL(a)[i]-VAL(b)[i])) > tol)
			return 0;
	}

	return 1;

}

static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src)
{
	ghost_vec_t *new = ghost_newVector(src->traits->nrows, src->traits->flags);
	memcpy(new->val, src->val, src->traits->nrows*sizeof(ghost_vdat_t));

	return new;
}

static ghost_vdat_t *ghost_vecVal(ghost_vec_t *vec)
{

	return (ghost_vdat_t *)vec->val;
}
