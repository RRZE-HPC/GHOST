#define _XOPEN_SOURCE 500 
#include "ghost_types.h"
#include "ghost_vec.h"
#include "ghost_util.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


#ifdef CUDA
#include <cuda_runtime.h> // TODO in cu_util
#endif

#define VAL(vec) ((ghost_dt *)(vec->val))

const char name[] = "Vector plugin for ghost";
const char version[] = "0.1a";

static void vec_print(ghost_vec_t *vec);
static void vec_scale(ghost_vec_t *vec, void *scale);
static void vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale);
static void vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res);
static void vec_fromFunc(ghost_vec_t *vec, void (*fp)(int,int,void *));
static void vec_fromVec(ghost_vec_t *vec, ghost_vec_t *vec2, int offs1, int offs2, int nv);
static void vec_fromRand(ghost_vec_t *vec);
static void vec_fromScalar(ghost_vec_t *vec, void *val);
static void vec_fromFile(ghost_vec_t *vec, char *path, off_t offset);
static void vec_toFile(ghost_vec_t *vec, char *path, off_t offset, int);
static void         ghost_zeroVector(ghost_vec_t *vec);
//static ghost_vec_t *ghost_newVector( const int nrows, unsigned int flags );
static void         ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2);
static void         ghost_normalizeVector( ghost_vec_t *vec);
static void ghost_distributeVector(ghost_vec_t *vec, ghost_vec_t **nodeVec, ghost_comm_t *comm);
static void ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec, ghost_context_t *context);
static void         ghost_freeVector( ghost_vec_t* const vec );
static void ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm); 
static int ghost_vecEquals(ghost_vec_t *a, ghost_vec_t *b);
static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src);
static void vec_entry(ghost_vec_t *, int, void *);
static ghost_vec_t * vec_extract (ghost_vec_t * mv, int k, int n);
static ghost_vec_t * vec_view (ghost_vec_t *src, int k, int n);

ghost_vec_t *init(ghost_vtraits_t *traits)
{
	ghost_vec_t *vec = (ghost_vec_t *)allocateMemory(sizeof(ghost_vec_t),"vector");
	vec->traits = traits;

	DEBUG_LOG(1,"Initializing vector");

	vec->dotProduct = &vec_dotprod;
	vec->scale = &vec_scale;
	vec->axpy = &vec_axpy;
	vec->print = &vec_print;
	vec->fromFunc = &vec_fromFunc;
	vec->fromVec = &vec_fromVec;
	vec->fromRand = &vec_fromRand;
	vec->fromScalar = &vec_fromScalar;
	vec->fromFile = &vec_fromFile;
	vec->toFile = &vec_toFile;
	vec->zero = &ghost_zeroVector;
	vec->distribute = &ghost_distributeVector;
	vec->collect = &ghost_collectVectors;
	vec->swap = &ghost_swapVectors;
	vec->normalize = &ghost_normalizeVector;
	vec->destroy = &ghost_freeVector;
	vec->permute = &ghost_permuteVector;
	vec->equals = &ghost_vecEquals;
	vec->clone = &ghost_cloneVector;
	vec->entry = &vec_entry;
	vec->extract = &vec_extract;
	vec->view = &vec_view;
	
	vec->val = NULL;
	vec->isView = 0;

	DEBUG_LOG(1,"The vector has %d sub-vectors with %d rows and %lu bytes per entry",traits->nvecs,traits->nrows,sizeof(ghost_dt));
/*
	ghost_vidx_t i,v;

#pragma omp parallel for
	for (v=0; v<traits->nvecs; v++) {
		for (i=0; i<traits->nrows; i++) {
			VAL(vec)[v*traits->nrows+i] = 0.+I*0.;
		}
	}
*/
	return vec;
}

static ghost_vec_t * vec_view (ghost_vec_t *src, int k, int n)
{
	DEBUG_LOG(1,"Extracting %d sub-vectors starting from %d",n,k);
	ghost_vec_t *new;
	ghost_vtraits_t *newTraits = (ghost_vtraits_t *)malloc(sizeof(ghost_vtraits_t));
	newTraits->flags = src->traits->flags;
	newTraits->nrows = src->traits->nrows;
	newTraits->nvecs = n;
	newTraits->datatype = src->traits->datatype;

	new = ghost_initVector(newTraits);
	new->val = &(VAL(src)[k*src->traits->nrows]);

	new->isView = 1;
	return new;
}
	
ghost_vec_t * vec_extract (ghost_vec_t * src, int k, int n)
{
	DEBUG_LOG(1,"Extracting %d sub-vectors starting from %d",n,k);
	ghost_vec_t *new;
	ghost_vtraits_t *newTraits = (ghost_vtraits_t *)malloc(sizeof(ghost_vtraits_t));
	newTraits->flags = src->traits->flags;
	newTraits->nrows = src->traits->nrows;
	newTraits->nvecs = n;
	newTraits->datatype = src->traits->datatype;

	new = ghost_initVector(newTraits);
	new->fromVec(new,src,0,k,n);

	return new;

}

static void ghost_normalizeVector( ghost_vec_t *vec)
{
	ghost_dt s;
    vec_dotprod(vec,vec,&s);
	s = (ghost_dt)1./SQRT(s);
	vec_scale(vec,&s);

#ifdef OPENCL
	CL_uploadVector(vec);
#endif
#ifdef CUDA
	CU_uploadVector(vec);
#endif
}

static void vec_print(ghost_vec_t *vec)
{
	ghost_vidx_t i,v;
	for (i=0; i<vec->traits->nrows; i++) {
		printf("vec[%d] = ",i);
		for (v=0; v<vec->traits->nvecs; v++) {
#if GHOST_MY_DT & GHOST_BINCRS_DT_COMPLEX
			printf("%f + %f\t",REAL(VAL(vec)[v*vec->traits->nrows+i]),IMAG(VAL(vec)[v*vec->traits->nrows+i]));
#else
			printf("%f\t",VAL(vec)[v*vec->traits->nrows+i]);
#endif
		}
		printf("\n");
	}

}

static void vec_fromVec(ghost_vec_t *vec, ghost_vec_t *vec2, int offs1, int offs2, int nv)
{
	vec->val = (ghost_dt *)allocateMemory(vec->traits->nvecs*vec->traits->nrows*sizeof(ghost_dt),"vec->val");
	ghost_vidx_t i,v;
	ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

#pragma omp parallel for private(i) 
	for (v=0; v<nv; v++) {
		for (i=0; i<nr; i++) {
			VAL(vec)[(offs1+v)*vec->traits->nrows+i] = VAL(vec2)[(offs2+v)*vec2->traits->nrows+i];
		}
	}
}

static void vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
	ghost_vidx_t i;
	ghost_dt s = *(ghost_dt *)scale;
	ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

#pragma omp parallel for 
	for (i=0; i<nr; i++) {
		VAL(vec)[i] += VAL(vec2)[i] * s;
	}
}

static void vec_scale(ghost_vec_t *vec, void *scale)
{
	ghost_vidx_t i;
	ghost_dt s = *(ghost_dt *)scale;

#pragma omp parallel for 
	for (i=0; i<vec->traits->nrows; i++) {
		VAL(vec)[i] *= s;
	}


}

static void vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{
	ghost_dt sum;
	ghost_vidx_t i;
	ghost_vidx_t nr = MIN(vec->traits->nrows,vec2->traits->nrows);

#pragma omp parallel for reduction(+:sum)
	for (i=0; i<nr; i++) {
		sum += VAL(vec)[i]*VAL(vec2)[i];
	}

	*(ghost_dt *)res = sum;
}

static void vec_entry(ghost_vec_t * vec, int i, void *val)
{
	*((ghost_dt *)val) = VAL(vec)[i];
}

static void vec_fromRand(ghost_vec_t *vec)
{
	vec->val = (ghost_dt *)allocateMemory(vec->traits->nvecs*vec->traits->nrows*sizeof(ghost_dt),"vec->val");
	int i;

#pragma omp parallel for schedule(runtime)
	for (i=0; i<vec->traits->nrows; i++) {
		VAL(vec)[i] = (ghost_dt)(rand()*(ghost_dt)1./RAND_MAX + I*rand()*(ghost_dt)1./RAND_MAX);
	}

}

static void vec_fromScalar(ghost_vec_t *vec, void *val)
{
	vec->val = (ghost_dt *)allocateMemory(vec->traits->nvecs*vec->traits->nrows*sizeof(ghost_dt),"vec->val");
	int i,v;

#pragma omp parallel for schedule(runtime) private(i)
	for (v=0; v<vec->traits->nvecs; v++) {
		for (i=0; i<vec->traits->nrows; i++) {
			VAL(vec)[v*vec->traits->nrows+i] = *(ghost_dt *)val;
		}
	}
}

static void vec_toFile(ghost_vec_t *vec, char *path, off_t offset, int skipHeader)
{
	DEBUG_LOG(1,"Writing vector to file %s",path);
	int file;

	if ((file = open(path, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)) == -1){
		ABORT("Could not open vector file %s",path);
	}

	int offs;
   
	if (!skipHeader) {
		offs = 0;
		int32_t endianess = ghost_archIsBigEndian();
		int32_t version = 1;
		int32_t order = GHOST_BINVEC_ORDER_COL_FIRST;
		int32_t datatype = vec->traits->datatype;
		int64_t nrows = (int64_t)vec->traits->nrows;
		int64_t ncols = (int64_t)1;

		pwrite(file,&endianess,sizeof(endianess),offs);
		pwrite(file,&version,sizeof(version),    offs+=sizeof(endianess));
		pwrite(file,&order,sizeof(order),        offs+=sizeof(version));
		pwrite(file,&datatype,sizeof(datatype),  offs+=sizeof(order));
		pwrite(file,&nrows,sizeof(nrows),        offs+=sizeof(datatype));
		pwrite(file,&ncols,sizeof(ncols),        offs+=sizeof(nrows));
		offs += sizeof(ncols);
	} else {
		offs = 4*sizeof(int32_t)+2*sizeof(int64_t); 
	}

	pwrite(file,vec->val,sizeof(ghost_dt)*vec->traits->nrows,offs+offset*sizeof(ghost_dt));

	close(file);

}

static void vec_fromFile(ghost_vec_t *vec, char *path, off_t offset)
{
	vec->val = (ghost_dt *)allocateMemory(vec->traits->nvecs*vec->traits->nrows*sizeof(ghost_dt),"vec->val");
	DEBUG_LOG(1,"Reading vector from file %s",path);
	int file;

	if ((file = open(path, O_RDONLY)) == -1){
		ABORT("Could not open vector file %s",path);
	}

	int32_t endianess;
	int32_t version;
	int32_t order;
	int32_t datatype;
	
	int64_t nrows;
	int64_t ncols;

	int offs = 0;

	pread(file,&endianess,sizeof(endianess),offs);
	if (endianess != GHOST_BINCRS_LITTLE_ENDIAN)
		ABORT("Cannot read big endian vectors");
	
	pread(file,&version,sizeof(version),offs+=sizeof(endianess));
	if (version != 1)
		ABORT("Cannot read vector files with format != 1");
	
	pread(file,&order,sizeof(order),offs+=sizeof(version));
	// Order does not matter for vectors
	
	pread(file,&datatype,sizeof(datatype),offs+=sizeof(order));
	if (datatype != vec->traits->datatype)
		ABORT("The data types don't match!");

	pread(file,&nrows,sizeof(nrows),offs+=sizeof(datatype));
	// I will read as many rows as the vector has

	pread(file,&ncols,sizeof(ncols),offs+=sizeof(nrows));
	if (ncols != 1)
		ABORT("The number of columns has to be 1!");
	
	pread(file,vec->val,sizeof(ghost_dt)*vec->traits->nrows,offs+=sizeof(ncols)+offset*sizeof(ghost_dt));

	close(file);
	
}

static void vec_fromFunc(ghost_vec_t *vec, void (*fp)(int,int,void *))
{
	DEBUG_LOG(1,"Filling vector via function");
	vec->val = (ghost_dt *)allocateMemory(vec->traits->nvecs*vec->traits->nrows*sizeof(ghost_dt),"vec->val");
	int i,v;

#pragma omp parallel for schedule(runtime) private(i)
	for (v=0; v<vec->traits->nvecs; v++) {
		for (i=0; i<vec->traits->nrows; i++) {
			fp(i,v,&VAL(vec)[v*vec->traits->nrows+i]);
		}
	}
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
/*
static ghost_vec_t* ghost_newVector( const int nrows, unsigned int flags ) 
{
	UNUSED(nrows);
	UNUSED(flags);
	ghost_vec_t* vec = NULL;
	size_t size_val;
	int i;

	size_val = (size_t)( ghost_pad(nrows,VEC_PAD) * sizeof(ghost_dt) );
	vec = (ghost_vec_t*) allocateMemory( sizeof( ghost_vec_t ), "vec");


	VAL(vec) = (ghost_dt*) allocateMemory( size_val, "VAL(vec)");
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
}*/

static void ghost_distributeVector(ghost_vec_t *vec, ghost_vec_t **nodeVec, ghost_comm_t *comm)
{
	DEBUG_LOG(1,"Distributing vector");
#ifdef MPI
	int me = ghost_getRank();

/*	ghost_vidx_t nrows;

	MPI_safecall(MPI_Bcast(&(vec->traits->flags),1,MPI_INT,0,MPI_COMM_WORLD));

	if (vec->traits->flags & GHOST_VEC_RHS)
		nrows = comm->lnrows[me]+comm->halo_elements;
	else if (vec->traits->flags & GHOST_VEC_LHS)
		nrows = comm->lnrows[me];
	else
		ABORT("No valid type for vector (has to be one of GHOST_VEC_LHS/_RHS/_BOTH");


	DEBUG_LOG(2,"Creating local vector with %"PRvecIDX" rows",nrows);
*/
	DEBUG_LOG(2,"Scattering global vector to local vectors");
	MPI_safecall(MPI_Scatterv ( VAL(vec), (int *)comm->lnrows, (int *)comm->lfRow, ghost_mpi_dt, VAL((*nodeVec)), (int)comm->lnrows[me], ghost_mpi_dt, 0, MPI_COMM_WORLD ));
#else
	UNUSED(comm);
	/*ghost_vec_t *nodeVec = ghost_newVector( vec->traits->nrows, vec->traits->flags ); 
	int i;
	for (i=0; i<vec->traits->nrows; i++) VAL(nodeVec)[i] = VAL(vec)[i];*/
	*nodeVec = vec->clone(vec);
#endif

#ifdef OPENCL // TODO depending on flag
	CL_uploadVector(*nodeVec);
#endif
#ifdef CUDA // TODO depending on flag
	CU_uploadVector(*nodeVec);
#endif

	DEBUG_LOG(1,"Vector distributed successfully");
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
	MPI_safecall(MPI_Gatherv(VAL(vec),(int)context->communicator->lnrows[me],ghost_mpi_dt,totalVec->val,
				(int *)context->communicator->lnrows,(int *)context->communicator->lfRow,ghost_mpi_dt,0,MPI_COMM_WORLD));
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
	ghost_dt *dtmp;

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


static void ghost_freeVector( ghost_vec_t* vec ) 
{
	if( vec ) {
		if (!vec->isView) {
#ifdef CUDA_PINNEDMEM
			if (vec->traits->flags & GHOST_VEC_DEVICE)
				CU_safecall(cudaFreeHost(VAL(vec)));
#else
				free(vec->val);
#endif
		}
		//		freeMemory( (size_t)(vec->traits->nrows*sizeof(ghost_dt)), "VAL(vec)",  VAL(vec) );
#ifdef OPENCL
		if (vec->traits->flags & GHOST_VEC_DEVICE)
			CL_freeDeviceMemory( vec->CL_val_gpu );
#endif
#ifdef CUDA
		if (vec->traits->flags & GHOST_VEC_DEVICE)
			CU_freeDeviceMemory( vec->CU_val );
#endif
		free(vec);
		// TODO free traits ???
	}
}

static void ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm) 
{
	/* permutes values in vector so that i-th entry is mapped to position perm[i] */
	ghost_midx_t i;
	ghost_vidx_t len = vec->traits->nrows;
	ghost_dt* tmp;

	if (perm == NULL) {
		DEBUG_LOG(1,"Permutation vector is NULL, returning.");
		return;
	} else {
		DEBUG_LOG(1,"Permuting vector");
	}


	tmp = (ghost_dt*)allocateMemory(sizeof(ghost_dt)*len, "permute tmp");

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
		if (REAL(ABS(VAL(a)[i]-VAL(b)[i])) > tol || IMAG(ABS(VAL(a)[i]-VAL(b)[i])) > tol)
			return 0;
	}

	return 1;

}

static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src)
{
	return src->extract(src,0,src->traits->nvecs);
/*	ghost_vec_t *new;
	ghost_vtraits_t *newTraits = (ghost_vtraits_t *)malloc(sizeof(ghost_vtraits_t));
	newTraits->flags = src->traits->flags;
	newTraits->nrows = src->traits->nrows;
	newTraits->datatype = src->traits->datatype;

	new = ghost_initVector(newTraits);
	new->fromVec(new,src,0,0,1);
 
//	= ghost_newVector(src->traits->nrows, src->traits->flags);
//	memcpy(new->val, src->val, src->traits->nrows*sizeof(ghost_dt));

	return new;*/
}
