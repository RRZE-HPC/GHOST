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

//#define VAL(vec) ((ghost_dt *)(vec->val))

void (*ghost_normalizeVector_funcs[4]) (ghost_vec_t *) = 
{&s_ghost_normalizeVector, &d_ghost_normalizeVector, &c_ghost_normalizeVector, &z_ghost_normalizeVector};

void (*ghost_vec_dotprod_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*) = 
{&s_ghost_vec_dotprod, &d_ghost_vec_dotprod, &c_ghost_vec_dotprod, &z_ghost_vec_dotprod};

void (*ghost_vec_scale_funcs[4]) (ghost_vec_t *, void*) = 
{&s_ghost_vec_scale, &d_ghost_vec_scale, &c_ghost_vec_scale, &z_ghost_vec_scale};

void (*ghost_vec_axpy_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*) = 
{&s_ghost_vec_axpy, &d_ghost_vec_axpy, &c_ghost_vec_axpy, &z_ghost_vec_axpy};

void (*ghost_vec_gaxpy_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*, void*) = 
{&s_ghost_vec_gaxpy, &d_ghost_vec_gaxpy, &c_ghost_vec_gaxpy, &z_ghost_vec_gaxpy};

void (*ghost_vec_fromRand_funcs[4]) (ghost_vec_t *) = 
{&s_ghost_vec_fromRand, &d_ghost_vec_fromRand, &c_ghost_vec_fromRand, &z_ghost_vec_fromRand};

int (*ghost_vecEquals_funcs[4]) (ghost_vec_t *, ghost_vec_t *) = 
{&s_ghost_vecEquals, &d_ghost_vecEquals, &c_ghost_vecEquals, &z_ghost_vecEquals};


static void vec_print(ghost_vec_t *vec);
static void vec_scale(ghost_vec_t *vec, void *scale);
static void vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale);
static void vec_gaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b);
static void vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res);
static void vec_fromFunc(ghost_vec_t *vec, void (*fp)(int,int,void *));
static void vec_fromVec(ghost_vec_t *vec, ghost_vec_t *vec2, ghost_vidx_t roffs, ghost_vidx_t coffs);
static void vec_fromRand(ghost_vec_t *vec);
static void vec_fromScalar(ghost_vec_t *vec, void *val);
static void vec_fromFile(ghost_vec_t *vec, char *path, off_t offset);
static void vec_toFile(ghost_vec_t *vec, char *path, off_t offset, int);
static void         ghost_zeroVector(ghost_vec_t *vec);
static void         ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2);
static void         ghost_normalizeVector( ghost_vec_t *vec);
static void ghost_distributeVector(ghost_vec_t *vec, ghost_vec_t **nodeVec, ghost_comm_t *comm);
static void ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec, ghost_mat_t *mat); 
static void         ghost_freeVector( ghost_vec_t* const vec );
static void ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm); 
static int ghost_vecEquals(ghost_vec_t *a, ghost_vec_t *b);
static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src);
static void vec_entry(ghost_vec_t *, int, void *);
static ghost_vec_t * vec_extract (ghost_vec_t * src, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs);
static ghost_vec_t * vec_view (ghost_vec_t *src, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs);
static void vec_viewPlain (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda);
#ifdef CUDA
static void vec_CUupload (ghost_vec_t *);
static void vec_CUdownload (ghost_vec_t *);
#endif
#ifdef OPENCL
static void vec_CLupload (ghost_vec_t *);
static void vec_CLdownload (ghost_vec_t *);
#endif

ghost_vec_t *ghost_createVector(ghost_context_t *ctx, ghost_vtraits_t *traits)
{
	ghost_vec_t *vec = (ghost_vec_t *)ghost_malloc(sizeof(ghost_vec_t));
	vec->context = ctx;
	vec->traits = traits;

	DEBUG_LOG(1,"The vector has %"PRvecIDX" sub-vectors with %"PRvecIDX" rows and %lu bytes per entry",traits->nvecs,traits->nrows,ghost_sizeofDataType(vec->traits->datatype));
	DEBUG_LOG(1,"Initializing vector");

	vec->dotProduct = &vec_dotprod;
	vec->scale = &vec_scale;
	vec->axpy = &vec_axpy;
	vec->gaxpy = &vec_gaxpy;
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
	vec->viewPlain = &vec_viewPlain;

#ifdef CUDA
	vec->CUupload = &vec_CUupload;
	vec->CUdownload = &vec_CUdownload;
#endif
#ifdef OPENCL
	vec->CLupload = &vec_CLupload;
	vec->CLdownload = &vec_CLdownload;
#endif

	vec->val = NULL;
	vec->isView = 0;

	return vec;
	}

#ifdef CUDA
static void vec_CUupload (ghost_vec_t *vec)
{
	CU_copyHostToDevice(vec->CU_val,vec->val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}

static void vec_CUdownload (ghost_vec_t *vec)
{
	CU_copyDeviceToHost(vec->val,vec->CU_val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}
#endif

#ifdef OPENCL
static void vec_CLupload( ghost_vec_t *vec )
{
	CL_copyHostToDevice(vec->CL_val_gpu,vec->val,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}

static void vec_CLdownload( ghost_vec_t *vec )
{
	CL_copyDeviceToHost(vec->val,vec->CL_val_gpu,vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
}
#endif

static ghost_vec_t * vec_view (ghost_vec_t *src, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs)
{
	DEBUG_LOG(1,"Viewing a %"PRvecIDX"x%"PRvecIDX" dense matrix with offset %"PRvecIDX"x%"PRvecIDX,nr,nc,roffs,coffs);
	ghost_vec_t *new;
	ghost_vtraits_t *newTraits = ghost_cloneVtraits(src->traits);
	newTraits->nrows = nr;
	newTraits->nvecs = nc;

	new = ghost_createVector(src->context,newTraits);
	new->val = &VAL(src,src->traits->nrowspadded*coffs+roffs);

	new->isView = 1;
	return new;
}

static void vec_viewPlain (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda)
{
	DEBUG_LOG(1,"Viewing a %"PRvecIDX"x%"PRvecIDX" dense matrix from plain data with offset %"PRvecIDX"x%"PRvecIDX,nr,nc,roffs,coffs);

	vec->val = &((char *)data)[(lda*coffs+roffs)*ghost_sizeofDataType(vec->traits->datatype)];
	vec->isView = 1;
}

ghost_vec_t * vec_extract (ghost_vec_t * src, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs)
{
	DEBUG_LOG(1,"Extracting a %"PRvecIDX"x%"PRvecIDX" dense matrix with offset %"PRvecIDX"x%"PRvecIDX,nr,nc,roffs,coffs);
	ghost_vec_t *new;
	ghost_vtraits_t *newTraits = ghost_cloneVtraits(src->traits);
	newTraits->nrows = nr;
	newTraits->nvecs = nc;

	new = ghost_createVector(src->context,newTraits);
	new->fromVec(new,src,roffs,coffs);

	return new;

}

static void ghost_normalizeVector( ghost_vec_t *vec)
{
	ghost_normalizeVector_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec);
}

static void vec_print(ghost_vec_t *vec)
{
	char prefix[16];
#ifdef GHOST_MPI
	if (vec->context->communicator != NULL) {
		int rank = ghost_getRank(vec->context->mpicomm);
		int ndigits = floor(log10(abs(rank))) + 1;
		snprintf(prefix,4+nDigits,"PE%d: ",rank);
	} else {
		snprintf(prefix,1,"\0");
	}
#else
	snprintf(prefix,1,"\0");
#endif


	ghost_vidx_t i,v;
	for (i=0; i<vec->traits->nrows; i++) {
		for (v=0; v<vec->traits->nvecs; v++) {
			if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
				if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
					printf("%svec[%"PRvecIDX"][%"PRvecIDX"] = %f + %fi\t",
							prefix,v,i,
							crealf(((complex float *)(vec->val))[v*vec->traits->nrows+i]),
							cimagf(((complex float *)(vec->val))[v*vec->traits->nrows+i]));
				} else {
					printf("%svec[%"PRvecIDX"][%"PRvecIDX"] = %f + %fi\t",
							prefix,v,i,
							creal(((complex double *)(vec->val))[v*vec->traits->nrows+i]),
							cimag(((complex double *)(vec->val))[v*vec->traits->nrows+i]));
				}
			} else {
				if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
					printf("%s(s) v[%"PRvecIDX"][%"PRvecIDX"] = %f\t",prefix,v,i,((float *)(vec->val))[v*vec->traits->nrowspadded+i]);
				} else {
					printf("%s(d) v[%"PRvecIDX"][%"PRvecIDX"] = %f\t",prefix,v,i,((double *)(vec->val))[v*vec->traits->nrowspadded+i]);
				}
			}
		}
		printf("\n");
	}

}

void getNrowsFromContext(ghost_vec_t *vec)
{
	DEBUG_LOG(1,"Computing the number of vector rows from the context");

	if (vec->context != NULL) {
	if (vec->traits->nrows == 0) {
		DEBUG_LOG(2,"nrows for vector not given. determining it from the context");
		if (vec->traits->flags & GHOST_VEC_DUMMY) {
			vec->traits->nrows = 0;
		} else if ((vec->context->flags & GHOST_CONTEXT_GLOBAL) || (vec->traits->flags & GHOST_VEC_GLOBAL))
		{
			if (vec->traits->flags & GHOST_VEC_LHS) {
				vec->traits->nrows = vec->context->gnrows;
			} else if (vec->traits->flags & GHOST_VEC_RHS) {
				vec->traits->nrows = vec->context->gncols;
			}
		} 
		else 
		{
			vec->traits->nrows = vec->context->communicator->lnrows[ghost_getRank(vec->context->mpicomm)];
		}
	}
	if (vec->traits->nrowshalo == 0) {
		DEBUG_LOG(2,"nrowshalo for vector not given. determining it from the context");
		if (vec->traits->flags & GHOST_VEC_DUMMY) {
			vec->traits->nrowshalo = 0;
		} else if ((vec->context->flags & GHOST_CONTEXT_GLOBAL) || (vec->traits->flags & GHOST_VEC_GLOBAL))
		{
			vec->traits->nrowshalo = vec->traits->nrows;
		} 
		else 
		{
			if (!(vec->traits->flags & GHOST_VEC_GLOBAL) && vec->traits->flags & GHOST_VEC_RHS)
				vec->traits->nrowshalo = vec->traits->nrows+vec->context->communicator->halo_elements;
			else
				vec->traits->nrowshalo = vec->traits->nrows;
		}	
	}
	} else {
		WARNING_LOG("The vector's context is NULL.");
	}


	if (vec->traits->nrowspadded == 0) {
		DEBUG_LOG(2,"nrowspadded for vector not given. determining it from the context");
		vec->traits->nrowspadded = ghost_pad(MAX(vec->traits->nrowshalo,vec->traits->nrows),GHOST_PAD_MAX); // TODO needed?
	}
	DEBUG_LOG(1,"The vector has %"PRvecIDX" w/ %"PRvecIDX" halo elements (padded: %"PRvecIDX") rows",
			vec->traits->nrows,vec->traits->nrowshalo-vec->traits->nrows,vec->traits->nrowspadded);
}


static void vec_fromVec(ghost_vec_t *vec, ghost_vec_t *vec2, ghost_vidx_t roffs, ghost_vidx_t coffs)
{
	DEBUG_LOG(1,"Initializing vector from vector w/ offset %"PRvecIDX"x%"PRvecIDX,roffs,coffs);
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
	vec->val = ghost_malloc_align(vec->traits->nvecs*vec->traits->nrowspadded*sizeofdt,GHOST_DATA_ALIGNMENT);
	ghost_vidx_t i,v;

#pragma omp parallel for private(i) 
	for (v=0; v<vec->traits->nvecs; v++) {
		for (i=0; i<vec->traits->nrows; i++) {
			memcpy(&VAL(vec,v*vec->traits->nrowspadded+i),&VAL(vec2,(coffs+v)*vec2->traits->nrowspadded+roffs+i),sizeofdt);
		}
	}
}

static void vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
	ghost_vec_axpy_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,scale);
}

static void vec_gaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b)
{
	ghost_vec_gaxpy_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,scale,b);
}

static void vec_scale(ghost_vec_t *vec, void *scale)
{
	ghost_vec_scale_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,scale);
}

static void vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{
	ghost_vec_dotprod_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,res);
}

static void vec_entry(ghost_vec_t * vec, int i, void *val)
{
	memcpy(val,&VAL(vec,i),ghost_sizeofDataType(vec->traits->datatype));
}

static void vec_fromRand(ghost_vec_t *vec)
{
	ghost_vec_fromRand_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec);
}

static void vec_fromScalar(ghost_vec_t *vec, void *val)
{
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
	getNrowsFromContext(vec);

	DEBUG_LOG(1,"Initializing vector from scalar value with %"PRvecIDX" rows",vec->traits->nrows);
	vec->val = ghost_malloc_align(vec->traits->nvecs*vec->traits->nrowspadded*sizeofdt,GHOST_DATA_ALIGNMENT);
	int i,v;

	if (vec->traits->nvecs > 1) {
#pragma omp parallel for schedule(runtime) private(i)
		for (v=0; v<vec->traits->nvecs; v++) {
			for (i=0; i<vec->traits->nrows; i++) {
				memcpy(&VAL(vec,v*vec->traits->nrowspadded+i),val,sizeofdt);
			}
		}
	} else {
#pragma omp parallel for schedule(runtime)
		for (i=0; i<vec->traits->nrows; i++) {
			memcpy(&VAL(vec,i),val,sizeofdt);
		}
	}

	if (!(vec->traits->flags & GHOST_VEC_HOST)) {
#ifdef CUDA
#ifdef CUDA_PINNEDMEM
		CU_safecall(cudaHostGetDevicePointer((void **)&vec->CU_val,vec->val,0));
#else
		vec->CU_val = CU_allocDeviceMemory(vec->traits->nvecs*vec->traits->nrows*sizeofdt);
#endif
		vec->CUupload(vec);
#endif
#ifdef OPENCL
		vec->CL_val_gpu = CL_allocDeviceMemoryMapped(vec->traits->nvecs*vec->traits->nrows*sizeofdt,vec->val,CL_MEM_READ_WRITE );
		vec->CLupload(vec);
#endif
	}	
}

static void vec_toFile(ghost_vec_t *vec, char *path, off_t offset, int skipHeader)
{
	DEBUG_LOG(1,"Writing vector to file %s",path);
	int file;

	if ((file = open(path, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)) == -1){
		ABORT("Could not open vector file %s",path);
	}

	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
	int offs;

	if (!skipHeader) {
		offs = 0;
		int32_t endianess = ghost_archIsBigEndian();
		int32_t version = 1;
		int32_t order = GHOST_BINVEC_ORDER_COL_FIRST;
		int32_t datatype = vec->traits->datatype;
		int64_t nrows = (int64_t)vec->traits->nrows;
		int64_t ncols = (int64_t)vec->traits->nvecs;

		//TODO pwrite -> fwrite
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

	int v;
	for (v=0; v<vec->traits->nvecs; v++) {
		pwrite(file,((char *)(vec->val))+v*sizeofdt*vec->traits->nrowspadded,sizeofdt*vec->traits->nrows,offs+offset*sizeofdt+v*sizeofdt*vec->traits->nrows);
	}

	close(file);

}

static void vec_fromFile(ghost_vec_t *vec, char *path, off_t offset)
{
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

	getNrowsFromContext(vec);


	vec->val = ghost_malloc_align(vec->traits->nvecs*vec->traits->nrowspadded*sizeofdt,GHOST_DATA_ALIGNMENT);
	DEBUG_LOG(1,"Reading vector from file %s",path);
	FILE *filed;
	size_t ret;

	if ((filed = fopen64(path, "r")) == NULL){
		ABORT("Could not vector file %s",path);
	}

	int32_t endianess;
	int32_t version;
	int32_t order;
	int32_t datatype;

	int64_t nrows;
	int64_t ncols;


	if ((ret = fread(&endianess, sizeof(endianess), 1,filed)) != 1)
		ABORT("fread failed: %lu",ret);

	if (endianess != GHOST_BINCRS_LITTLE_ENDIAN)
		ABORT("Cannot read big endian vectors");

	if ((ret = fread(&version, sizeof(version), 1,filed)) != 1)
		ABORT("fread failed");
	
	if (version != 1)
		ABORT("Cannot read vector files with format != 1");

	if ((ret = fread(&order, sizeof(order), 1,filed)) != 1)
		ABORT("fread failed");
	// Order does not matter for vectors

	if ((ret = fread(&datatype, sizeof(datatype), 1,filed)) != 1)
		ABORT("fread failed");
	if (datatype != vec->traits->datatype)
		ABORT("The data types don't match! Cast-while-read is not yet implemented for vectors.");

	if ((ret = fread(&nrows, sizeof(nrows), 1,filed)) != 1)
		ABORT("fread failed");
	// I will read as many rows as the vector has

	if ((ret = fread(&ncols, sizeof(ncols), 1,filed)) != 1)
		ABORT("fread failed");
	//if (ncols != 1)
	//	ABORT("The number of columns has to be 1!");

	int v;
	for (v=0; v<vec->traits->nvecs; v++) {
		if (fseeko(filed,offset*sizeofdt,SEEK_CUR))
			ABORT("Seek failed");

		if ((ret = fread(((char *)(vec->val))+v*sizeofdt*vec->traits->nrowspadded, sizeofdt, vec->traits->nrows,filed)) != vec->traits->nrows)
			ABORT("fread failed");

//		pread(file,((char *)(vec->val))+v*sizeofdt*vec->traits->nrowspadded,sizeofdt*vec->traits->nrows,offs+(offset+v*vec->traits->nrows)*vec->traits->nrows);
	}

	fclose(filed);

	if (!(vec->traits->flags & GHOST_VEC_HOST)) {
#ifdef CUDA
#ifdef CUDA_PINNEDMEM
		CU_safecall(cudaHostGetDevicePointer((void **)&vec->CU_val,vec->val,0));
#else
		vec->CU_val = CU_allocDeviceMemory(vec->traits->nvecs*vec->traits->nrows*sizeofdt);
#endif
		vec->CUupload(vec);
#endif
#ifdef OPENCL
		vec->CL_val_gpu = CL_allocDeviceMemoryMapped(vec->traits->nvecs*vec->traits->nrows*sizeofdt,vec->val,CL_MEM_READ_WRITE );
		vec->CLupload(vec);
#endif
	}	

}

static void vec_fromFunc(ghost_vec_t *vec, void (*fp)(int,int,void *))
{
	DEBUG_LOG(1,"Filling vector via function");
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
	getNrowsFromContext(vec);

	vec->val = ghost_malloc_align(vec->traits->nvecs*vec->traits->nrowspadded*sizeofdt,GHOST_DATA_ALIGNMENT);
	int i,v;

	if (vec->traits->nvecs > 1) {
#pragma omp parallel for schedule(runtime) private(i)
		for (v=0; v<vec->traits->nvecs; v++) {
			for (i=0; i<vec->traits->nrows; i++) {
				fp(i,v,&VAL(vec,v*vec->traits->nrowspadded+i));
			}
		}
	} else {
#pragma omp parallel for schedule(runtime)
		for (i=0; i<vec->traits->nrows; i++) {
			fp(i,v,((char*)(vec->val))+sizeofdt*i);
		}
	}


	if (!(vec->traits->flags & GHOST_VEC_HOST)) {
#ifdef CUDA
#ifdef CUDA_PINNEDMEM
		CU_safecall(cudaHostGetDevicePointer((void **)&vec->CU_val,vec->val,0));
#else
		vec->CU_val = CU_allocDeviceMemory(vec->traits->nvecs*vec->traits->nrows*sizeofdt);
#endif
		vec->CUupload(vec);
#endif
#ifdef OPENCL
		vec->CL_val_gpu = CL_allocDeviceMemoryMapped(vec->traits->nvecs*vec->traits->nrows*sizeofdt,vec->val,CL_MEM_READ_WRITE );
		vec->CLupload(vec);
#endif
	}	


}

static void ghost_zeroVector(ghost_vec_t *vec) 
{
	DEBUG_LOG(1,"Zeroing vector");
	memset(vec->val,0,vec->traits->nrowspadded*vec->traits->nvecs*ghost_sizeofDataType(vec->traits->datatype));

#ifdef OPENCL
	vec->CLupload(vec);
#endif
#ifdef CUDA
	vec->CUupload(vec);
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
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
#ifdef GHOST_MPI
	int me = ghost_getRank((*nodeVec)->context->mpicomm);
	DEBUG_LOG(2,"Scattering global vector to local vectors");

	MPI_Datatype mpidt;


	if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
		if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
			mpidt = GHOST_MPI_DT_C;
		} else {
			mpidt = GHOST_MPI_DT_Z;
		}
	} else {
		if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
			mpidt = MPI_FLOAT;
		} else {
			mpidt = MPI_DOUBLE;
		}
	}

	int nprocs = ghost_getNumberOfRanks((*nodeVec)->context->mpicomm);
	int i;

	MPI_Request req[2*(nprocs-1)];
	MPI_Status stat[2*(nprocs-1)];
	int msgcount = 0;

	for (i=0;i<2*(nprocs-1);i++) 
		req[i] = MPI_REQUEST_NULL;

	if (ghost_getRank((*nodeVec)->context->mpicomm) != 0) {
		MPI_safecall(MPI_Irecv((*nodeVec)->val,comm->lnrows[me],mpidt,0,me,(*nodeVec)->context->mpicomm,&req[msgcount]));
		msgcount++;
	} else {
		memcpy((*nodeVec)->val,vec->val,sizeofdt*comm->lnrows[0]);
		for (i=1;i<nprocs;i++) {
			MPI_safecall(MPI_Isend(((char *)(vec->val))+sizeofdt*comm->lfRow[i],comm->lnrows[i],mpidt,i,i,(*nodeVec)->context->mpicomm,&req[msgcount]));
			msgcount++;
		}
	}
	MPI_safecall(MPI_Waitall(msgcount,req,stat));
#else
	UNUSED(comm);
	memcpy((*nodeVec)->val,vec->val,vec->traits->nrowspadded*sizeofdt);
//	*nodeVec = vec->clone(vec);
#endif

#ifdef OPENCL
	if (!((*nodeVec)->traits->flags & GHOST_VEC_HOST))
		(*nodeVec)->CLupload(*nodeVec);
#endif
#ifdef CUDA
	if (!((*nodeVec)->traits->flags & GHOST_VEC_HOST))
		(*nodeVec)->CUupload(*nodeVec);
#endif

	DEBUG_LOG(1,"Vector distributed successfully");
}

static void ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec, ghost_mat_t *mat) 
{
#ifdef GHOST_MPI
	MPI_Datatype mpidt;

	if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
		if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
			mpidt = GHOST_MPI_DT_C;
		} else {
			mpidt = GHOST_MPI_DT_Z;
		}
	} else {
		if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
			mpidt = MPI_FLOAT;
		} else {
			mpidt = MPI_DOUBLE;
		}
	}
	int me = ghost_getRank(vec->context->mpicomm);
	if (mat != NULL)
		vec->permute(vec,mat->invRowPerm); 

	int nprocs = ghost_getNumberOfRanks(vec->context->mpicomm);
	int i;
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

	ghost_comm_t *comm = vec->context->communicator;
	MPI_Request req[2*(nprocs-1)];
	MPI_Status stat[2*(nprocs-1)];
	int msgcount = 0;

	for (i=0;i<2*(nprocs-1);i++) 
		req[i] = MPI_REQUEST_NULL;

	if (ghost_getRank(vec->context->mpicomm) != 0) {
		MPI_safecall(MPI_Isend(vec->val,comm->lnrows[me],mpidt,0,me,vec->context->mpicomm,&req[msgcount]));
		msgcount++;
	} else {
		memcpy(totalVec->val,vec->val,sizeofdt*comm->lnrows[0]);
		for (i=1;i<nprocs;i++) {
			MPI_safecall(MPI_Irecv(((char *)(totalVec->val))+sizeofdt*comm->lfRow[i],comm->lnrows[i],mpidt,i,i,vec->context->mpicomm,&req[msgcount]));
			msgcount++;
		}
	}
	MPI_safecall(MPI_Waitall(msgcount,req,stat));
#else
	if (mat != NULL)
		vec->permute(vec,mat->invRowPerm); 
	memcpy(totalVec->val,vec->val,totalVec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
#endif
}

static void ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2) 
{
	char *dtmp;

	dtmp = v1->val;
	v1->val = v2->val;
	v2->val = dtmp;
#ifdef OPENCL
	cl_mem tmp;
	tmp = v1->CL_val_gpu;
	v1->CL_val_gpu = v2->CL_val_gpu;
	v2->CL_val_gpu = tmp;
#endif
#ifdef CUDA
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
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
	ghost_midx_t i;
	ghost_vidx_t len = vec->traits->nrows;
	char* tmp;

	if (perm == NULL) {
		DEBUG_LOG(1,"Permutation vector is NULL, returning.");
		return;
	} else {
		DEBUG_LOG(1,"Permuting vector");
	}


	tmp = ghost_malloc(sizeofdt*len);

	for(i = 0; i < len; ++i) {
		if( perm[i] >= len ) {
			ABORT("Permutation index out of bounds: %"PRmatIDX" > %"PRmatIDX,perm[i],len);
		}

		memcpy(&tmp[sizeofdt*perm[i]],&VAL(vec,i),sizeofdt);
	}
	for(i=0; i < len; ++i) {
		memcpy(&VAL(vec,i),&tmp[sizeofdt*i],sizeofdt);
	}

	free(tmp);
}

static int ghost_vecEquals(ghost_vec_t *a, ghost_vec_t *b)
{
	return ghost_vecEquals_funcs[ghost_dataTypeIdx(a->traits->datatype)](a,b);
}

static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src)
{
	return src->extract(src,src->traits->nrows,src->traits->nvecs,0,0);
}
