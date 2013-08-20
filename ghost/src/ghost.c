#define _GNU_SOURCE

#include "ghost.h"
#include "ghost_util.h"
#include "ghost_mat.h"
#include "ghost_vec.h"
#include "ghost_taskq.h"
#include "cpuid.h"
#include "sell.h"
#include "crs.h"

#ifdef GHOST_MPI
#include <mpi.h>
#include "ghost_mpi_util.h"
#endif

#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>
#include <dlfcn.h>
#include <dirent.h>
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <string.h>
#include <omp.h>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

#ifdef CUDA
#include <cuda_runtime.h>
#endif
#ifdef BLAS_MKL
#include <mkl.h>
#endif
#include "ghost_blas_mangle.h"

//static int options;
#ifdef GHOST_MPI
static int MPIwasInitialized;
MPI_Datatype GHOST_MPI_DT_C;
MPI_Op GHOST_MPI_OP_SUM_C;
MPI_Datatype GHOST_MPI_DT_Z;
MPI_Op GHOST_MPI_OP_SUM_Z;
#endif


#ifdef GHOST_MPI
#ifdef GHOST_VEC_COMPLEX
typedef struct 
{
	ghost_vdat_el_t x;
	ghost_vdat_el_t y;
} 
MPI_vComplex;

static void MPI_vComplAdd(MPI_vComplex *invec, MPI_vComplex *inoutvec, int *len)
{
	int i;
	MPI_vComplex c;

	for (i=0; i<*len; i++, invec++, inoutvec++){
		c.x = invec->x + inoutvec->x;
		c.y = invec->y + inoutvec->y;
		*inoutvec = c;
	}
}
#endif

#ifdef GHOST_MAT_COMPLEX

typedef struct 
{
	ghost_mdat_el_t x;
	ghost_mdat_el_t y;
} 
MPI_mComplex;

static void MPI_mComplAdd(MPI_mComplex *invec, MPI_mComplex *inoutvec, int *len)
{
	int i;
	MPI_mComplex c;

	for (i=0; i<*len; i++, invec++, inoutvec++){
		c.x = invec->x + inoutvec->x;
		c.y = invec->y + inoutvec->y;
		*inoutvec = c;
	}
}
#endif
#endif

#ifdef GHOST_MPI
typedef struct 
{
	float x;
	float y;
} 
MPI_c;

static void MPI_add_c(MPI_c *invec, MPI_c *inoutvec, int *len)
{
	int i;
	MPI_c c;

	for (i=0; i<*len; i++, invec++, inoutvec++){
		c.x = invec->x + inoutvec->x;
		c.y = invec->y + inoutvec->y;
		*inoutvec = c;
	}
}
typedef struct 
{
	double x;
	double y;
} 
MPI_z;

static void MPI_add_z(MPI_z *invec, MPI_z *inoutvec, int *len)
{
	int i;
	MPI_z c;

	for (i=0; i<*len; i++, invec++, inoutvec++){
		c.x = invec->x + inoutvec->x;
		c.y = invec->y + inoutvec->y;
		*inoutvec = c;
	}
}
#endif

int ghost_init(int argc, char **argv)
{
	int me;

#ifdef GHOST_MPI
	int req, prov;

#ifdef GHOST_OPENMP
	req = MPI_THREAD_MULTIPLE; // TODO not if not all kernels configured
#else
	req = MPI_THREAD_SINGLE;
#endif

	MPI_safecall(MPI_Initialized(&MPIwasInitialized));
	if (!MPIwasInitialized) {
		MPI_safecall(MPI_Init_thread(&argc, &argv, req, &prov ));

		if (req != prov) {
			WARNING_LOG("Required MPI threading level (%d) is not "
					"provided (%d)!",req,prov);
		}
	}
	me = ghost_getRank(MPI_COMM_WORLD);

	setupSingleNodeComm();
	MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&GHOST_MPI_DT_C));
	MPI_safecall(MPI_Type_commit(&GHOST_MPI_DT_C));
	MPI_safecall(MPI_Op_create((MPI_User_function *)&MPI_add_c,1,&GHOST_MPI_OP_SUM_C));

	MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&GHOST_MPI_DT_Z));
	MPI_safecall(MPI_Type_commit(&GHOST_MPI_DT_Z));
	MPI_safecall(MPI_Op_create((MPI_User_function *)&MPI_add_z,1,&GHOST_MPI_OP_SUM_Z));
	/*
#ifdef GHOST_MAT_COMPLEX
if (GHOST_MY_MDATATYPE & GHOST_BINCRS_DT_COMPLEX) {
if (GHOST_MY_MDATATYPE & GHOST_BINCRS_DT_FLOAT) {
MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&ghost_mpi_dt_mdat));
} else {
MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&ghost_mpi_dt_mdat));
}
MPI_safecall(MPI_Type_commit(&ghost_mpi_dt_mdat));
MPI_safecall(MPI_Op_create((MPI_User_function *)&MPI_mComplAdd,1,&ghost_mpi_sum_mdat));
} 
#endif
#ifdef GHOST_VEC_COMPLEX
if (GHOST_MY_VDATATYPE & GHOST_BINCRS_DT_COMPLEX) {
if (GHOST_MY_VDATATYPE & GHOST_BINCRS_DT_FLOAT) {
MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&ghost_mpi_dt_vdat));
} else {
MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&ghost_mpi_dt_vdat));
}
MPI_safecall(MPI_Type_commit(&ghost_mpi_dt_vdat));
MPI_safecall(MPI_Op_create((MPI_User_function *)&MPI_vComplAdd,1,&ghost_mpi_sum_vdat));
} 
#endif*/

#else // ifdef GHOST_MPI
	UNUSED(argc);
	UNUSED(argv);
	me = 0;

#endif // ifdef GHOST_MPI

	/*if (ghostOptions & GHOST_OPTION_PIN || ghostOptions & GHOST_OPTION_PIN_SMT) {
	  } else {
#pragma omp parallel
{
DEBUG_LOG(2,"Thread %d is running on core %d",omp_get_thread_num(),ghost_getCore());
}
}*/

#ifdef LIKWID_PERFMON
	LIKWID_MARKER_INIT;
#pragma omp parallel
	LIKWID_MARKER_THREADINIT;

#endif

#ifdef OPENCL
	CL_init();
#endif
#ifdef CUDA
	CU_init();
#endif

	ghost_cpuid_init();
	ghost_thpool_init(ghost_getNumberOfPhysicalCores());
	ghost_taskq_init(ghost_cpuid_topology.numSockets);

	//	options = ghostOptions;

	//ghost_setCore(ghost_getCore());
	//threadpool = (ghost_threadstate_t *)ghost_malloc(sizeof(ghost_threadstate_t)*ghost_getNumberOfHwThreads());
	//threadpool[ghost_getCore()].state = GHOST_THREAD_MGMT;
	//threadpool[ghost_getCore()].desc = "main";
	//for (int i=1; i<ghost_getNumberOfHwThreads(); i++) {
	//		threadpool[i].state = GHOST_THREAD_HALTED;
	//	}


	return me;
	}

void ghost_finish()
{

	ghost_cpuid_finish();
	ghost_taskq_finish();
	ghost_thpool_finish();

#ifdef LIKWID_PERFMON
	LIKWID_MARKER_CLOSE;
#endif

#ifdef OPENCL
	CL_finish();
#endif

#ifdef GHOST_MPI
	MPI_safecall(MPI_Type_free(&GHOST_MPI_DT_C));
	MPI_safecall(MPI_Type_free(&GHOST_MPI_DT_Z));
	if (!MPIwasInitialized) {
		MPI_Finalize();
	}
#endif

}

ghost_mat_t *ghost_createMatrix(ghost_context_t *context, ghost_mtraits_t *traits, int nTraits)
{
	ghost_mat_t *mat;
	UNUSED(nTraits);

	mat = ghost_initMatrix(traits);
	mat->context = context;
	/*	mat->fromBin(mat,matrixPath,context,options);

		if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
		mat->split(mat,options,context,traits);
		} else {
		mat->localPart = NULL;
		mat->remotePart = NULL;
		context->communicator = NULL;
		}*/


	return mat;
}

ghost_context_t *ghost_createContext(int64_t gnrows, int64_t gncols, int context_flags, char *matrixPath, MPI_Comm comm) 
{
	DEBUG_LOG(1,"Creating context with dimension %ldx%ld",gnrows,gncols);
	ghost_context_t *context;
	int i;

	context = (ghost_context_t *)ghost_malloc(sizeof(ghost_context_t));
	context->flags = context_flags;

	if ((gnrows == GHOST_GET_DIM_FROM_MATRIX) || (gncols == GHOST_GET_DIM_FROM_MATRIX)) {
		ghost_matfile_header_t fileheader;
		ghost_readMatFileHeader(matrixPath,&fileheader);
		if (gnrows == GHOST_GET_DIM_FROM_MATRIX)
			context->gnrows = (ghost_midx_t)fileheader.nrows;
		if (gncols == GHOST_GET_DIM_FROM_MATRIX)
			context->gncols = (ghost_midx_t)fileheader.ncols;

	} else {
		context->gnrows = (ghost_midx_t)gnrows;
		context->gncols = (ghost_midx_t)gncols;

	}

#ifdef GHOST_MPI
	if (!(context->flags & GHOST_CONTEXT_DISTRIBUTED) && !(context->flags & GHOST_CONTEXT_GLOBAL)) {
		DEBUG_LOG(1,"Context is set to be distributed");
		context->flags |= GHOST_CONTEXT_DISTRIBUTED;
	}
#else
	if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
		ABORT("Creating a distributed matrix without MPI is not possible");
} else if (!(context->flags & GHOST_CONTEXT_GLOBAL)) {
		DEBUG_LOG(1,"Context is set to be global");
		context->flags |= GHOST_CONTEXT_GLOBAL;
	}
#endif

	context->solvers = (ghost_solver_t *)ghost_malloc(sizeof(ghost_solver_t)*GHOST_NUM_MODES);
	for (i=0; i<GHOST_NUM_MODES; i++) context->solvers[i] = NULL;
#ifdef GHOST_MPI
	context->solvers[GHOST_SPMVM_MODE_VECTORMODE_IDX] = &hybrid_kernel_I;
	context->solvers[GHOST_SPMVM_MODE_GOODFAITH_IDX] = &hybrid_kernel_II;
	context->solvers[GHOST_SPMVM_MODE_TASKMODE_IDX] = &hybrid_kernel_III;
#else
	context->solvers[GHOST_SPMVM_MODE_NOMPI_IDX] = &ghost_solver_nompi;
#endif

#ifdef GHOST_MPI
	if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
		context->communicator = (ghost_comm_t*) ghost_malloc( sizeof(ghost_comm_t));

		context->communicator->mpicomm = comm;

		context->communicator->halo_elements = 0;
		
		int nprocs = ghost_getNumberOfRanks(context->communicator->mpicomm);
		
		context->communicator->lnEnts   = (ghost_mnnz_t*)       ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
		context->communicator->lfEnt    = (ghost_mnnz_t*)       ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
		context->communicator->lnrows   = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 
		context->communicator->lfRow    = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 

		if (context->flags & GHOST_CONTEXT_WORKDIST_NZE)
		{ // read rpt and fill lfrow, lnrows, lfent, lnents
			ghost_midx_t *rpt = (ghost_midx_t *)ghost_malloc(sizeof(ghost_midx_t)*(context->gnrows+1));
			ghost_mnnz_t gnnz;

			if (ghost_getRank(context->communicator->mpicomm) == 0) {
				int file;

				if ((file = open(matrixPath, O_RDONLY)) == -1){
					ABORT("Could not open binary CRS file %s",matrixPath);
				}
#ifdef LONGIDX
				pread(file,&rpt[0], GHOST_BINCRS_SIZE_RPT_EL*(context->gnrows+1), GHOST_BINCRS_SIZE_HEADER);
#else // casting
				DEBUG_LOG(1,"Casting from 64 bit to 32 bit row pointers");
				int64_t *tmp = (int64_t *)ghost_malloc((context->gnrows+1)*8);
				pread(file,tmp, GHOST_BINCRS_SIZE_COL_EL*(context->gnrows+1), GHOST_BINCRS_SIZE_HEADER );
				for( i = 0; i < context->gnrows+1; i++ ) {
					rpt[i] = (ghost_midx_t)(tmp[i]);
				}
				// TODO little/big endian
#endif
				context->rpt = rpt;
				gnnz = rpt[context->gnrows];
				ghost_mnnz_t target_nnz;
				target_nnz = (gnnz/nprocs)+1; /* sonst bleiben welche uebrig! */

				context->communicator->lfRow[0]  = 0;
				context->communicator->lfEnt[0] = 0;
				int j = 1;

				for (i=0;i<context->gnrows;i++){
					if (rpt[i] >= j*target_nnz){
						context->communicator->lfRow[j] = i;
						context->communicator->lfEnt[j] = rpt[i];
						j = j+1;
					}
				}
				for (i=0; i<nprocs-1; i++){
					context->communicator->lnrows[i] = context->communicator->lfRow[i+1] - context->communicator->lfRow[i] ;
					context->communicator->lnEnts[i] = context->communicator->lfEnt[i+1] - context->communicator->lfEnt[i] ;
				}

				context->communicator->lnrows[nprocs-1] = context->gnrows - context->communicator->lfRow[nprocs-1] ;
				context->communicator->lnEnts[nprocs-1] = gnnz - context->communicator->lfEnt[nprocs-1];

			}
			MPI_safecall(MPI_Bcast(context->communicator->lfRow,  nprocs, ghost_mpi_dt_midx, 0, context->communicator->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, context->communicator->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnrows, nprocs, ghost_mpi_dt_midx, 0, context->communicator->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnEnts, nprocs, ghost_mpi_dt_midx, 0, context->communicator->mpicomm));


		} else 
		{ // don't read rpt, only fill lfrow, lnrows
			UNUSED(matrixPath);
			//	int nprocs = ghost_getNumberOfProcesses();
			//	context->communicator->lnrows   = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 
			//	context->communicator->lfRow    = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 

			ghost_midx_t target_rows = (context->gnrows/nprocs);

			context->communicator->lfRow[0] = 0;

			for (i=1; i<nprocs; i++){
				context->communicator->lfRow[i] = context->communicator->lfRow[i-1]+target_rows;
			}
			for (i=0; i<nprocs-1; i++){
				context->communicator->lnrows[i] = context->communicator->lfRow[i+1] - context->communicator->lfRow[i] ;
			}
			context->communicator->lnrows[nprocs-1] = context->gnrows - context->communicator->lfRow[nprocs-1] ;
			MPI_safecall(MPI_Bcast(context->communicator->lfRow,  nprocs, ghost_mpi_dt_midx, 0, context->communicator->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnrows, nprocs, ghost_mpi_dt_midx, 0, context->communicator->mpicomm));
		}

	} else {
		context->communicator = NULL;
	}

#else
	context->communicator = NULL;
#endif

	DEBUG_LOG(1,"Context created successfully");
	return context;
}

ghost_mat_t * ghost_initMatrix(ghost_mtraits_t *traits)
{
	ghost_mat_t* mat;
	switch (traits->format) {
		case GHOST_SPM_FORMAT_CRS:
			mat = ghost_CRS_init(traits);
			break;
		case GHOST_SPM_FORMAT_SELL:
			mat = ghost_SELL_init(traits);
			break;
		default:
			WARNING_LOG("Invalid sparse matrix format. Falling back to CRS!");
			traits->format = GHOST_SPM_FORMAT_CRS;
			mat = ghost_CRS_init(traits);
	}
	return mat;	
}

int ghost_spmvm(ghost_context_t *context, ghost_vec_t *res, ghost_mat_t *mat, ghost_vec_t *invec, 
		int *spmvmOptions)
{
	ghost_solver_t solver = NULL;
	ghost_pickSpMVMMode(context,spmvmOptions);
	solver = context->solvers[ghost_getSpmvmModeIdx(*spmvmOptions)];

	if (!solver)
		return GHOST_FAILURE;

	solver(context,res,mat,invec,*spmvmOptions);

	return GHOST_SUCCESS;
}

void ghost_freeContext(ghost_context_t *context)
{
	DEBUG_LOG(1,"Freeing context");
	if (context != NULL) {
		free(context->solvers);
		ghost_freeCommunicator(context->communicator);

		free(context);
	}
	DEBUG_LOG(1,"Context freed successfully");
}


void ghost_normalizeVec(ghost_vec_t *vec)
{
	if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
		if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
			complex float res;
			ghost_dotProduct(vec,vec,&res);
			res = 1.f/csqrtf(res);
			vec->scale(vec,&res);
		} else {
			float res;
			ghost_dotProduct(vec,vec,&res);
			res = 1.f/sqrtf(res);
			vec->scale(vec,&res);
		}
	} else {
		if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
			complex double res;
			ghost_dotProduct(vec,vec,&res);
			res = 1./csqrt(res);
			vec->scale(vec,&res);
		} else {
			double res;
			ghost_dotProduct(vec,vec,&res);
			res = 1./sqrt(res);
			vec->scale(vec,&res);
		}
	}
}

void ghost_dotProduct(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{
	vec->dotProduct(vec,vec2,res);
#ifdef GHOST_MPI
	int v;
	if (!(vec->traits->flags & GHOST_VEC_GLOBAL)) {
		for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
			MPI_safecall(MPI_Allreduce(MPI_IN_PLACE, (char *)res+ghost_sizeofDataType(vec->traits->datatype)*v, 1, ghost_mpi_dataType(vec->traits->datatype), MPI_SUM, vec->context->communicator->mpicomm));
		}
	}
#endif

}

void ghost_vecToFile(ghost_vec_t *vec, char *path, ghost_context_t *ctx)
{
	int64_t nrows = vec->traits->nrows;
#ifdef GHOST_MPI
	MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,&nrows,1,MPI_INTEGER8,MPI_SUM,vec->context->communicator->mpicomm));
#endif
	if (ghost_getRank(vec->context->communicator->mpicomm) == 0) { // write header

		int file;

		if ((file = open(path, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR)) == -1){
			ABORT("Could not open vector file %s",path);
		}

		int offs = 0;

		int32_t endianess = ghost_archIsBigEndian();
		int32_t version = 1;
		int32_t order = GHOST_BINVEC_ORDER_COL_FIRST;
		int32_t datatype = vec->traits->datatype;
		int64_t ncols = (int64_t)1;

		pwrite(file,&endianess,sizeof(endianess),offs);
		pwrite(file,&version,sizeof(version),    offs+=sizeof(endianess));
		pwrite(file,&order,sizeof(order),        offs+=sizeof(version));
		pwrite(file,&datatype,sizeof(datatype),  offs+=sizeof(order));
		pwrite(file,&nrows,sizeof(nrows),        offs+=sizeof(datatype));
		pwrite(file,&ncols,sizeof(ncols),        offs+=sizeof(nrows));

		close(file);


	}
	if ((ctx==NULL) || !(ctx->flags & GHOST_CONTEXT_DISTRIBUTED))
		vec->toFile(vec,path,0,1);
	else
		vec->toFile(vec,path,ctx->communicator->lfRow[ghost_getRank(vec->context->communicator->mpicomm)],1);
}

void ghost_vecFromFile(ghost_vec_t *vec, char *path, ghost_context_t *ctx)
{
	if ((ctx == NULL) || !(ctx->flags & GHOST_CONTEXT_DISTRIBUTED))
		vec->fromFile(vec,ctx,path,0);
	else
		vec->fromFile(vec,ctx,path,ctx->communicator->lfRow[ghost_getRank(vec->context->communicator->mpicomm)]);
}
void ghost_vecFromScalar(ghost_vec_t *v, ghost_context_t *c, void *s)
{
	v->fromScalar(v,c,s);
}

void ghost_vecFromFunc(ghost_vec_t *v, ghost_context_t *c, void (*func)(int,int,void*))
{
	v->fromFunc(v,c,func);
}

void ghost_freeVec(ghost_vec_t *vec)
{
	vec->destroy(vec);
}

void ghost_matFromFile(ghost_mat_t *m, ghost_context_t *c, char *p)
{
	m->fromFile(m,c,p);
}

int ghost_gemm(char *transpose, ghost_vec_t *v, ghost_vec_t *w, ghost_vec_t *x, void *alpha, void *beta, int reduce)
{

	// TODO if rhs vector data will not be continous
	complex double zero = 0.+I*0.;
	if (v->traits->nrows != w->traits->nrows) {
		WARNING_LOG("GEMM with vector of different size does not work");
		return GHOST_FAILURE;
	}
	if (v->traits->datatype != w->traits->datatype) {
		WARNING_LOG("GEMM with vector of different datatype does not work");
		return GHOST_FAILURE;
	}

	//	ghost_vtraits_t *restraits = (ghost_vtraits_t*)ghost_malloc(sizeof(ghost_vtraits_t));;
	//	restraits->flags = GHOST_VEC_DEFAULT;
	//	restraits->nrows = v->traits->nvecs; //TODO set padded, halo to zero?
	//	restraits->nvecs = w->traits->nvecs;
	//	restraits->datatype = v->traits->datatype;


	//	*res = ghost_createVector(restraits);
	//	(*res)->fromScalar(*res,NULL,&zero); //vec rows are valid, ctx can be NULL

#ifdef LONGIDX // TODO
	ABORT("GEMM with LONGIDX not implemented");
#endif

	int m,n,k;
	m = v->traits->nvecs;
	n = w->traits->nvecs;
	k = v->traits->nrows;

	if (v->traits->datatype != w->traits->datatype) {
		ABORT("Dgemm with mixed datatypes does not work!");
	}

	DEBUG_LOG(1,"Calling XGEMM with (%dx%d) * (%dx%d) = (%dx%d)",m,k,k,n,m,n);

	if (v->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
		if (v->traits->datatype & GHOST_BINCRS_DT_DOUBLE) {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(x->context->communicator->mpicomm) == 0) { 
					zgemm(transpose,"N", &m, &n, &k, (BLAS_Complex16 *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (BLAS_Complex16 *)beta, x->val, &(x->traits->nrowspadded)); 
				} else {
					zgemm(transpose,"N", &m, &n, &k, (BLAS_Complex16 *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (BLAS_Complex16 *)&zero, x->val, &(x->traits->nrowspadded)); 
				}
			} else {
				zgemm(transpose,"N", &m, &n, &k, (BLAS_Complex16 *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (BLAS_Complex16 *)beta, x->val, &(x->traits->nrowspadded)); 
			}

		} else {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(x->context->communicator->mpicomm) == 0) { 
					cgemm(transpose,"N", &m, &n, &k, (BLAS_Complex8 *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (BLAS_Complex8 *)beta, x->val, &(x->traits->nrowspadded));
				} else {
					cgemm(transpose,"N", &m, &n, &k, (BLAS_Complex8 *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (BLAS_Complex8 *)&zero, x->val, &(x->traits->nrowspadded));
				}
			} else {
				cgemm(transpose,"N", &m, &n, &k, (BLAS_Complex8 *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (BLAS_Complex8 *)beta, x->val, &(x->traits->nrowspadded));
			}
		}	
	} else {
		if (v->traits->datatype & GHOST_BINCRS_DT_DOUBLE) {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(x->context->communicator->mpicomm) == 0) { 
					dgemm(transpose,"N", &m, &n, &k, (double *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (double *)beta, x->val, &(x->traits->nrowspadded));
				} else {
					dgemm(transpose,"N", &m, &n, &k, (double *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (double *)&zero, x->val, &(x->traits->nrowspadded));
				}
			} else {
				dgemm(transpose,"N", &m, &n, &k, (double *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (double *)beta, x->val, &(x->traits->nrowspadded));
			}
		} else {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(x->context->communicator->mpicomm) == 0) { 
					sgemm(transpose,"N", &m, &n, &k, (float *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (float *)beta, x->val, &(x->traits->nrowspadded));
				} else {
					sgemm(transpose,"N", &m, &n, &k, (float *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (float *)&zero, x->val, &(x->traits->nrowspadded));
				}
			} else {
				sgemm(transpose,"N", &m, &n, &k, (float *)alpha, v->val, &(v->traits->nrowspadded), w->val, &(w->traits->nrowspadded), (float *)beta, x->val, &(x->traits->nrowspadded));
			}
		}	
	}

#ifdef GHOST_MPI // TODO get rid of for loops
	int i,j;
	if (reduce == GHOST_GEMM_NO_REDUCE) {
		return GHOST_SUCCESS;
	} else if (reduce == GHOST_GEMM_ALL_REDUCE) {
		for (i=0; i<x->traits->nvecs; ++i) {
			for (j=0; j<x->traits->nrows; ++j) {
				MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,((char *)(x->val))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),1,ghost_mpi_dataType(x->traits->datatype),MPI_SUM,x->context->communicator->mpicomm));

			}
		}
	} else {
		for (i=0; i<x->traits->nvecs; ++i) {
			for (j=0; j<x->traits->nrows; ++j) {
				if (ghost_getRank(x->context->communicator->mpicomm) == reduce) {
					MPI_safecall(MPI_Reduce(MPI_IN_PLACE,((char *)(x->val))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),1,ghost_mpi_dataType(x->traits->datatype),MPI_SUM,reduce,x->context->communicator->mpicomm));
				} else {
					MPI_safecall(MPI_Reduce(((char *)(x->val))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),NULL,1,ghost_mpi_dataType(x->traits->datatype),MPI_SUM,reduce,x->context->communicator->mpicomm));
				}
			}
		}
	}
#else
	UNUSED(reduce);
#endif

	return GHOST_SUCCESS;

}
