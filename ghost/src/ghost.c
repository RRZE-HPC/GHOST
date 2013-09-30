#define _GNU_SOURCE

#include "ghost.h"
#include "ghost_util.h"
#include "ghost_mat.h"
#include "ghost_vec.h"
#include "ghost_taskq.h"
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
#include <errno.h>

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

hwloc_topology_t topology;

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

	setupSingleNodeComm();
	MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&GHOST_MPI_DT_C));
	MPI_safecall(MPI_Type_commit(&GHOST_MPI_DT_C));
	MPI_safecall(MPI_Op_create((MPI_User_function *)&MPI_add_c,1,&GHOST_MPI_OP_SUM_C));

	MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&GHOST_MPI_DT_Z));
	MPI_safecall(MPI_Type_commit(&GHOST_MPI_DT_Z));
	MPI_safecall(MPI_Op_create((MPI_User_function *)&MPI_add_z,1,&GHOST_MPI_OP_SUM_Z));

#else // ifdef GHOST_MPI
	UNUSED(argc);
	UNUSED(argv);

#endif // ifdef GHOST_MPI

#ifdef LIKWID_PERFMON
	LIKWID_MARKER_INIT;
#pragma omp parallel
	LIKWID_MARKER_THREADINIT;

#endif

#ifdef OPENCL
	CL_init();
#endif
//#ifdef CUDA
//	CU_init();
//#endif
	
	hwloc_topology_init(&topology);
	hwloc_topology_load(topology);

	return GHOST_SUCCESS;
}

void ghost_finish()
{

	ghost_taskq_finish();
	ghost_thpool_finish();
//	hwloc_topology_destroy(topology);

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
	mat->context = context;
	return mat;	
}

ghost_context_t *ghost_createContext(int64_t gnrows, int64_t gncols, int context_flags, char *matrixPath, MPI_Comm comm, double weight) 
{
	ghost_context_t *context;
	int i;

	context = (ghost_context_t *)ghost_malloc(sizeof(ghost_context_t));
	context->flags = context_flags;
	context->rowPerm = NULL;
	context->invRowPerm = NULL;
	context->mpicomm = comm;

	if ((gnrows == GHOST_GET_DIM_FROM_MATRIX) || (gncols == GHOST_GET_DIM_FROM_MATRIX)) {
		ghost_matfile_header_t fileheader;
		ghost_readMatFileHeader(matrixPath,&fileheader);
#ifndef LONGIDX
		if ((fileheader.nrows >= (int64_t)INT_MAX) || (fileheader.ncols >= (int64_t)INT_MAX)) {
			ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
		}
#endif
		if (gnrows == GHOST_GET_DIM_FROM_MATRIX)
			context->gnrows = (ghost_midx_t)fileheader.nrows;
		if (gncols == GHOST_GET_DIM_FROM_MATRIX)
			context->gncols = (ghost_midx_t)fileheader.ncols;

	} else {
#ifndef LONGIDX
		if ((gnrows >= (int64_t)INT_MAX) || (gncols >= (int64_t)INT_MAX)) {
			ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
		}
#endif
		context->gnrows = (ghost_midx_t)gnrows;
		context->gncols = (ghost_midx_t)gncols;
	}
	DEBUG_LOG(1,"Creating context with dimension %"PRmatIDX"x%"PRmatIDX,context->gnrows,context->gncols);

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
		context->communicator->halo_elements = -1;

		int nprocs = ghost_getNumberOfRanks(context->mpicomm);

		context->communicator->lnEnts   = (ghost_mnnz_t*)       ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
		context->communicator->lfEnt    = (ghost_mnnz_t*)       ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
		context->communicator->lnrows   = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 
		context->communicator->lfRow    = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 

		if (context->flags & GHOST_CONTEXT_WORKDIST_NZE)
		{ // read rpt and fill lfrow, lnrows, lfent, lnents
			ghost_midx_t *rpt = (ghost_midx_t *)ghost_malloc(sizeof(ghost_midx_t)*(context->gnrows+1));
			ghost_mnnz_t gnnz;

			if (ghost_getRank(context->mpicomm) == 0) {
				FILE * filed;
				size_t ret;

				if ((filed = fopen64(matrixPath, "r")) == NULL){
					ABORT("Could not open binary CRS file %s",matrixPath);
				}
				if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
					ABORT("Seek failed");
				}
#ifdef LONGIDX
				if ((ret = fread(rpt, GHOST_BINCRS_SIZE_RPT_EL, context->gnrows+1,filed)) != (context->gnrows+1)){
					ABORT("fread failed: %s (%lu)",strerror(errno),ret);
				}
#else // casting
				DEBUG_LOG(1,"Casting from 64 bit to 32 bit row pointers");
				int64_t *tmp = (int64_t *)ghost_malloc((context->gnrows+1)*8);
				if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, context->gnrows+1,filed)) != (context->gnrows+1)){
					ABORT("fread failed: %s (%lu)",strerror(errno),ret);
				}
				for( i = 0; i < context->gnrows+1; i++ ) {
					if (tmp[i] >= (int64_t)INT_MAX) {
						ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
					}
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

				fclose(filed);
			}
			MPI_safecall(MPI_Bcast(context->communicator->lfRow,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnrows, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnEnts, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));


		} else
		{ // don't read rpt, only fill lfrow, lnrows
			UNUSED(matrixPath);
			double allweights;
			MPI_safecall(MPI_Allreduce(&weight,&allweights,1,MPI_DOUBLE,MPI_SUM,context->mpicomm))
			
//			ghost_midx_t target_rows = (context->gnrows/nprocs);
			ghost_midx_t target_rows = (ghost_midx_t)(context->gnrows*((double)weight/(double)allweights));

			context->communicator->lfRow[0] = 0;

			for (i=1; i<nprocs; i++){
				context->communicator->lfRow[i] = context->communicator->lfRow[i-1]+target_rows;
			}
			for (i=0; i<nprocs-1; i++){
				context->communicator->lnrows[i] = context->communicator->lfRow[i+1] - context->communicator->lfRow[i] ;
			}
			context->communicator->lnrows[nprocs-1] = context->gnrows - context->communicator->lfRow[nprocs-1] ;
			MPI_safecall(MPI_Bcast(context->communicator->lfRow,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
			MPI_safecall(MPI_Bcast(context->communicator->lnrows, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
		}

	} else {
		context->communicator = NULL;
	}

#else
	UNUSED(weight);
	context->communicator = NULL;
#endif

	DEBUG_LOG(1,"Context created successfully");
	return context;
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
		free(context->rowPerm);
		free(context->invRowPerm);
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
	/*	double zero = 0.;
		ghost_context_t *context;
		ghost_vec_t *gv1, *gv2;
		context = ghost_createContext(vec->context->gnrows,vec->context->gncols,GHOST_CONTEXT_GLOBAL,NULL,MPI_COMM_WORLD);

		ghost_vtraits_t gtraits = GHOST_VTRAITS_INIT(.flags = GHOST_VEC_RHS|GHOST_VEC_HOST, .datatype = vec->traits->datatype);
		gv1 = ghost_createVector(context, &gtraits);
		gv2 = ghost_createVector(context, &gtraits);
		gv1->fromScalar(gv1,&zero);
		gv2->fromScalar(gv2,&zero);

		vec->collect(vec,gv1,NULL);
		vec2->collect(vec2,gv2,NULL);

		if (ghost_getRank(vec->context->mpicomm) == 0) {
		gv1->dotProduct(gv1,gv2,res);	
		}
		MPI_safecall(MPI_Bcast(res,1,ghost_mpi_dataType(vec->traits->datatype),0,vec->context->mpicomm));
		gv1->destroy(gv1);
		gv2->destroy(gv2);
	 */	

	vec->dotProduct(vec,vec2,res);
#ifdef GHOST_MPI
	int v;
	if (!(vec->traits->flags & GHOST_VEC_GLOBAL)) {
		for (v=0; v<MIN(vec->traits->nvecs,vec2->traits->nvecs); v++) {
			MPI_safecall(MPI_Allreduce(MPI_IN_PLACE, (char *)res+ghost_sizeofDataType(vec->traits->datatype)*v, 1, ghost_mpi_dataType(vec->traits->datatype), ghost_mpi_op_sum(vec->traits->datatype), vec->context->mpicomm));
		}
	}
#endif

}

void ghost_vecToFile(ghost_vec_t *vec, char *path)
{
#ifdef GHOST_MPI
	size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
	
	int32_t endianess = ghost_archIsBigEndian();
	int32_t version = 1;
	int32_t order = GHOST_BINVEC_ORDER_COL_FIRST;
	int32_t datatype = vec->traits->datatype;
	int64_t nrows = (int64_t)vec->context->gnrows;
	int64_t ncols = (int64_t)vec->traits->nvecs;
	MPI_File fileh;
	MPI_Status status;
	MPI_safecall(MPI_File_open(vec->context->mpicomm,path,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&fileh));

	if (ghost_getRank(vec->context->mpicomm) == 0) 
	{ // write header AND portion
		MPI_safecall(MPI_File_write(fileh,&endianess,1,MPI_INT,&status));
		MPI_safecall(MPI_File_write(fileh,&version,1,MPI_INT,&status));
		MPI_safecall(MPI_File_write(fileh,&order,1,MPI_INT,&status));
		MPI_safecall(MPI_File_write(fileh,&datatype,1,MPI_INT,&status));
		MPI_safecall(MPI_File_write(fileh,&nrows,1,MPI_LONG_LONG,&status));
		MPI_safecall(MPI_File_write(fileh,&ncols,1,MPI_LONG_LONG,&status));
	
	}	
	ghost_vidx_t v;
	MPI_Datatype mpidt = ghost_mpi_dataType(vec->traits->datatype);
	MPI_safecall(MPI_File_set_view(fileh,4*sizeof(int32_t)+2*sizeof(int64_t),mpidt,mpidt,"native",MPI_INFO_NULL));
	MPI_Offset fileoffset = vec->context->communicator->lfRow[ghost_getRank(vec->context->mpicomm)];
	ghost_vidx_t vecoffset = 0;
	for (v=0; v<vec->traits->nvecs; v++) {
		MPI_safecall(MPI_File_write_at(fileh,fileoffset,((char *)(vec->val))+vecoffset,vec->traits->nrows,mpidt,&status));
		fileoffset += nrows;
		vecoffset += vec->traits->nrowspadded*sizeofdt;
	}
	MPI_safecall(MPI_File_close(&fileh));


#else
		vec->toFile(vec,path);
#endif
}

void ghost_vecFromFile(ghost_vec_t *vec, char *path)
{
	if ((vec->context == NULL) || !(vec->context->flags & GHOST_CONTEXT_DISTRIBUTED))
		vec->fromFile(vec,path,0);
	else
		vec->fromFile(vec,path,vec->context->communicator->lfRow[ghost_getRank(vec->context->mpicomm)]);
}
void ghost_vecFromScalar(ghost_vec_t *v, void *s)
{
	v->fromScalar(v,s);
}

void ghost_vecFromFunc(ghost_vec_t *v, void (*func)(int,int,void*))
{
	v->fromFunc(v,func);
}

void ghost_freeVec(ghost_vec_t *vec)
{
	vec->destroy(vec);
}

void ghost_matFromFile(ghost_mat_t *m, char *p)
{
	m->fromFile(m,p);
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

	ghost_vidx_t m,n,k;
	m = v->traits->nvecs;
	n = w->traits->nvecs;
	k = v->traits->nrows;

	if (v->traits->datatype != w->traits->datatype) {
		ABORT("Dgemm with mixed datatypes does not work!");
	}

	DEBUG_LOG(1,"Calling XGEMM with (%"PRvecIDX"x%"PRvecIDX") * (%"PRvecIDX"x%"PRvecIDX") = (%"PRvecIDX"x%"PRvecIDX")",m,k,k,n,m,n);

	if (v->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
		if (v->traits->datatype & GHOST_BINCRS_DT_DOUBLE) {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(x->context->mpicomm) == 0) { 
					zgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, 
							(BLAS_Complex16 *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, 
							(ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex16 *)beta, x->val, 
							(ghost_blas_idx_t *)&(x->traits->nrowspadded)); 
				} else {
					zgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex16 *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex16 *)&zero, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded)); 
				}
			} else {
				zgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex16 *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex16 *)beta, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded)); 
			}

		} else {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(x->context->mpicomm) == 0) { 
					cgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex8 *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex8 *)beta, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				} else {
					cgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex8 *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex8 *)&zero, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				}
			} else {
				cgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (BLAS_Complex8 *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (BLAS_Complex8 *)beta, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
			}
		}	
	} else {
		if (v->traits->datatype & GHOST_BINCRS_DT_DOUBLE) {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(x->context->mpicomm) == 0) { 
					dgemm(transpose,"N", (ghost_blas_idx_t *)&m,(ghost_blas_idx_t *) &n, (ghost_blas_idx_t *)&k, (double *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (double *)beta, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				} else {
					dgemm(transpose,"N",(ghost_blas_idx_t *) &m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (double *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (double *)&zero, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				}
			} else {
				dgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (double *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (double *)beta, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
			}
		} else {
			if (reduce == GHOST_GEMM_ALL_REDUCE) { // make sure that the initial value of x only gets added up once
				if (ghost_getRank(x->context->mpicomm) == 0) { 
					sgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (float *)alpha, v->val,(ghost_blas_idx_t *) &(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (float *)beta, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				} else {
					sgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (float *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (float *)&zero, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
				}
			} else {
				sgemm(transpose,"N", (ghost_blas_idx_t *)&m, (ghost_blas_idx_t *)&n, (ghost_blas_idx_t *)&k, (float *)alpha, v->val, (ghost_blas_idx_t *)&(v->traits->nrowspadded), w->val, (ghost_blas_idx_t *)&(w->traits->nrowspadded), (float *)beta, x->val, (ghost_blas_idx_t *)&(x->traits->nrowspadded));
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
				MPI_safecall(MPI_Allreduce(MPI_IN_PLACE,((char *)(x->val))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),1,ghost_mpi_dataType(x->traits->datatype),ghost_mpi_op_sum(x->traits->datatype),x->context->mpicomm));

			}
		}
	} else {
		for (i=0; i<x->traits->nvecs; ++i) {
			for (j=0; j<x->traits->nrows; ++j) {
				if (ghost_getRank(x->context->mpicomm) == reduce) {
					MPI_safecall(MPI_Reduce(MPI_IN_PLACE,((char *)(x->val))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),1,ghost_mpi_dataType(x->traits->datatype),ghost_mpi_op_sum(x->traits->datatype),reduce,x->context->mpicomm));
				} else {
					MPI_safecall(MPI_Reduce(((char *)(x->val))+(i*x->traits->nrowspadded+j)*ghost_sizeofDataType(x->traits->datatype),NULL,1,ghost_mpi_dataType(x->traits->datatype),ghost_mpi_op_sum(x->traits->datatype),reduce,x->context->mpicomm));
				}
			}
		}
	}
#else
	UNUSED(reduce);
#endif

	return GHOST_SUCCESS;

}
