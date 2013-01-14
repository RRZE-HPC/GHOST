#define _GNU_SOURCE

#include "ghost.h"
#include "ghost_util.h"
#include "ghost_mat.h"
#include "ghost_vec.h"

#ifdef MPI
#include <mpi.h>
#include "mpihelper.h"
#endif

#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>
#include <dlfcn.h>
#include <dirent.h>
#include <linux/limits.h>


#include <string.h>
#include <sched.h>
#include <omp.h>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

static int options;
#ifdef MPI
static int MPIwasInitialized;
#endif


static ghost_mnnz_t context_gnnz (ghost_context_t * context)
{
	ghost_mnnz_t gnnz;
	ghost_mnnz_t lnnz = context->fullMatrix->nnz(context->fullMatrix);

#ifdef MPI
	if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
		MPI_safecall(MPI_Allreduce(&lnnz,&gnnz,1,ghost_mpi_dt_mnnz,MPI_SUM,MPI_COMM_WORLD));
	} else {
		gnnz = lnnz;
	}
#else
	gnnz = lnnz;
#endif

	return gnnz;
}

static ghost_mnnz_t context_lnnz (ghost_context_t * context)
{
	return context->fullMatrix->nnz(context->fullMatrix);
}

static ghost_mnnz_t context_gnrows (ghost_context_t * context)
{
	ghost_mnnz_t gnrows;
	ghost_mnnz_t lnrows = context->fullMatrix->nrows(context->fullMatrix);

#ifdef MPI
	if (context->flags & GHOST_CONTEXT_DISTRIBUTED) { 
		MPI_safecall(MPI_Allreduce(&lnrows,&gnrows,1,ghost_mpi_dt_midx,MPI_SUM,MPI_COMM_WORLD));
	} else {
		gnrows = lnrows;
	}
#else
	gnrows = lnrows;
#endif

	return gnrows;
}

static ghost_mnnz_t context_lnrows (ghost_context_t * context)
{
	return context->fullMatrix->nrows(context->fullMatrix);
}

static ghost_mnnz_t context_gncols (ghost_context_t * context)
{
	ghost_mnnz_t gncols;
	ghost_mnnz_t lncols = context->fullMatrix->ncols(context->fullMatrix);

#ifdef MPI
	if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
		MPI_safecall(MPI_Allreduce(&lncols,&gncols,1,ghost_mpi_dt_midx,MPI_SUM,MPI_COMM_WORLD));
	} else {
		gncols = lncols;
	}
#else
	gncols = lncols;
#endif

	return gncols;
}

static ghost_mnnz_t context_lncols (ghost_context_t * context)
{
	return context->fullMatrix->ncols(context->fullMatrix);
}

#ifdef MPI
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

int ghost_init(int argc, char **argv, int spmvmOptions)
{
	int me;

#ifdef MPI
	int req, prov;

	req = MPI_THREAD_FUNNELED; // TODO not if not all kernels configured

	MPI_safecall(MPI_Initialized(&MPIwasInitialized));
	if (!MPIwasInitialized) {
		MPI_safecall(MPI_Init_thread(&argc, &argv, req, &prov ));

		if (req != prov) {
			DEBUG_LOG(0,"Warning! Required MPI threading level (%d) is not "
					"provided (%d)!",req,prov);
		}
	}
	me = ghost_getRank();;

	setupSingleNodeComm();

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
#endif

#else // ifdef MPI
	UNUSED(argc);
	UNUSED(argv);
	me = 0;
	spmvmOptions |= GHOST_OPTION_SERIAL_IO; // important for createMatrix()

#endif // ifdef MPI

	if (spmvmOptions & GHOST_OPTION_PIN || spmvmOptions & GHOST_OPTION_PIN_SMT) {
		int nCores;
		int nPhysCores = ghost_getNumberOfPhysicalCores();
		if (spmvmOptions & GHOST_OPTION_PIN)
			nCores = nPhysCores;
		else
			nCores = ghost_getNumberOfHwThreads();

		int offset = nPhysCores/ghost_getNumberOfRanksOnNode();
		omp_set_num_threads(nCores/ghost_getNumberOfRanksOnNode());
#pragma omp parallel
		{
			int error;
			int coreNumber;

			if (spmvmOptions & GHOST_OPTION_PIN_SMT)
				coreNumber = omp_get_thread_num()/2+(offset*(ghost_getLocalRank()))+(omp_get_thread_num()%2)*nPhysCores;
			else
				coreNumber = omp_get_thread_num()+(offset*(ghost_getLocalRank()));

			DEBUG_LOG(1,"Pinning thread %d to core %d",omp_get_thread_num(),coreNumber);
			cpu_set_t cpu_set;
			CPU_ZERO(&cpu_set);
			CPU_SET(coreNumber, &cpu_set);

			error = sched_setaffinity((pid_t)0, sizeof(cpu_set_t), &cpu_set);

			if (error != 0) {
				DEBUG_LOG(0,"Pinning thread to core %d failed (%d): %s", 
						coreNumber, error, strerror(error));
			}
		}
	}

#ifdef LIKWID_PERFMON
	LIKWID_MARKER_INIT;

	//#pragma omp parallel
	//	LIKWID_MARKER_THREADINIT;
#endif

#ifdef OPENCL
	CL_init();
#endif

	options = spmvmOptions;



	return me;
}

void ghost_finish()
{

#ifdef LIKWID_PERFMON
	LIKWID_MARKER_CLOSE;
#endif

#ifdef OPENCL
	CL_finish();
#endif

#ifdef MPI
	if (!MPIwasInitialized)
		MPI_Finalize();
#endif

}

ghost_vec_t *ghost_createVector(ghost_context_t *context, unsigned int flags, ghost_vdat_t (*fp)(int))
{

	ghost_vdat_t *val;
	ghost_vidx_t nrows;
	size_t size_val;
	ghost_mat_t *matrix = context->fullMatrix;


	if ((context->flags & GHOST_CONTEXT_GLOBAL) || (flags & GHOST_VEC_GLOBAL))
	{
		size_val = (size_t)matrix->nrows(matrix)*sizeof(ghost_vdat_t);
		val = (ghost_vdat_t*) allocateMemory( size_val, "vec->val");
		nrows = matrix->nrows(matrix);


		DEBUG_LOG(1,"NUMA-aware allocation of vector with %"PRmatIDX" rows",nrows);

		ghost_midx_t i;
		if (fp) {
#pragma omp parallel for schedule(runtime)
			for (i=0; i<matrix->nrows(matrix); i++) 
				val[i] = fp(i);
		}else {
#ifdef GHOST_VEC_COMPLEX
#pragma omp parallel for schedule(runtime)
			for (i=0; i<matrix->nrows(matrix); i++) val[i] = 0.+I*0.;
#else
#pragma omp parallel for schedule(runtime)
			for (i=0; i<matrix->nrows(matrix); i++) val[i] = 0.;
#endif
		}
		if (matrix->traits->flags & GHOST_SPM_PERMUTECOLIDX)
			ghost_permuteVector(val,matrix->rowPerm,nrows);

	} 
	else 
	{
		ghost_comm_t *lcrp = context->communicator;
		ghost_midx_t i;
		int me = ghost_getRank();

		if (flags & GHOST_VEC_LHS)
			nrows = lcrp->lnrows[me];
		else if (flags & GHOST_VEC_RHS)
			nrows = lcrp->lnrows[me]+lcrp->halo_elements;
		else
			ABORT("No valid type for vector (has to be one of GHOST_VEC_LHS/_RHS/_BOTH");

		size_val = (size_t)( nrows * sizeof(ghost_vdat_t) );

		val = (ghost_vdat_t*) allocateMemory( size_val, "vec->val");
		nrows = nrows;

		DEBUG_LOG(1,"NUMA-aware allocation of vector with %"PRmatIDX"+%"PRmatIDX" rows",lcrp->lnrows[me],lcrp->halo_elements);

		if (fp) {
#pragma omp parallel for schedule(runtime)
			for (i=0; i<lcrp->lnrows[me]; i++) 
				val[i] = fp(lcrp->lfRow[me]+i);
#pragma omp parallel for schedule(runtime)
			for (i=lcrp->lnrows[me]; i<nrows; i++) 
				val[i] = fp(lcrp->lfRow[me]+i);
		}else {
#ifdef GHOST_VEC_COMPLEX
#pragma omp parallel for schedule(runtime)
			for (i=0; i<lcrp->lnrows[me]; i++) val[i] = 0.+I*0.;
#pragma omp parallel for schedule(runtime)
			for (i=lcrp->lnrows[me]; i<nrows; i++) val[i] = 0.+I*0.;
#else
#pragma omp parallel for schedule(runtime)
			for (i=0; i<lcrp->lnrows[me]; i++) val[i] = 0.;
#pragma omp parallel for schedule(runtime)
			for (i=lcrp->lnrows[me]; i<nrows; i++) val[i] = 0.;
#endif
		}
	}

	ghost_vec_t* vec;
	vec = (ghost_vec_t*) allocateMemory( sizeof( ghost_vec_t ), "vec");
	vec->val = val;
	vec->nrows = nrows;
	vec->flags = flags;	

	if (!(flags & GHOST_VEC_HOST)) {
#ifdef OPENCL
		DEBUG_LOG(1,"Creating vector on OpenCL device");
		int flag;
		if (flags & GHOST_VEC_LHS) {
			if (options & GHOST_OPTION_AXPY)
				flag = CL_MEM_READ_WRITE;
			else
				flag = CL_MEM_WRITE_ONLY;
		} else if (flags & GHOST_VEC_RHS) {
			flag = CL_MEM_READ_ONLY;
		} else {
			ABORT("No valid type for vector (has to be one of GHOST_VEC_LHS/_RHS/_BOTH");
		}
		vec->CL_val_gpu = CL_allocDeviceMemoryMapped( size_val,vec->val,flag );
		CL_uploadVector(vec);
#endif

	} else {
		DEBUG_LOG(1,"Host-only vector created successfully");
	}

	return vec;

}

ghost_context_t *ghost_createContext(char *matrixPath, ghost_mtraits_t *traits, int nTraits, unsigned int context_flags) 
{
	DEBUG_LOG(1,"Creating context");
	ghost_context_t *context;
	CR_TYPE *cr = NULL;

	// copy is needed because basename() changes the string
	char *matrixPathCopy = (char *)allocateMemory(strlen(matrixPath),"matrixPathCopy");
	strncpy(matrixPathCopy,matrixPath,strlen(matrixPath));

	context = (ghost_context_t *)allocateMemory(sizeof(ghost_context_t),"context");
	context->flags = context_flags;
	context->matrixName = strtok(basename(matrixPathCopy),".");

#ifdef MPI
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

	if (context_flags & GHOST_CONTEXT_GLOBAL) {
		DEBUG_LOG(1,"Forcing serial I/O as the matrix format is a global one");
		options |= GHOST_OPTION_SERIAL_IO;
	}

	context->solvers = (ghost_solver_t *)allocateMemory(sizeof(ghost_solver_t)*GHOST_NUM_MODES,"solvers");


	if (context->flags & GHOST_CONTEXT_DISTRIBUTED)
	{ // distributed matrix
#ifdef MPI
		if (!(options & GHOST_OPTION_NO_SPLIT_KERNELS)) {
			if (!(options & GHOST_OPTION_NO_COMBINED_KERNELS)) {
				if (nTraits != 3) {
					ABORT("The number of traits has to be THREE (is: %d) if all distributed kernels are enabled",nTraits);
				}
			}
		}

		if (options & GHOST_OPTION_SERIAL_IO) 
			ghost_createDistributedContextSerial(context, cr, options, traits);
		else
			ghost_createDistributedContext(context, matrixPath, options, traits);

		context->solvers[GHOST_MODE_NOMPI] = NULL;
		context->solvers[GHOST_MODE_VECTORMODE] = &hybrid_kernel_I;
		context->solvers[GHOST_MODE_GOODFAITH] = &hybrid_kernel_II;
		context->solvers[GHOST_MODE_TASKMODE] = &hybrid_kernel_III;
#endif
	} 
	else 
	{ // global matrix
		if (nTraits != 1)
			DEBUG_LOG(1,"Warning! Ignoring all but the first given matrix traits for the global matrix.");
		UNUSED(cr); // TODO
		context->fullMatrix = ghost_initMatrix(&traits[0]);

		if (isMMfile(matrixPath))
			context->fullMatrix->fromMM(context->fullMatrix,matrixPath);
		else
			context->fullMatrix->fromBin(context->fullMatrix,matrixPath);

#ifdef OPENCL
		if (!(traits[0].flags & GHOST_SPM_HOST))
			context->fullMatrix->CLupload(context->fullMatrix);
#endif

		DEBUG_LOG(1,"Created global %s matrix",context->fullMatrix->formatName(context->fullMatrix));

		context->solvers[GHOST_MODE_NOMPI] = &ghost_solver_nompi;
		context->solvers[GHOST_MODE_VECTORMODE] = NULL;
		context->solvers[GHOST_MODE_GOODFAITH] = NULL;
		context->solvers[GHOST_MODE_TASKMODE] = NULL;
	}

	//#endif
	context->lnnz = &context_lnnz;
	context->lnrows = &context_lnrows;
	context->lncols = &context_lncols;
	context->gnnz = &context_gnnz;
	context->gnrows = &context_gnrows;
	context->gncols = &context_gncols;

	DEBUG_LOG(1,"%"PRmatIDX"x%"PRmatIDX" matrix (%"PRmatNNZ" nonzeros) created successfully",context->gncols(context),context->gnrows(context),context->gnnz(context));


	DEBUG_LOG(1,"Context created successfully");
	return context;
}

ghost_mat_t * ghost_initMatrix(ghost_mtraits_t *traits)
{
	char pluginPath[PATH_MAX];
	DIR * pluginDir = opendir(PLUGINPATH);
	struct dirent * dirEntry;
	ghost_spmf_plugin_t myPlugin;


	DEBUG_LOG(1,"Searching in %s for plugin providing %s",PLUGINPATH,traits->format);
	if (pluginDir) {
		while (0 != (dirEntry = readdir(pluginDir))) {
			snprintf(pluginPath,PATH_MAX,"%s/%s",PLUGINPATH,dirEntry->d_name);
			DEBUG_LOG(2,"Trying %s",pluginPath);
			myPlugin.so = dlopen(pluginPath,RTLD_LAZY);
			if (!myPlugin.so) {
				DEBUG_LOG(2,"Could not open %s",pluginPath);
				continue;
			}

			myPlugin.formatID = (char *)dlsym(myPlugin.so,"formatID");
			if (!myPlugin.formatID) ABORT("The plugin does not provide a formatID!");
			if (!strcasecmp(traits->format,myPlugin.formatID)) 
			{
				DEBUG_LOG(1,"Found plugin: %s",pluginPath);

				myPlugin.init = (ghost_spmf_init_t)dlsym(myPlugin.so,"init");
				myPlugin.name = (char *)dlsym(myPlugin.so,"name");
				myPlugin.version = (char *)dlsym(myPlugin.so,"version");

				DEBUG_LOG(1,"Successfully registered %s v%s",myPlugin.name, myPlugin.version);

				closedir(pluginDir);
				return myPlugin.init(traits);
			} else {
				DEBUG_LOG(2,"Skipping plugin: %s",myPlugin.formatID);
			}

		}
		closedir(pluginDir);
		ABORT("There is no such plugin providing %s",traits->format);

	} else {
		ABORT("The plugin directory does not exist");
	}

	return NULL;


}



int ghost_spmvm(ghost_vec_t *res, ghost_context_t *context, ghost_vec_t *invec, 
		int kernel)
{

	ghost_solver_t solver = NULL;
	solver = context->solvers[kernel];

	if (!solver)
		return GHOST_FAILURE;

	solver(res,context,invec,options);

	return GHOST_SUCCESS;
}

void ghost_freeContext(ghost_context_t *context)
{
	/*	if (context) {
		if (context->fullMatrix && context->fullMatrix->destroy)
		context->fullMatrix->destroy(context->fullMatrix);

		if (context->localMatrix && context->localMatrix->destroy)
		context->localMatrix->destroy(context->localMatrix);

		if (context->remoteMatrix && context->remoteMatrix->destroy)
		context->remoteMatrix->destroy(context->remoteMatrix);

		free(context->solvers);

		if (context->communicator)
		ghost_freeCommunicator(context->communicator);

		free(context);
		}*/
	UNUSED(context);
	//TODO
}
