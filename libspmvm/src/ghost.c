#define _GNU_SOURCE

#include "ghost.h"
#include "ghost_util.h"
#include "matricks.h"
#include "ghost_vec.h"

#ifdef MPI
#include <mpi.h>
#include "mpihelper.h"
#endif

#include "kernel.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/time.h>
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

static double wctime()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return (double) (tp.tv_sec + tp.tv_usec/1000000.0);
}

static ghost_mnnz_t setup_gnnz (ghost_setup_t * setup)
{
	ghost_mnnz_t gnnz;
	ghost_mnnz_t lnnz = setup->fullMatrix->nnz(setup->fullMatrix);

	MPI_safecall(MPI_Allreduce(&lnnz,&gnnz,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD));

	return gnnz;
}

static ghost_mnnz_t setup_lnnz (ghost_setup_t * setup)
{
	return setup->fullMatrix->nnz(setup->fullMatrix);
}
	
static ghost_mnnz_t setup_gnrows (ghost_setup_t * setup)
{
	ghost_mnnz_t gnrows;
	ghost_mnnz_t lnrows = setup->fullMatrix->nrows(setup->fullMatrix);

	MPI_safecall(MPI_Allreduce(&lnrows,&gnrows,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD));

	return gnrows;
}

static ghost_mnnz_t setup_lnrows (ghost_setup_t * setup)
{
	return setup->fullMatrix->nrows(setup->fullMatrix);
}

static ghost_mnnz_t setup_gncols (ghost_setup_t * setup)
{
	ghost_mnnz_t gncols;
	ghost_mnnz_t lncols = setup->fullMatrix->ncols(setup->fullMatrix);

	MPI_safecall(MPI_Allreduce(&lncols,&gncols,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD));

	return gncols;
}

static ghost_mnnz_t setup_lncols (ghost_setup_t * setup)
{
	return setup->fullMatrix->ncols(setup->fullMatrix);
}

#if defined(COMPLEX) && defined(MPI)
typedef struct 
{
#ifdef DOUBLE
	double x;
	double y;
#endif
#ifdef SINGLE
	float x;
	float y;
#endif

} 
MPI_complex;

static void MPI_complAdd(MPI_complex *invec, MPI_complex *inoutvec, int *len)
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

int ghost_init(int argc, char **argv, int spmvmOptions)
{
	int me;

#ifdef MPI
	int req, prov, init;

	req = MPI_THREAD_FUNNELED; // TODO not if not all kernels configured

	MPI_safecall(MPI_Initialized(&init));
	if (!init) {
		MPI_safecall(MPI_Init_thread(&argc, &argv, req, &prov ));

		if (req != prov) {
			DEBUG_LOG(0,"Warning! Required MPI threading level (%d) is not "
					"provided (%d)!",req,prov);
		}
	}
	me = ghost_getRank();;

	setupSingleNodeComm();

#ifdef COMPLEX
#ifdef DOUBLE
	MPI_safecall(MPI_Type_contiguous(2,MPI_DOUBLE,&MPI_MYDATATYPE));
#endif
#ifdef SINGLE
	MPI_safecall(MPI_Type_contiguous(2,MPI_FLOAT,&MPI_MYDATATYPE));
#endif
	MPI_safecall(MPI_Type_commit(&MPI_MYDATATYPE));
	MPI_safecall(MPI_Op_create((MPI_User_function *)&MPI_complAdd,1,&MPI_MYSUM));
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

#pragma omp parallel
	LIKWID_MARKER_THREADINIT;
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
	CL_finish(options);
#endif

#ifdef MPI
	MPI_Finalize();
#endif

}

ghost_vec_t *ghost_createVector(ghost_setup_t *setup, unsigned int flags, ghost_mdat_t (*fp)(int))
{

	ghost_mdat_t *val;
	ghost_midx_t nrows;
	size_t size_val;
	ghost_mat_t *matrix = setup->fullMatrix;


	if (setup->flags & GHOST_SETUP_GLOBAL || flags & GHOST_VEC_GLOBAL)
	{
		size_val = (size_t)matrix->nrows(matrix)*sizeof(ghost_mdat_t);
		val = (ghost_mdat_t*) allocateMemory( size_val, "vec->val");
		nrows = matrix->nrows(matrix);


		DEBUG_LOG(1,"NUMA-aware allocation of vector with %"PRmatIDX" rows",nrows);

		ghost_midx_t i;
		if (fp) {
#pragma omp parallel for schedule(runtime)
			for (i=0; i<matrix->nrows(matrix); i++) 
				val[i] = fp(i);
		}else {
#ifdef COMPLEX
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
		ghost_comm_t *lcrp = setup->communicator;
		ghost_midx_t i;
		int me = ghost_getRank();

		if (flags & GHOST_VEC_LHS)
			nrows = lcrp->lnrows[me];
		else if (flags & GHOST_VEC_RHS)
			nrows = lcrp->lnrows[me]+lcrp->halo_elements;
		else
			ABORT("No valid type for vector (has to be one of GHOST_VEC_LHS/_RHS/_BOTH");

		size_val = (size_t)( nrows * sizeof(ghost_mdat_t) );

		val = (ghost_mdat_t*) allocateMemory( size_val, "vec->val");
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
#ifdef COMPLEX
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

ghost_setup_t *ghost_createSetup(char *matrixPath, ghost_mtraits_t *traits, int nTraits, unsigned int setup_flags) 
{
	DEBUG_LOG(1,"Creating setup");
	ghost_setup_t *setup;
	CR_TYPE *cr = NULL;

	// copy is needed because basename() changes the string
	char *matrixPathCopy = (char *)allocateMemory(strlen(matrixPath),"matrixPathCopy");
	strncpy(matrixPathCopy,matrixPath,strlen(matrixPath));

	setup = (ghost_setup_t *)allocateMemory(sizeof(ghost_setup_t),"setup");
	setup->flags = setup_flags;
	setup->matrixName = strtok(basename(matrixPathCopy),".");

	if (setup_flags & GHOST_SETUP_GLOBAL) {
		DEBUG_LOG(1,"Forcing serial I/O as the matrix format is a global one");
		options |= GHOST_OPTION_SERIAL_IO;
	}

#ifdef MPI
	if (setup_flags & GHOST_SETUP_DISTRIBUTED) {
	/*	if (ghost_getRank() == 0) 
		{ // root process reads row pointers (parallel IO) or entire matrix
			if (!isMMfile(matrixPath)){
				if (options & GHOST_OPTION_SERIAL_IO)
					cr = readCRbinFile(matrixPath,0,traits[0].format & GHOST_SPMFORMAT_CRSCD);
				else
					cr = readCRbinFile(matrixPath,1,traits[0].format & GHOST_SPMFORMAT_CRSCD);
			} else{
				MM_TYPE *mm = readMMFile( matrixPath);
				cr = convertMMToCRMatrix( mm );
				freeMMMatrix(mm);
			}
		} else 
		{ // dummy CRS for scattering
			cr = (CR_TYPE *)allocateMemory(sizeof(CR_TYPE),"cr");
		}

		MPI_safecall(MPI_Bcast(&(cr->nEnts),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));
		MPI_safecall(MPI_Bcast(&(cr->nrows),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));
		MPI_safecall(MPI_Bcast(&(cr->ncols),1,MPI_UNSIGNED,0,MPI_COMM_WORLD));*/
	}
#endif
	setup->solvers = (ghost_solver_t *)allocateMemory(sizeof(ghost_solver_t)*GHOST_NUM_MODES,"solvers");


	if (setup_flags & GHOST_SETUP_DISTRIBUTED)
	{ // distributed matrix
#ifdef MPI
		if (!(options & GHOST_OPTION_NO_SPLIT_KERNELS)) {
			if (!(options & GHOST_OPTION_NO_COMBINED_KERNELS)) {
				if (nTraits != 3) {
					ABORT("The number of traits has to be THREE (is: %d) if all distributed kernels are enabled",nTraits);
				}
			}
		}

		/*DEBUG_LOG(1,"Creating distributed %s-%s-%s matrices",
				ghost_matrixFormatName(traits[0]),
				ghost_matrixFormatName(traits[1]),
				ghost_matrixFormatName(traits[2]));
*/
		if (options & GHOST_OPTION_SERIAL_IO) 
			ghost_createDistributedSetupSerial(setup, cr, options, traits);
		else
			ghost_createDistributedSetup(setup, matrixPath, options, traits);

		//setup->lnrows = setup->communicator->lnrows[ghost_getRank()];

		setup->solvers[GHOST_MODE_NOMPI] = NULL;
		setup->solvers[GHOST_MODE_VECTORMODE] = &hybrid_kernel_I;
		setup->solvers[GHOST_MODE_GOODFAITH] = &hybrid_kernel_II;
		setup->solvers[GHOST_MODE_TASKMODE] = &hybrid_kernel_III;
#else
		ABORT("Creating a distributed matrix without MPI is not possible");
#endif // MPI
	} 
	else 
	{ // global matrix
		if (nTraits != 1)
			DEBUG_LOG(1,"Warning! Ignoring all but the first given matrix traits for the global matrix.");
		UNUSED(cr); // TODO
		setup->fullMatrix = ghost_initMatrix(&traits[0]);

		if (isMMfile(matrixPath))
			setup->fullMatrix->fromMM(setup->fullMatrix,matrixPath);
		else
			setup->fullMatrix->fromBin(setup->fullMatrix,matrixPath);



		//	setup->fullMatrix = ghost_createMatrixFromCRS(cr,traits[0]);
		DEBUG_LOG(1,"Created global %s matrix",setup->fullMatrix->formatName(setup->fullMatrix));
		//setup->nnz = setup->fullMatrix->nnz(setup->fullMatrix);
		//setup->nrows = setup->fullMatrix->nrows(setup->fullMatrix);
		//setup->ncols = setup->fullMatrix->ncols(setup->fullMatrix);
		//setup->lnrows = setup->nrows;

		setup->solvers[GHOST_MODE_NOMPI] = &hybrid_kernel_0;
		setup->solvers[GHOST_MODE_VECTORMODE] = NULL;
		setup->solvers[GHOST_MODE_GOODFAITH] = NULL;
		setup->solvers[GHOST_MODE_TASKMODE] = NULL;
	}

	/*#ifdef OPENCL
	  if (!(flags & GHOST_SPM_HOSTONLY))
	  {
	  DEBUG_LOG(1,"Skipping device matrix creation because the matrix ist host-only.");
	  } else if (deviceFormats == NULL) 
	  {
	  ABORT("Device matrix formats have to be passed to SPMVM_distributeCRS!");
	  } else 
	  {
	  CL_uploadCRS ( mat, (GHOST_SPM_GPUFORMATS *)deviceFormats, options);
	  }
#else*/
	//#endif
	setup->lnnz = &setup_lnnz;
	setup->lnrows = &setup_lnrows;
	setup->lncols = &setup_lncols;
	setup->gnnz = &setup_gnnz;
	setup->gnrows = &setup_gnrows;
	setup->gncols = &setup_gncols;

	DEBUG_LOG(1,"%"PRmatIDX"x%"PRmatIDX" matrix (%"PRmatNNZ" nonzeros) created successfully",setup->gncols(setup),setup->gnrows(setup),setup->gnnz(setup));

	DEBUG_LOG(1,"Setup created successfully");
	return setup;
}

ghost_mat_t * ghost_initMatrix(ghost_mtraits_t *traits)
{
	char pluginPath[PATH_MAX];
	DIR * pluginDir = opendir(PLUGINPATH);
	struct dirent * dirEntry;
	ghost_spmf_plugin_t myPlugin;
	int found = 0;


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
			if (!strcasecmp(traits->format,myPlugin.formatID)) {
				DEBUG_LOG(1,"Found plugin: %s",pluginPath);
				found = 1;
				break;
			} else {
				DEBUG_LOG(2,"Skipping plugin: %s",myPlugin.formatID);
			}

		}
		closedir(pluginDir);

	} else {
		ABORT("The plugin directory does not exist");
	}
	if (!found) ABORT("There is no such plugin providing %s",traits->format);
	myPlugin.init = (ghost_spmf_init_t)dlsym(myPlugin.so,"init");
	myPlugin.name = (char *)dlsym(myPlugin.so,"name");
	myPlugin.version = (char *)dlsym(myPlugin.so,"version");

	DEBUG_LOG(1,"Successfully registered %s v%s",myPlugin.name, myPlugin.version);
	
	return myPlugin.init(traits);
}



double ghost_spmvm(ghost_vec_t *res, ghost_setup_t *setup, ghost_vec_t *invec, 
		int kernel, int nIter)
{
	int it;
	double time = 0;
	double oldtime=1e9;

	ghost_solver_t solver = NULL;
	solver = setup->solvers[kernel];

	if (!solver)
		return -1.0;

#ifdef MPI
	MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
#endif
#ifdef OPENCL
	CL_barrier();
#endif

	for( it = 0; it < nIter; it++ ) {
		time = wctime();
		solver(res,setup,invec,options);

#ifdef MPI
		MPI_safecall(MPI_Barrier(MPI_COMM_WORLD));
#endif
#ifdef OPENCL
		CL_barrier();
#endif
		time = wctime()-time;
		time = time<oldtime?time:oldtime;
		oldtime=time;
	}

#ifdef OPENCL
	CL_downloadVector(res);
#endif

	if ( 0x1<<kernel & GHOST_MODES_COMBINED)  {
		ghost_permuteVector(res->val,setup->fullMatrix->invRowPerm,setup->lnrows(setup));
	} else if ( 0x1<<kernel & GHOST_MODES_SPLIT ) {
		// one of those must return immediately
		ghost_permuteVector(res->val,setup->localMatrix->invRowPerm,setup->lnrows(setup));
		ghost_permuteVector(res->val,setup->remoteMatrix->invRowPerm,setup->lnrows(setup));
	}

	return time;
}

void ghost_freeSetup(ghost_setup_t *setup)
{
/*	if (setup) {
		if (setup->fullMatrix && setup->fullMatrix->destroy)
			setup->fullMatrix->destroy(setup->fullMatrix);

		if (setup->localMatrix && setup->localMatrix->destroy)
			setup->localMatrix->destroy(setup->localMatrix);

		if (setup->remoteMatrix && setup->remoteMatrix->destroy)
			setup->remoteMatrix->destroy(setup->remoteMatrix);

		free(setup->solvers);

		if (setup->communicator)
			ghost_freeCommunicator(setup->communicator);

		free(setup);
	}*/
	UNUSED(setup);
	//TODO
}
