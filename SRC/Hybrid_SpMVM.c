#include "matricks.h"
#include "mpihelper.h"
#include <string.h>
#include "timing.h"
#include <math.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include <omp.h>
#include <sched.h>

#include <likwid.h>

/* Global variables */
const char* SPM_FORMAT_NAME[]= {"ELR", "pJDS"};


int error_count, acc_error_count;

int coreId=2;
#ifdef LIKWID
int RegionId;
int numberOfThreads = 1;
int numberOfRegions = 1;
int threadId = 0;
#endif




int main( int nArgs, char* arg[] ) {

#ifdef INDIVIDUAL
	double the_cycles, the_time;
	double *perf_array;
#endif


	int pseudo_ldim;
	double acc_cycles, acc_time;

	/* const int N_MULTS = 100; */
	int N_MULTS;
	int mypid;
	int i,j; 
	size_t ws;
	unsigned long mystringlength;

	char restartfilename[50];
	char testcase[50];
	char mach_name[50];
	char kernel_version[50];
	char hostname[50];
	char model_name[50];
	char this_executable[160];
	char benchmark[50];
	char rb_flag[50];
	char cm_flag[50];
	char lw_flag[50];
	char pr_flag[50];
	char pm_flag[50];
	char io_flag[50];
	char wd_flag[50];
	char matrixpath[1024];


	VECTOR_TYPE* hlpvec_out = NULL;
	VECTOR_TYPE* hlpvec_in  = NULL;
	MM_TYPE* mm = NULL;
	CR_TYPE* cr = NULL;
	//CR_P_TYPE* cr_parallel = NULL;
	LCRP_TYPE* lcrp = NULL;
	REVBUF_TYPE* RevBuf = NULL;

	double* scatterBuf = NULL;

	int outer_iter, outer_it;
	int iteration;
	int performed=0;

	VECTOR_TYPE* rhsVec = NULL;
	VECTOR_TYPE* resCR  = NULL;

	uint64 asm_cycles, asm_cyclecounter;

	double time_it_took;
	int version;
	int numthreads;

	/* Number of nodes */
	int n_nodes;

	/* Error-code for MPI */
	int ierr;

	/* Rank of this node */
	int me;

	/* Memory page size in bytes*/
	const int pagesize=4096;

	uint64 cache_size=0ULL;

	int place_rhs;

	int required_threading_level;
	int provided_threading_level;

	int io_format, work_dist;
	double recal_clockfreq;

	int job_flag;
	int rb_cnt;
	int this_one;


	int me_node;

	MPI_Status status;

	MPI_Request req_vec[3];
	MPI_Status stat_vec[3];

	/****************************************************************************
	 *******            ........ Executable statements ........          ********
	 ***************************************************************************/

	MATRIX_FORMATS matrixFormats;
	matrixFormats.format[0] = SPM_FORMAT_ELR;
	matrixFormats.format[1] = SPM_FORMAT_PJDS;
	matrixFormats.format[2] = SPM_FORMAT_ELR;
	matrixFormats.T[0] = 2;
	matrixFormats.T[1] = 2;
	matrixFormats.T[2] = 1;

	if (matrixFormats.format[1] == matrixFormats.format[2] && matrixFormats.format[1] == SPM_FORMAT_PJDS)
		myabort ("There must NOT be pJDS for the local AND remote part");


	allocatedMem = 0;
	required_threading_level = MPI_THREAD_MULTIPLE;

	ierr = MPI_Init_thread(&nArgs, &arg, required_threading_level, 
			&provided_threading_level );

	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );
	ierr = MPI_Comm_size ( MPI_COMM_WORLD, &n_nodes );

#ifdef _OPENMP
	/* Hier werden die threads gepinnt. Dauert manchmal ein bisschen, schadet
	 *     * hier aber nichts. Auf der cray ist es fuer aprun wichtig, dass zwischen
	 *         * dem MPI_Init_thread und der ersten parallelen Region _keine_ Systemaufrufe
	 *             * wie z.B. dem get_hostname liegen. Sonst verwendet das pinning eine falsche
	 *                 * skip-mask */
#pragma omp parallel private(coreId) shared (numthreads, hostname)
	{
		coreId = likwid_threadGetProcessorId();
#pragma omp critical
		{
			if (thishost(hostname)>49) MPI_Abort(MPI_COMM_WORLD, 999);
			numthreads = omp_get_num_threads();
			printf ("Rank %d Thread (%d/%d) running on node %s core %d \n",
					me, omp_get_thread_num(), numthreads, hostname, coreId);
			fflush(stdout);
		}
	}

#else
	numthreads = 1;
	coreId = likwid_threadGetProcessorId();
	if (thishost(hostname)>49) MPI_Abort(MPI_COMM_WORLD, 999);
	printf ("Rank %d running on node %s core %d \n",
			me, hostname, coreId);
	fflush(stdout);
#endif

	mypid = getpid();
	total_mem = my_amount_of_mem();
	cache_size=cachesize();


	/* get nodal MPI communicator ******************************/
	setupSingleNodeComm( hostname, &single_node_comm, &me_node);
#ifdef OCLKERNEL
	/* select cards on node *******************/
  int node_rank, node_size;

  ierr = MPI_Comm_size( single_node_comm, &node_size);
  ierr = MPI_Comm_rank( single_node_comm, &node_rank);
  	MPI_Barrier( MPI_COMM_WORLD );
  	CL_init( node_rank, node_size, hostname, matrixFormats);
#endif

#ifdef DAXPY
	job_flag = IS_DAXPY;
	sprintf(benchmark, "y=y+A*x");
#else
	job_flag = IS_AX;
	sprintf(benchmark, "y=A*x");
#endif

#ifdef REVBUF
	sprintf(rb_flag, "yes");
#else
	sprintf(rb_flag, "no");
#endif

#ifdef NO_PLACEMENT
	sprintf(pm_flag, "NONE");
#else
	sprintf(pm_flag, "yes");
#endif


#ifdef CMEM
	sprintf(cm_flag, "yes");

	coreId = likwid_processGetProcessorId();
	IF_DEBUG(0) if (coreId==0) 
		printf("PE%d: sweep memory on %s\n", me, hostname);

	sweepMemory(GLOBAL);

	ierr = MPI_Barrier(MPI_COMM_WORLD);
#else
	sprintf(cm_flag, "no");
#endif


#ifdef LIKWID
	sprintf(lw_flag, "yes");
	numberOfThreads = numthreads;
	likwid_markerInit(numberOfThreads, numberOfRegions);
	//   RegionId = likwid_markerRegisterRegion("Main");
#else
	sprintf(lw_flag, "no");
#endif




	if (me == 0){
		N_MULTS    = atoi(arg[1]);
		place_rhs  = atoi(arg[2]);
		io_format  = atoi(arg[4]);
		work_dist  = atoi(arg[5]);
		jobmask    = atoi(arg[6]);
		outer_it   = atoi(arg[7]);

		if      (place_rhs == 1) sprintf(pr_flag, "CRS");
		else if (place_rhs == 2) sprintf(pr_flag, "LNL");
		else                     sprintf(pr_flag, "NONE");

		if      (io_format == 1) sprintf(io_flag, "MPI_IO");
		else if (io_format == 2) sprintf(io_flag, "BINARY");
		else                     sprintf(io_flag, "ASCII");

		if      (work_dist == 1) sprintf(wd_flag, "EQ_NZE");
		else if (work_dist == 2) sprintf(wd_flag, "EQ_LNZE");
		else                     sprintf(wd_flag, "EQ_ROWS");


		mystringlength = strlen(arg[3]);
		if (mystringlength > 49) myabort("testcase longer than field"); 
		strcpy(testcase, arg[3]);

		mystringlength = strlen(arg[0]);
		if (mystringlength > 159) myabort("executable longer than field"); 
		strcpy(this_executable, arg[0]);

		if ( (machname(mach_name)   > 49) || (kernelversion(kernel_version) > 49) ||
				(modelname(model_name) > 49)    ) 
			myabort("some system variable longer than field"); 

		getMatrixPath(testcase,matrixpath);
		if (!matrixpath)
			myabort("Matrix could not be read");

		/* Get overhead of cycles measurement */
		AS_CYCLE_START;
		AS_CYCLE_STOP;
		cycles4measurement = asm_cycles;
		clockfreq = myCpuClockFrequency(); 
		recal_clockfreq = RecalFrequency(cycles4measurement, clockfreq);

		total_mem = my_amount_of_mem();
		mypid = getpid();

		printf("=====================================================\n");
		printf("-------   Architecture and operating system   -------\n");
		printf("-----------------------------------------------------\n");
		printf("Running on machine          : %12s\n", mach_name); 
		printf("Identifier of current run   : %12i\n", mypid); 
		printf("Running kernel version      : %12s\n", kernel_version); 
		printf("CPU-Type                    : %12s\n", model_name); 
		printf("CPUClockFrequency [MHz]     : %12.3f\n", clockfreq/1e6); 
		printf("Frequency rdtsc [MHz]       : %12.3f\n", recal_clockfreq/1e6); 
		printf("Cache size per socket [kB]  : %12llu\n", cache_size); 
		printf("Total memory [kB]           : %12.0f\n", total_mem/1024.0); 
		printf("Number of MPI processes     : %12i\n", n_nodes); 
		printf("Number of OpenMP threads    : %12i\n", numthreads); 
		printf("Value of $KMP_AFFINITY      : %12s\n", getenv("KMP_AFFINITY")); 
		printf("Value of $OMP_SCHEDULE      : %12s\n", getenv("OMP_SCHEDULE")); 
		printf("Value of $LD_PRELOAD        : %12s\n", getenv("LD_PRELOAD")); 
		printf("Value of $I_MPI_DEVICE      : %12s\n", getenv("I_MPI_DEVICE")); 

		switch(provided_threading_level){
			case MPI_THREAD_SINGLE:
				printf("Threading support of MPI    : %12s\n", "MPI_THREAD_SINGLE");
				break;
			case MPI_THREAD_FUNNELED:
				printf("Threading support of MPI    : %12s\n", "MPI_THREAD_FUNNELED");
				break;
			case MPI_THREAD_SERIALIZED:
				printf("Threading support of MPI    : %12s\n", "MPI_THREAD_SERIALIZED");
				break;
			case MPI_THREAD_MULTIPLE:
				printf("Threading support of MPI    : %12s\n", "MPI_THREAD_MULTIPLE");
				break;
		}

		printf("-----------------------------------------------------\n");
		printf("-------        Command line arguments         -------\n");
		printf("-----------------------------------------------------\n");
		printf("Executable                  : %12s\n", this_executable);
		printf("Number of multiplications   : %12i\n", N_MULTS); 
		printf("NUMA-placement of RHS       : %12s\n", pr_flag); 
		printf("NUMA-placement of matrix    : %12s\n", pm_flag); 
		printf("Use of revolving buffers    : %12s\n", rb_flag); 
		printf("Performing initial memsweep : %12s\n", cm_flag); 
		printf("Using LIKWID marker API     : %12s\n", lw_flag); 
		printf("Type of matrix-file I/O     : %12s\n", io_flag); 
		printf("Matrix distribution on PEs  : %12s\n", wd_flag); 
		printf("Jobmask (integer)           : %12d\n", jobmask); 
		printf("Jobmask (hexadecimal)       : %#12x\n", jobmask); 
		printf("Type of benchmark           : %12s\n", benchmark);
#ifdef OCLKERNEL	
		printf("Full matrix format          : %9s-%2d\n", SPM_FORMAT_NAME[matrixFormats.format[0]],matrixFormats.T[0]); 
		printf("Local matrix format         : %9s-%2d\n", SPM_FORMAT_NAME[matrixFormats.format[1]],matrixFormats.T[1]); 
		printf("Remote matrix format        : %9s-%2d\n", SPM_FORMAT_NAME[matrixFormats.format[2]],matrixFormats.T[2]); 
#endif
		printf("-----------------------------------------------------\n");


		/* Kein restart-file in CRS und JDS-Format vorhanden: Lese Matrix im 
		 * Standardformat von MM ein und konvertiere in die entsprechenden 
		 * anderen Formate. Danach Ausgabe der entsprechenden restart files */

		clockfreq = recal_clockfreq;

		/*if (io_format == 1){
			// read-in CR-row numbers in binary format for lateron use 
			cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );

			AS_CYCLE_START;
			pio_read_cr_rownumbers(cr, testcase);
			AS_CYCLE_STOP;
			IF_DEBUG(1) AS_WRITE_TIME("Binary read of CR row numbers");
		}
		else */
		if (!isMMfile(matrixpath)){
			/* binary format *************************************/
			cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );

			/* Binary read of matrix in serial CRS-format */
			AS_CYCLE_START;
			bin_read_cr(cr, matrixpath);
			AS_CYCLE_STOP;
			IF_DEBUG(1) AS_WRITE_TIME("Binary reading of CR");

			/* Write out CR-row numbers in binary format for lateron use */
			AS_CYCLE_START;
			pio_write_cr_rownumbers(cr, testcase);
			AS_CYCLE_STOP;
			IF_DEBUG(1) AS_WRITE_TIME("Binary write of CR row numbers");

			/*printf("unFORTRAN CRS:\n");
			  for(i=0; i < cr->nEnts; ++i) {
			  cr->col[i] -= 1;
			  }
			  sprintf(restartfilename, "%s_unfort", testcase);
			  AS_CYCLE_START;
			  bin_write_cr(cr, restartfilename);
			  AS_CYCLE_STOP;
			  AS_WRITE_TIME("Binary write of CR");*/

			/*#ifdef CMEM
			  if (allocatedMem > 0.02*total_mem){
			  AS_CYCLE_START;
			  IF_DEBUG(1) printf("CR setup: Large matrix -- allocated mem=%8.3f MB\n",
			  (float)(allocatedMem)/(1024.0*1024.0)); 
			  sweepMemory(SINGLE);
			  IF_DEBUG(1) printf("Nach memsweep\n"); fflush(stdout);
			  AS_WRITE_TIME("Flushing buffer cache");
			  }
#endif
			 */
		}
		else{
			/* ascii format *************************************/
			AS_CYCLE_START;
			printf("file: %s \n", matrixpath);
			/* Kein threashold beim Einlesen: nehme alle Elemente komplett mit */
			mm = readMMFile( matrixpath, 0.0 );
			AS_CYCLE_STOP;
			IF_DEBUG(1) AS_WRITE_TIME("Reading of MM");

			/* Convert general MM-matrix to CRS-format */
			AS_CYCLE_START;
			cr = convertMMToCRMatrix( mm );
			AS_CYCLE_STOP;
			IF_DEBUG(1) AS_WRITE_TIME("Setup of CR");

			/* Write out CR-matrix in binary format for lateron use */
			AS_CYCLE_START;
			bin_write_cr(cr, testcase);
			AS_CYCLE_STOP;
			IF_DEBUG(1) AS_WRITE_TIME("Binary write of CR");

			/* Write out CR-row numbers in binary format for lateron use */
			AS_CYCLE_START;
			pio_write_cr_rownumbers(cr, testcase);
			AS_CYCLE_STOP;
			IF_DEBUG(1) AS_WRITE_TIME("Binary write of CR row numbers");

			/* Free memory for MM matrix */
			freeMMMatrix(mm);
			printf("memory for MM deallocated\n"); fflush(stdout);
		}

		if (io_format != 1){

			/* convert column indices in CRS format to FORTRAN-numbering, required for CPU kernel */
			crColIdToFortran(cr);

			rhsVec = newVector( cr->nCols );
			resCR = newVector( cr->nCols );

			IF_DEBUG(1) printf("Vectors allocated\n"); fflush(stdout);

			/* Initialisiere invec */
			for (i=0; i<cr->nCols; i++) rhsVec->val[i] = i+1;

			/* Serial CRS-multiplication to get reference result */
			fortrancrs_(&(cr->nRows), &(cr->nEnts), 
					resCR->val, rhsVec->val, cr->val , cr->col, cr->rowOffset);
		}

		acc_cycles = 0.0;
		acc_time = 0.0;

	} // end if me==0
	else{

		IF_DEBUG(1) printf("PE%d vor Allokation\n", me);fflush(stdout);

		/* Allokiere minimalen Speicher fuer Dummyversion der globalen Matrix */
		total_mem = my_amount_of_mem();
		mm            = (MM_TYPE*) allocateMemory( sizeof(MM_TYPE), "mm" );
		cr            = (CR_TYPE*) allocateMemory( sizeof(CR_TYPE), "cr" );
		cr->nRows     = 0;
		cr->nEnts     = 1;
		cr->rowOffset = (int*)     allocateMemory( sizeof(int),     "rowOffset" );
		cr->col       = (int*)     allocateMemory( sizeof(int),     "col" );
		cr->val       = (double*)  allocateMemory( sizeof(double),  "val" );
		rhsVec = newVector( 1 );
		resCR  = newVector( 1 );

		AS_CYCLE_START;
		AS_CYCLE_STOP;
		cycles4measurement = asm_cycles;
		clockfreq = myCpuClockFrequency(); 

		IF_DEBUG(1) printf("PE%d nach Allokation\n", me);fflush(stdout);
	}

	IF_DEBUG(1) printf("PE%d nach Einleseregion\n", me);fflush(stdout);

	/* Distribute command line parameters to all PEs */
	ierr = MPI_Bcast(&N_MULTS,   1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(&jobmask,   1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(&place_rhs, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(&outer_it,  1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(&work_dist, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(&io_format, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	ierr = MPI_Bcast(testcase, 50, MPI_CHAR, 0, MPI_COMM_WORLD);

	/* Gerald's old parallel code, not working?
	   if (io_format == 1){
	   PAS_CYCLE_START
	//lcrp = parallel_MatRead(testcase, work_dist);
	lcrp = new_pio_read(testcase, work_dist);
	PAS_CYCLE_STOP;
	PAS_WRITE_TIME("Parallel reading of CR & setup");
	}

	IF_DEBUG(1) printf("PE%d nach parallelem file-I/O\n", me);fflush(stdout);
	 */
	/* Determine overhead of parallel time measurement */
	ierr = MPI_Barrier(MPI_COMM_WORLD);


	IF_DEBUG(2) printf("PE%d nach first Barrier\n", me);fflush(stdout);
	/* zweite barrier aufgrund des defaults der Umgebungsvariable
	 * I_MPI_DYNAMIC_CONNECTION noetig. Sonst braucht die erste
	 * Zeitmessung wesentlich laenger, d.h. der so gemessene overhead
	 * wird deutlich (Faktor 100!) ueberschaetzt.
	 * Mit der doppelten barrier davor kann ich dann auch gleich 
	 * die erste Messung verwenden -- hier nehme ich trotzdem die 10te*/
	ierr = MPI_Barrier(MPI_COMM_WORLD);

	for(i=0;i<10;i++){
		if (me == 0) for_timing_start_asm_( &asm_cyclecounter); 
		ierr = MPI_Barrier(MPI_COMM_WORLD);
		ierr = MPI_Barrier(MPI_COMM_WORLD);			      
		if (me == 0){
			for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);	
			p_cycles4measurement = asm_cycles;
			time_it_took =  (1.0*asm_cycles)/clockfreq; 
			IF_DEBUG(2) printf("Versuch %d: p_cycles4measurement: %llu cycles, %12.3f ms\n", 
					i, p_cycles4measurement, 1000*time_it_took);
		}
	}

#ifdef CMEM
	PAS_CYCLE_START;
	IF_DEBUG(1) printf("CR setup: Large matrix -- allocated mem=%8.3f MB\n",
			(float)(allocatedMem)/(1024.0*1024.0));
	sweepMemory(GLOBAL);
	IF_DEBUG(1) printf("Nach memsweep\n"); fflush(stdout);
	PAS_WRITE_TIME("Flushing buffer cache");
#endif


	/* Setup of communication pattern between all PEs */
	//   ierr= MPI_Barrier(MPI_COMM_WORLD); if (me==0) printf("before setup_communication\n");
	PAS_CYCLE_START;
	if(io_format!=1) {
		lcrp = setup_communication(cr, work_dist,matrixFormats);
	}
	else {
		lcrp = setup_communication_parallel(cr, work_dist, testcase);
	}
	IF_DEBUG(1) PAS_WRITE_TIME("Setup of Communication");


#ifdef OCLKERNEL
  	MPI_Barrier( MPI_COMM_WORLD );
	if( jobmask & 503 ) { 
		CL_bindMatrixToKernel(lcrp->fullMatrix,lcrp->fullFormat,matrixFormats.T[0],0);

	} 
	if( jobmask & 261640 ) { // only if jobtype requires split computation
		CL_bindMatrixToKernel(lcrp->localMatrix,lcrp->localFormat,matrixFormats.T[1],1);
		CL_bindMatrixToKernel(lcrp->remoteMatrix,lcrp->remoteFormat,matrixFormats.T[2],2);
	}
#endif




	/* Free memory for CR stored matrix and sweep memory */
	/* freeCRMatrix( cr ); moved to setup_communication */

	/* sollte nicht noetig sein. Im Gegensatz zum buffer-cache hilft ein richtiges free
#ifdef CMEM
sweepMemory(GLOBAL);
#endif
	 */
	lcrp->threads = numthreads; 

#ifndef FAST_EXIT
	/* Diagnostic output corresponding to given output level */
	/*PAS_CYCLE_START;
	  check_lcrp(me, lcrp);
	  PAS_WRITE_TIME("Check of lcrp");
	 */
#endif

	/****************************************************************************
	 **********      Setup of right-hand-side and result vector        ********** 
	 ***************************************************************************/

	PAS_CYCLE_START;

	//NUMA_CHECK("before placement of RHS & Solution");

	pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;
	//size = (size_t)( lcrp->lnRows[me] * sizeof(double) );
	//hlpvec_out = (double*) allocateMemory(size, "hlpvec_out");
	hlpvec_out = newVector( lcrp->lnRows[me] );

	//#pragma omp parallel for schedule(runtime)
#pragma omp parallel for schedule(static)
	for (i=0; i<lcrp->lnRows[me]; i++) hlpvec_out->val[i] = -63.5;

	hlpvec_in = newVector( pseudo_ldim );  

#ifdef REVBUF

	/* Setup revolving buffer */
	RevBuf = revolvingBuffer( cache_size, pagesize, pseudo_ldim );
	free( hlpvec_in->val );

	for (rb_cnt=0; rb_cnt<RevBuf->numvecs; rb_cnt++){
		hlpvec_in->val = RevBuf->vec[rb_cnt];
		IF_DEBUG(2) printf("PE%i: Adresse von hlpvec: %p\n",me, hlpvec_in->val); fflush(stdout);
		if       (place_rhs == 1){
			//#pragma omp parallel for schedule(runtime)
#pragma omp parallel for schedule(static)
			for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in->val[i] = 0.0;
		}
		if       (place_rhs == 2){
#pragma omp parallel for schedule(runtime)
			for( i = 0; i < lcrp->lnRows[me]; i++ ) hlpvec_in->val[i] = 0.0;
#pragma omp parallel for schedule(runtime)
			for( i = lcrp->lnRows[me]; i < pseudo_ldim; i++ ) hlpvec_in->val[i] = 0.0;
		}
		else{
			for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in->val[i] = 0.0;
		}
	}

	/*IF_DEBUG(1) printf("PE%i: pre-scatter\n\trecvbuff: %p\n\trecvcnt: %i ( %f MB )\n\troot: %i\n", 
	  me, RevBuf->vec[0], lcrp->lnRows[me], (double)lcrp->lnRows[me]*sizeof(double)/(1024*1024), 
	  0); fflush(stdout);
	 */


	/*IF_DEBUG(1) if(0==me) {
	  printf("PE%i: scattering from: %p\n", me, rhsVec->val);
	  for(i=0; i < n_nodes; ++i) { 
	  printf("PE%i: scattering to %i:\tsendcnt: %i ( %f MB )\tdispls: %i\n",
	  me, i, lcrp->lnRows[i], (double)lcrp->lnRows[i]*sizeof(double)/(1024*1024), 
	  lcrp->lfRow[i]); fflush(stdout);
	  for(j=0; j<10 && j<lcrp->lnRows[i];++j) printf("PE%i: to %i Buffer[%i]=%f\n",me,i,j,rhsVec->val[lcrp->lfRow[i]+j]);
	  }
	  }
	 */

	MPI_Comm_size( MPI_COMM_WORLD, &i );
	IF_DEBUG(1) printf("PE%i: pre-scattering: n_nodes=%i, lcrp->nodes=%i\n",
			me, i, lcrp->nodes); fflush(stdout);


	/* Scatter the input vector from the master node to all others */
	ierr = MPI_Scatterv ( rhsVec->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
			RevBuf->vec[0], lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

	// plain MPI alternative to Scatterv
	/* 
	   if( 0==me) {
	   j=0;
	   for(i=1; i<n_nodes; ++i) {
	   j= j >  lcrp->lnRows[i] ? j : lcrp->lnRows[i];
	   }
	   printf("PE%i: maxlength=%i\n", me, j);
	   scatterBuf = (double*)malloc(sizeof(double)*j);

	   for(i=n_nodes-1; i>0; --i) {
	   IF_DEBUG(1) { printf("PE%i: to %i: %i entries, %i offset, %i rhs length, from adress %p\n",
	   me,i,lcrp->lnRows[i],lcrp->lfRow[i],rhsVec->nRows, &rhsVec->val[lcrp->lfRow[i]] ); fflush(stdout); 
	   }
	   for(j=0; j<lcrp->lnRows[i]; ++j) {
	   scatterBuf[j] = rhsVec->val[lcrp->lfRow[i]+j];
	   }
	   int k = lcrp->lnRows[i];
	   ierr = MPI_Send(scatterBuf, k, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
	   printf("PE%i: ierr=%i\n",me,ierr);fflush(stdout);
	//ierr = MPI_Isend(&rhsVec->val[lcrp->lfRow[i]], lcrp->lnRows[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &req_vec[i-1]);
	}

	IF_DEBUG(1) { printf("PE%i: lnRows[%i]=%i\n",me,me,lcrp->lnRows[me]); fflush(stdout); }

	//MPI_Waitall(n_nodes-1, req_vec, stat_vec);

	IF_DEBUG(1) { printf("PE%i: finished sending\n",me); fflush(stdout);  }
	for(i=0; i<lcrp->lnRows[me]; ++i) {
	RevBuf->vec[0][i] = rhsVec->val[i];
	}
	} else {
	ierr = MPI_Recv(RevBuf->vec[0], lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	IF_DEBUG(1) { printf("PE%i: received %i entries\n",me,lcrp->lnRows[me]); fflush(stdout); }
	}
	 */

	IF_DEBUG(1) printf("PE%i: scattered - status: %i\n",me, ierr); fflush(stdout);
	IF_DEBUG(1) for(i=0; i<10 && i<lcrp->lnRows[me]; ++i) printf("PE%i: local Revbuf[0][%i]=%f\n",
			me,i,RevBuf->vec[0][i]);fflush(stdout);
	IF_DEBUG(1) for(i=lcrp->lnRows[me]-10; i<lcrp->lnRows[me]; ++i) printf("PE%i: local Revbuf[0][%i]=%f\n",
			me,i,RevBuf->vec[0][i]);fflush(stdout);
	IF_DEBUG(2) for(i=0; i<lcrp->lnRows[me];++i) printf("PE%i: local RevBuf[0][%i]=%f\n",me,i,RevBuf->vec[0][i]);fflush(stdout);

	/* Copy locally the invec to all vectors in RevBuf */
	for (rb_cnt=1; rb_cnt<RevBuf->numvecs; rb_cnt++){
		for (i=0; i<lcrp->lnRows[me]; i++) 
			RevBuf->vec[rb_cnt][i] = RevBuf->vec[0][i];
	}

	/* Fill up halo with some markers */
	for (rb_cnt=0; rb_cnt<RevBuf->numvecs; rb_cnt++){
		for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++)
			RevBuf->vec[rb_cnt][i] = 77.0;
	}


#else



	/* Placement of RHS Vector */
	if       (place_rhs == 1){
#pragma omp parallel for schedule(runtime)
		for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in->val[i] = 0.0;
	}
	if       (place_rhs == 2){
#pragma omp parallel for schedule(runtime)
		for( i = 0; i < lcrp->lnRows[me]; i++ ) hlpvec_in->val[i] = 0.0;
#pragma omp parallel for schedule(runtime)
		for( i = lcrp->lnRows[me]; i < pseudo_ldim; i++ ) hlpvec_in->val[i] = 0.0;
	}
	else{ 
		for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in->val[i] = 0.0;
	}

	/* Scatter the input vector from the master node to all others */
	ierr = MPI_Scatterv ( rhsVec->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
			hlpvec_in->val, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

	/* Fill up halo with some markers */
	for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) hlpvec_in->val[i] = 77.0;

#endif

	IF_DEBUG(1) { printf("PE%i: donedonedone\n", me);fflush(stdout); }
	//NUMA_CHECK("after placement of RHS & Solution");

#ifdef OCLKERNEL	
	size_t fullMemSize, localMemSize, remoteMemSize, 
		   totalFullMemSize = 0, totalLocalMemSize = 0, totalRemoteMemSize = 0;
	if( jobmask & 503 ) { 
		fullMemSize = getBytesize(lcrp->fullMatrix,lcrp->fullFormat)/(1024*1024);
		MPI_Reduce(&fullMemSize, &totalFullMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);

	} 
	if( jobmask & 261640 ) { // only if jobtype requires split computation
		localMemSize = getBytesize(lcrp->localMatrix,lcrp->localFormat)/(1024*1024);
		remoteMemSize = getBytesize(lcrp->remoteMatrix,lcrp->remoteFormat)/(1024*1024);
		MPI_Reduce(&localMemSize, &totalLocalMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&remoteMemSize, &totalRemoteMemSize,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
	}
#endif	

	if(me==0){
		ws = ((lcrp->nRows+1)*sizeof(int) + lcrp->nEnts*(sizeof(double)+sizeof(int)))/(1024*1024);
		printf("-----------------------------------------------------\n");
		printf("-------         Statistics about matrix       -------\n");
		printf("-----------------------------------------------------\n");
		printf("Investigated matrix         : %12s\n", testcase); 
		printf("Dimension of matrix         : %12.0f\n", (float)lcrp->nRows); 
		printf("Non-zero elements           : %12.0f\n", (float)lcrp->nEnts); 
		printf("Average elements per row    : %12.3f\n", (float)lcrp->nEnts/(float)lcrp->nRows); 
		printf("Working set             [MB]: %12lu\n", ws);
#ifdef OCLKERNEL	
		if( jobmask & 503 ) 
			printf("Device matrix (combin.) [MB]: %12lu\n", totalFullMemSize); 
		if( jobmask & 261640 ) {
			printf("Device matrix (local)   [MB]: %12lu\n", totalLocalMemSize); 
			printf("Device matrix (remote)  [MB]: %12lu\n", totalRemoteMemSize); 
			printf("Device matrix (loc+rem) [MB]: %12lu\n", totalLocalMemSize+totalRemoteMemSize); 
		}
#endif
		printf("-----------------------------------------------------\n");
		printf("------   Hybrid SpMVM using kernel version     ------\n");
		printf("-----------------------------------------------------\n");
		fflush(stdout);
	}






	MPI_Barrier(MPI_COMM_WORLD);

	for (outer_iter=0; outer_iter<outer_it; outer_iter++){

		if (me==0){
			printf("*****************************\n");
			printf("**** Outer iteration %3d ****\n", outer_iter);
			printf("*****************************\n");
		}

		/****************************************************************************
		 ********************    Loop over all kernel versions   ******************** 
		 ***************************************************************************/
		MPI_Barrier(MPI_COMM_WORLD);
		for (version=0; version<NUMKERNELS; version++){

			/* Skip this version if kernel is not chosen by jobmask */
			if ( ((0x1<<version) & jobmask) == 0 ) continue; 

			/* Skip loop body if version does not make sense for used parametes */
			if (version==0 && lcrp->nodes>1) continue;      /* no MPI available */
			if (version>10 && version < 17 && lcrp->threads==1) continue; /* not enough threads */

			for( iteration = 0; iteration < N_MULTS+1; iteration++ ) {

#ifdef REVBUF
				/* choose current input vector from revolving buffer */
				this_one = iteration%RevBuf->numvecs;
				hlpvec_in->val = RevBuf->vec[this_one];
#ifdef OCLKERNEL
				IF_DEBUG(2) CL_vectorDeviceCopyCheck( hlpvec_in, me );
				IF_DEBUG(2) CL_vectorDeviceCopyCheck( hlpvec_out, me );
#endif
#endif

				/* Timing starts after the initialisation iteration */
				if (iteration==1) {
#ifdef LIKWID
					RegionId = likwid_markerRegisterRegion("HyK_kernel");                
#pragma omp parallel private(threadId, coreId)
					{                                                                 
						threadId = omp_get_thread_num();                               
						if (threadId == 0) coreId = likwid_processGetProcessorId();
						else               coreId = likwid_threadGetProcessorId();                        

						likwid_markerStartRegion(threadId, coreId);                    
					} 
#else
					PAS_CYCLE_START;
					//AS_CYCLE_START;
#endif
				}

				/* call to multiplication kernel */
				HyK[version].kernel( iteration, hlpvec_out, lcrp, hlpvec_in);

				/* To prevent an interleaving of communication and computation
				 * for two successive multiplications */
				MPI_Barrier(MPI_COMM_WORLD);

			}

#ifdef LIKWID  
#pragma omp parallel private(threadId, coreId)
			{                                                                 
				threadId = omp_get_thread_num();                               
				coreId = likwid_threadGetProcessorId();                        
				likwid_markerStopRegion(threadId, coreId, RegionId);           
			}                                                                 
#else
			PAS_WRITE_TIME(HyK[version].tag);
			IF_DEBUG(1) if (me==0) printf("%s\n", HyK[version].name);

			if (me==0){ 
				HyK[version].cycles = (double) asm_cycles;
				HyK[version].time   = time_it_took; 
			}
			
			IF_DEBUG(1) PAS_CYCLE_START;
			if ( ((0x1<<version) & 503) ) {
				permuteVector(hlpvec_out->val,lcrp->fullInvRowPerm,lcrp->lnRows[me]);
			} else if ( ((0x1<<version) & 261640) ) {
				permuteVector(hlpvec_out->val,lcrp->splitInvRowPerm,lcrp->lnRows[me]);
			}
			IF_DEBUG(1){PAS_WRITE_TIME("Permuting back vectors (if necessary)");}

			/* Perform correctness check once for each kernel version */ 
			performed++;
			IF_DEBUG(1) PAS_CYCLE_START;


			Correctness_check( resCR, lcrp, hlpvec_out->val );

			IF_DEBUG(1){PAS_WRITE_TIME("Correctness-Check");}
#endif
		}

#ifdef LIKWID
		likwid_markerClose();
		MPI_Finalize();
		return 0;
#endif


		if (me==0){
			printf("-------------------------------------------------------\n");
			printf("performed (%d/%d) successful error-checks\n", successful, performed);
			printf("-------------------------------------------------------\n");
			printf("serial time measurement [us]: %12.3f\n", (1e6*cycles4measurement)/clockfreq);
			printf("global sync & time      [us]: %12.3f\n", (1e6*p_cycles4measurement)/clockfreq);
			printf("Number of iterations        : %12.0f\n", 1.0*N_MULTS);
			printf("-------------------------------------------------------\n");
			printf("Kernel            Cyc/NZE  Time/MVM [ms]        MFlop/s\n"); 
			for (version=0; version<NUMKERNELS; version++){
				if ( ((0x1<<version) & jobmask) == 0 ) continue; 
				acc_cycles = (double) HyK[version].cycles;
				acc_time   = HyK[version].time;
				/* Skip loop body if version does not make sense for used parametes */
				if (version==0 && lcrp->nodes>1) continue;      /* no MPI available */
				if (version>10 && version<17 && lcrp->threads==1) continue; /* not enough threads */
				printf("Kern No. %3d %12.2f %14.2f %14.2f\n", 
						version, acc_cycles/((double)N_MULTS*(double)lcrp->nEnts), 
						1000*acc_time/((double)N_MULTS), 
						2.0e-6*(double)N_MULTS*(double)lcrp->nEnts/acc_time);
			}
			printf("=======================================================\n");

#ifdef INDIVIDUAL
			sprintf(statfilename, "./PerfStat_HyK%d_p%d_t%d_%s_%s.dat", 
					version, n_nodes, numthreads, testcase, mach_name);

			if ((STATFILE = fopen(statfilename, "w"))==NULL)
				mypaborts("Fehler beim Oeffnen von", statfilename);

			fprintf(STATFILE,"#%4d %19.5lf\n", 0, perf_array[0]);
			fprintf(STATFILE,"#----------------------------------------\n");
			for (i=1;i<N_MULTS+1;i++) fprintf(STATFILE,"%4d %19.5lf\n", i, perf_array[i]); 
			fclose(STATFILE);
#endif
		}

	} // end outer iteration

	MPI_Barrier(MPI_COMM_WORLD);

	IF_DEBUG(2) printf("Debug-free vor allem\n");fflush(stdout);
	freeVector( hlpvec_out );
#ifdef REVBUF
	hlpvec_in->val = (double*) allocateMemory(sizeof(double), "proforma hlpvec_in");
#endif
	freeVector( hlpvec_in );
	freeLcrpType( lcrp );
	freeVector( rhsVec );
	freeVector( resCR );
	/*
	//  printf("Debug-free: freed rhsVec\n");fflush(stdout);
	//  printf("Debug-free: freed resCR\n");fflush(stdout);
	freeCRMatrix( cr );
	// printf("Debug-free: freed cr\n");fflush(stdout);
	 */
	MPI_Finalize();

#ifdef OCLKERNEL
	CL_finish();
#endif

	return 0;
}
