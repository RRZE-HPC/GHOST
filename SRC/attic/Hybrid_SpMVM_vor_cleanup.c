#include "matricks.h"
#include <string.h>
#include "timing.h"
#include <math.h>
#include <sys/times.h>
#include <unistd.h>
#include <omp.h>
#include <sched.h>

#include <mpi.h>
#include <likwid.h>


static void actcrs_(int*, int*, double*, double*, double*, int*, int*);
static uint64 sp_mvm_timing_asm(int*, int*, double*, double*, double*, int*, int*);
static uint64 get_timing_overhead_asm();
static void sp_mvm_asm(int*, int*, double*, double*, double*, int*, int*);
static void do_stats_(int*, double*, int*, double*, double*, double*, double*, double*, double*);

int ierror;
int pseudo_ldim;
double the_cycles, the_time;
double acc_cycles, acc_time;
double mytol;

double tar[100], car[1000];

void myblockjds_resorted_(int*, int*, int*, int*, int*, int*, double*, double*, double*, double*);
//void myblockjds_resorted_(int*, double*, double*, double*, int*, int*);

void myblockjds_asm(int*, double*, double*, double*, int*, int*);

/*void testfomp_(void);*/
extern void readCycles(uint64* );
float myCpuClockFrequency();
unsigned long machname(char* );
unsigned long kernelversion(char* );
unsigned long modelname(char* );
uint64 cachesize(void);

int error_count, acc_error_count;

double *hlpvec_in, *hlpvec_out, *hlpres_serial;

double *perf_array;

uint64 cycles, counter;
void myJDS_pure_asm(int*, double*, double*, double*, int*, int*, int*, int*);

const double EPSILON = 1e-6;

int coreId=2;
#ifdef LIKWID
   int RegionId;
   int numberOfThreads = 1;
   int numberOfRegions = 1;
   int threadId = 0;
#endif


int main( int nArgs, char* arg[] ) {
   /* const int N_MULTS = 100; */
   int N_MULTS, mode, blocklen, oldblocklen;
   int N_TRIES=100;
   int outlev = 1;
   int mypid;
   double fblocklen;
   const double EPSILON = 1e-6;
   double mean, stdev, runaway;
   double ess_mean, ess_stdev, ess_run;
   float mybytes, mycycles;
   int i, rw_flag, hlpi;
   double startTime, stopTime, ct, req_time, num_flops, ws;
   unsigned long mystringlength;
   double nnz1, wdim1;
   struct tms dummy;
   FILE *RESTFILE;
   FILE *STATFILE;
   char statfilename[50];
   char restartfilename[50];
   char testcase[50];
   char mach_name[50];
   char kernel_version[50];
   char hostname[50];
   char model_name[50];
   MM_TYPE* mm = NULL;
   CR_TYPE* cr = NULL;
   CR_P_TYPE* cr_parallel = NULL;
   LCRP_TYPE* lcrp = NULL;

   REVBUF_TYPE* RevBuf = NULL;

   int iteration;


   JD_TYPE* jd = NULL;
   JD_RESORTED_TYPE* jdr = NULL;
   JD_OFFDIAGONAL_TYPE* jdo = NULL;
   VIP_TYPE* vip = NULL;
   VECTOR_TYPE* res_hybrid = NULL;
   VECTOR_TYPE* rhsVec = NULL;
   VECTOR_TYPE* resMM  = NULL;
   VECTOR_TYPE* resCR  = NULL;
   VECTOR_TYPE* resJD  = NULL;
   uint64 mystart=0ULL, mystop=0ULL, overhead;
   uint64 sumcycles=0ULL, cycles=0ULL, test=0ULL, min=1000000000ULL;
   uint64 mincycles=100000000000000ULL;
   uint64 asm_cycles, asm_cyclecounter;
   uint64 tmp_cycles, tmp_cyclecounter;

   double time_it_took;
   
   int version;

   int nd_worth;
   int block_start, block_end;
   int ib, diag,diagLen,offset;
   int aktblock;

   int errcount;
   int numthreads;

   int myblockwidth, myblocklength, thisblocklength, myblockcount, numblocks, j, blocksize, acclines;
   int* jd_resorted_col;
   int* blockdim_rows;
   int*blockdim_cols;
   double* jd_resorted_val;

   int ecount, elements, totalblocks, aktpos, oldpos, l, k, accelements;
   int* jd_resorted_blockptr;

   /* Number of nodes */
   int n_nodes;

   /* Number of treads per node */
   int n_threads;

   /* Error-code for MPI */
   int ierr;

   /* Rank of this node */
   int me;

   /* Memory page size in bytes*/
   const int pagesize=4096;

   uint64 cache_size=0ULL;
   

   char tmp_string[50];
   char this_executable[160];

   double* statistics;
   int* essential_flag;
   void test_omp_();
   void test_comp();
   void analyse_matrix_(int*, int*, int*, int*, int*);
   double my_amount_of_mem();
   int place_rhs;

   unsigned int cpusetsize;
   cpu_set_t* cpumask;

   int required_threading_level;
   int provided_threading_level;

   char benchmark[50];
   char rb_flag[50];
   char cm_flag[50];
   char lw_flag[50];
   char pr_flag[50];
   char pm_flag[50];

   double recal_clockfreq;

   int job_flag;
   int rb_cnt;
   int this_one;
   int size0 = 0;
   int size1 = 0;
   int free0 = 0;
   int free1 = 0;

#ifdef LIKWID
   int RegionId;
   int coreId = 2;
   int numberOfThreads = 1;
   int numberOfRegions = 1;
   int threadId = 0;
#endif

#ifdef CMEM
   uint64 num_fill, memcounter;
   double *tmparray;
#endif

 
   //   MPI_Init(&nArgs, &arg);

   required_threading_level = MPI_THREAD_MULTIPLE;

   ierr = MPI_Init_thread(&nArgs, &arg, required_threading_level, &provided_threading_level );


   ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );
   ierr = MPI_Comm_size ( MPI_COMM_WORLD, &n_nodes );

   mypid = getpid();
   cache_size=cachesize();
   //int sched_getaffinity(pid_t pid, unsigned int cpusetsize, cpu_set_t *mask);
   // ierr = sched_getaffinity(mypid, cpusetsize, cpumask);

   //   printf("PE %d: PID: %d, cpusetsize: %d, cpumask: %d", me, mypid, cpusetsize, cpumask);


/** Hier werden die threads gepinnt. Dauert manchmal ein bisschen, schadet hier aber nichts */
#ifdef _OPENMP
#pragma omp parallel 
   numthreads = omp_get_num_threads();
#else
   numthreads = 1;
#endif

#ifdef DAXPY
      job_flag = IS_DAXPY;
      sprintf(benchmark, "DAXPY: y=y+A*x");
#else
      job_flag = IS_AX;
      sprintf(benchmark, "pure multiply: y=A*x");
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
      total_mem = my_amount_of_mem();

      coreId = likwid_processGetProcessorId();
      if (thishost(hostname)>49) MPI_ABORT(999, MPI_COMM_WORLD);
      IF_DEBUG(1) printf("PE%d: hostname=%s coreId:%d\n", me, hostname, coreId);

  
      if (coreId==0){ /* nur ein MPI-Prozess pro Knoten macht memsweep */
	 num_fill = (uint64)(0.9*total_mem/(8.0));
	 IF_DEBUG(1){
	    printf("PE%d cleans up memory for host %s...\n", me, hostname); 
	    printf("Total memory (/cpu/meminfo): %10.0f kB\n", total_mem/(1024.0));
	    printf("Allocate %llu doubles to fill 90%% of memory\n", num_fill);
	 }
	 tmparray = (double*) malloc(num_fill * sizeof( double ));
	 for (i=0; i<num_fill; i++) tmparray[i]=0.0;
	 IF_DEBUG(1) printf("Freeing memory again ...\n"); 
	 free (tmparray);
	 IF_DEBUG(1) printf("... done\n"); 
	 IF_DEBUG(0){
	    if ( get_NUMA_info(&size0, &free0, &size1, &free1) != 0 ) 
	       myabort("failed to retrieve NUMA-info");
	    printf("PE%d: After memsweep: NUMA-LD-0: %5d (%5d )MB free\n", me, free0, size0);
	    printf("PE%d:              -- NUMA-LD-1: %5d (%5d )MB free\n", me, free1, size1);
	 }
      }
      ierror = MPI_Barrier(MPI_COMM_WORLD);
#else
      sprintf(cm_flag, "no");
#endif


#ifdef LIKWID
   sprintf(lw_flag, "yes");
   numberOfThreads = numthreads;
   likwid_markerInit(numberOfThreads, numberOfRegions);
   RegionId = likwid_markerRegisterRegion("Main");
#else
   sprintf(lw_flag, "no");
#endif



   if (me == 0){

      if ( nArgs != 4 ) myabort("expect input: [N_MULTS] [NUMA-placement RHS] [testcase]"); 

      N_MULTS    = atoi(arg[1]);
      place_rhs  = atoi(arg[2]);

      if      (place_rhs == 1) sprintf(pr_flag, "CRS");
      else if (place_rhs == 2) sprintf(pr_flag, "LNL");
      else                     sprintf(pr_flag, "NONE");

      mystringlength = strlen(arg[3]);
      if (mystringlength > 49) myabort("testcase longer than field"); 
      strcpy(testcase, arg[3]);

      mystringlength = strlen(arg[0]);
      if (mystringlength > 159) myabort("executable longer than field"); 
      strcpy(this_executable, arg[0]);

      if ( (machname(mach_name)   > 49) || (kernelversion(kernel_version) > 49) ||
	    (modelname(model_name) > 49)    ) 
	 myabort("some system variable longer than field"); 

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
      printf("Type of benchmark           : %12s\n", benchmark); 


      /* Kein restart-file in CRS und JDS-Format vorhanden: Lese Matrix im 
       * Standardformat von MM ein und konvertiere in die entsprechenden 
       * anderen Formate. Danach Ausgabe der entsprechenden restart files */

      clockfreq = recal_clockfreq;

      AS_CYCLE_START
	 sprintf(restartfilename, "./daten/%s.mtx", testcase);
      mm = readMMFile( restartfilename, 0.0 );
      AS_CYCLE_STOP;

      ws = (mm->nRows*20.0 + mm->nEnts*12.0)/(1024*1024);
      printf("-----------------------------------------------------\n");
      printf("-------         Statistics about matrix       -------\n");
      printf("-----------------------------------------------------\n");
      printf("Investigated matrix         : %12s\n", testcase); 
      printf("Dimension of matrix         : %12.0f\n", (float)mm->nRows); 
      printf("Non-zero elements           : %12.0f\n", (float)mm->nEnts); 
      printf("Average elements per row    : %12.3f\n", (float)mm->nEnts/(float)mm->nRows); 
      printf("Working set [MB]            : %12.3f\n", ws); 
      printf("-----------------------------------------------------\n");
      printf("------   Setup matrices in different formats   ------\n");
      printf("-----------------------------------------------------\n");

      AS_WRITE_TIME("Reading of MM");

      if( ! mm ) {
	 printf( "main: couldn't load matrix file\n" );
	 return 1;
      }

      IF_DEBUG(1){
	 printf( "main: Anzahl der Matrixeintraege = %i \n", mm->nEnts );
	 printf("N_MULTS = %i \n", N_MULTS);
      }

      rhsVec = newVector( mm->nCols );
      resMM  = newVector( mm->nCols );
      resCR  = newVector( mm->nCols );
      resJD  = newVector( mm->nCols );
      res_hybrid  = newVector( mm->nCols );


      AS_CYCLE_START;
      cr = convertMMToCRMatrix( mm );
      AS_CYCLE_STOP;
      AS_WRITE_TIME("Setup of CR");

      freeMMMatrix(mm);

      /* Initialisiere invec */
      for (i=0; i<cr->nCols; i++) rhsVec->val[i] = i+1;

      /* Serial CRS-multiplication to get reference result */
      fortrancrs_(&(cr->nRows), &(cr->nEnts), 
	    resCR->val, rhsVec->val, cr->val , cr->col, cr->rowOffset);

      perf_array = (double*) allocateMemory( (N_MULTS+1) *sizeof( double ), "perf_array" );

      acc_cycles = 0.0;
      acc_time = 0.0;
     
      printf("Null ist durch\n");

   }
   else{

      /* Allokiere minimalen Speicher fuer Dummyversion der globalen Matrix */
      total_mem = my_amount_of_mem();
      mm = (MM_TYPE*) allocateMemory( sizeof( CR_TYPE ), "mm" );
      cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
      cr->rowOffset = (int*) allocateMemory(  sizeof( int ), "rowOffset" );
      cr->col = (int*) allocateMemory( sizeof( int ), "col" );
      cr->val = (double*) allocateMemory( sizeof( double ), "val" );
      perf_array = (double*) allocateMemory( sizeof( double ), "perf_array" );
      rhsVec = newVector( 1 );
      resCR  = newVector( 1 );

      AS_CYCLE_START;
      AS_CYCLE_STOP;
      cycles4measurement = asm_cycles;
      clockfreq = myCpuClockFrequency(); 

      printf("Off ist durch %d \n", me);
   }

   ierror = MPI_Barrier(MPI_COMM_WORLD);
   printf("PE%d: nach split\n", me ); fflush(stdout);
   ierror = MPI_Barrier(MPI_COMM_WORLD);

   ierror = MPI_Bcast(&N_MULTS, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

   /* Determine overhead of parallel time measurement */

   ierror = MPI_Barrier(MPI_COMM_WORLD);
   /* zweite barrier aufgrund des defaults der Umgebungsvariable
    * I_MPI_DYNAMIC_CONNECTION noetig. Sonst braucht die erste
    * Zeitmessung wesentlich laenger, d.h. der so gemessene overhead
    * wird deutlich (Faktor 100!) ueberschaetzt.
    * Mit der doppelten barrier davor kann ich dann auch gleich 
    * die erste Messung verwenden -- hier nehme ich trotzdem die 10te*/
   ierror = MPI_Barrier(MPI_COMM_WORLD);

   for(i=0;i<10;i++){
      if (me == 0) for_timing_start_asm_( &asm_cyclecounter); 
      ierror = MPI_Barrier(MPI_COMM_WORLD);
      ierror = MPI_Barrier(MPI_COMM_WORLD);			      
      if (me == 0){
	 for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);	
	 p_cycles4measurement = asm_cycles;
	 time_it_took =  (1.0*asm_cycles)/clockfreq; 
	 IF_DEBUG(2) printf("Versuch %d: p_cycles4measurement: %llu cycles, %12.3f ms\n", 
	       i, p_cycles4measurement, 1000*time_it_took);
      }
   }

   ierror = MPI_Barrier(MPI_COMM_WORLD);
   printf("PE%d: vor setup_comm\n", me );
   ierror = MPI_Barrier(MPI_COMM_WORLD);

   PAS_CYCLE_START;
   lcrp = setup_communication(cr, n_nodes);
   PAS_WRITE_TIME("Setup of Communication");

   lcrp->threads = numthreads; 

   /* Diagnostic output corresponding to given output level */
   check_lcrp(me, lcrp);


   PAS_CYCLE_START;

   pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;
   hlpvec_out = (double*) allocateMemory( lcrp->lnRows[me] * sizeof( double ), "hlpvec_out" );

#pragma omp parallel for schedule(runtime)
   for (i=0; i<lcrp->lnRows[me]; i++) hlpvec_out[i] = 0.0;



#ifdef REVBUF
   
   /* Setup revolving buffer */
   RevBuf = revolvingBuffer( cache_size, pagesize, pseudo_ldim );

   for (rb_cnt=0; rb_cnt<RevBuf->numvecs; rb_cnt++){
      hlpvec_in = RevBuf->vec[rb_cnt];
      if       (place_rhs == 1){
#pragma omp parallel for schedule(runtime)
	 for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in[i] = 0.0;
      }
      if       (place_rhs == 2){
#pragma omp parallel for schedule(runtime)
	 for( i = 0; i < lcrp->lnRows[me]; i++ ) hlpvec_in[i] = 0.0;
#pragma omp parallel for schedule(runtime)
	 for( i = lcrp->lnRows[me]; i < pseudo_ldim; i++ ) hlpvec_in[i] = 0.0;
      }
      else{ 
	 for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in[i] = 0.0;
      } 
   

   /* Scatter the input vector from the master node to all others */
   ierror = MPI_Scatterv ( rhsVec->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
	 hlpvec_in, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

   /* Fill up halo with some markers */
   for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) hlpvec_in[i] = 77.0;

   }

#else



   hlpvec_in =  (double*) allocateMemory( pseudo_ldim      * sizeof( double ), "hlpvec_in" );

   /* Placement of RHS Vector */
   if       (place_rhs == 1){
#pragma omp parallel for schedule(runtime)
      for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in[i] = 0.0;
   }
   if       (place_rhs == 2){
#pragma omp parallel for schedule(runtime)
      for( i = 0; i < lcrp->lnRows[me]; i++ ) hlpvec_in[i] = 0.0;
#pragma omp parallel for schedule(runtime)
      for( i = lcrp->lnRows[me]; i < pseudo_ldim; i++ ) hlpvec_in[i] = 0.0;
   }
   else{ 
      for( i = 0; i < pseudo_ldim; i++ ) hlpvec_in[i] = 0.0;
   }


   /* Scatter the input vector from the master node to all others */
   ierror = MPI_Scatterv ( rhsVec->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
	 hlpvec_in, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

   /* Fill up halo with some markers */
   for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) hlpvec_in[i] = 77.0;



#endif




   PAS_WRITE_TIME("Setup of invec");


   fflush(stdout);
   MPI_Barrier(MPI_COMM_WORLD);



#ifdef CLOSER

#ifdef INDIVIDUAL
   for( i = 0; i < N_MULTS+1; i++ ) {
      PAS_CYCLE_START;
      hybrid_kernel( i, hlpvec_out,  lcrp, hlpvec_in);
      PAS_GET_TIME;
      if (me==0){
	 the_cycles = (double) asm_cycles;
	 the_time = time_it_took;
	 perf_array[i] = 2.0e-6*(double)cr->nEnts/the_time;
	 acc_cycles += the_cycles;
	 acc_time += the_time;
	 if (i==0){ /* nehme ersten Durchgang aus Mittelung raus */
	    acc_cycles=0;
	    acc_time=0;
	 }
      }
   }
   if(me==0) printf("Perform Hybrid-SpMVMs   [s] : %12.3f\n", acc_time);

#else 
   /* First pass of kernel not taken into account due to initialisations 
    * Thus the cycle counter is started for i==1 and we have to perform an
    * additional iteration */
   for( i = 0; i < N_MULTS+1; i++ ) {
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Perform Hybrid-SpMVMs");
   if (me==0){
      acc_cycles = (double) asm_cycles;
      acc_time = time_it_took;
   }
#endif

#else

if (1==0){

   /* Reine OpenMP-Variante fuer den Fall nur eines MPI-Prozesses */
   if (lcrp->nodes==1){
      for( i = 0; i < N_MULTS+1; i++ ) {
	 this_one = i%RevBuf->numvecs;
	 rhsVec->val = RevBuf->vec[this_one];
	 if (i==1) {PAS_CYCLE_START;}
	 hybrid_kernel_0( i, hlpvec_out,  lcrp, hlpvec_in);
      }
      PAS_WRITE_TIME("Hybrid-SpMVM KV0");
      if (me==0){ car[0] = (double) asm_cycles; tar[0] = time_it_took; }
   }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_I( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVI");
   if (me==0){ car[1] = (double) asm_cycles; tar[1] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* SendRecv omp-paralleles Umkopieren */
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_II( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVII");
   if (me==0){ car[2] = (double) asm_cycles; tar[2] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* AlltoAllv */
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_III( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVIII");
   if (me==0){ car[3] = (double) asm_cycles; tar[3] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* SendRecv Umkopieren collapse(2) */
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_IV( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVIV");
   if (me==0){ car[4] = (double) asm_cycles; tar[4] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* Isend/IRecv Baseline*/
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_V( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVV");
   if (me==0){ car[5] = (double) asm_cycles; tar[5] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* Isend/IRecv, Umkopieren OMP parallel */
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_VI( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVVI");
   if (me==0){ car[6] = (double) asm_cycles; tar[6] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* Isend/IRecv, parallele Region, MPI_THREAD_SERIALISED  */ 
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_VII( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVII");
   if (me==0){ car[7] = (double) asm_cycles; tar[7] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* Isend/IRecv, parallele Region, MPI_THREAD_MULTIPLE  */ 
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_VIII( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVVIII");
   if (me==0){ car[8] = (double) asm_cycles; tar[8] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* Isend/IRecv, separate lokale und nichtlokale array */
   for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
      if (i==1) {PAS_CYCLE_START;}
      hybrid_kernel_IX( i, hlpvec_out,  lcrp, hlpvec_in);
   }
   PAS_WRITE_TIME("Hybrid-SpMVM KVIX");
   if (me==0){ car[9] = (double) asm_cycles; tar[9] = time_it_took; }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* Isend/IRecv, separate lokale und nichtlokale array, Kommunikations-thread*/
   if (lcrp->threads>1){
      /* macht natuerlich nur Sinn bei mehr als einem thread pro MPI-Prozess */
      for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
	 if (i==1) {PAS_CYCLE_START;}
	 hybrid_kernel_X( i, hlpvec_out,  lcrp, hlpvec_in);
      }
      PAS_WRITE_TIME("Hybrid-SpMVM KVX");
      if (me==0){ car[10] = (double) asm_cycles; tar[10] = time_it_took; }
   }
   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* Isend/IRecv, separate lokale und nichtlokale array, Kommunikations-thread, copy mit allen*/
   if (lcrp->threads>1){
      /* macht natuerlich nur Sinn bei mehr als einem thread pro MPI-Prozess */
      for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
	 if (i==1) {PAS_CYCLE_START;}
	 hybrid_kernel_XI( i, hlpvec_out,  lcrp, hlpvec_in);
      }
      PAS_WRITE_TIME("Hybrid-SpMVM KVXI");
      if (me==0){ car[11] = (double) asm_cycles; tar[11] = time_it_took; }
   }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/

   /* Isend/IRecv, separate lokale und nichtlokale array, Kommunikations-thread, copy mit allen in einem block*/
   if (lcrp->threads>1){
      /* macht natuerlich nur Sinn bei mehr als einem thread pro MPI-Prozess */
      for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
	 if (i==1) {PAS_CYCLE_START;}
	 hybrid_kernel_XII( i, hlpvec_out,  lcrp, hlpvec_in);
      }
      PAS_WRITE_TIME("Hybrid-SpMVM KVXII");
      if (me==0){ car[12] = (double) asm_cycles; tar[12] = time_it_took; }
   }
   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/

   /* Isend/IRecv, separate lokale und nichtlokale array, Kommunikations-thread, 
    * copy mit allen in einem block, gezielt nicht ueberlappend */
   if (lcrp->threads>1){
      /* macht natuerlich nur Sinn bei mehr als einem thread pro MPI-Prozess */
      for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
	 if (i==1) {PAS_CYCLE_START;}
	 hybrid_kernel_XIII( i, hlpvec_out,  lcrp, hlpvec_in);
      }
      PAS_WRITE_TIME("Hybrid-SpMVM KVXIII");
      if (me==0){ car[13] = (double) asm_cycles; tar[13] = time_it_took; }
   }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   /* Isend/IRecv im separaten thread-Block, separate lokale und nichtlokale array,
    * Kommunikations-thread, copy mit allen in einem block */
   if (lcrp->threads>1){
      /* macht natuerlich nur Sinn bei mehr als einem thread pro MPI-Prozess */
      for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
	 if (i==1) {PAS_CYCLE_START;}
	 hybrid_kernel_XIV( i, hlpvec_out,  lcrp, hlpvec_in);
      }
      PAS_WRITE_TIME("Hybrid-SpMVM KVXIV");
      if (me==0){ car[14] = (double) asm_cycles; tar[14] = time_it_took; }
   }

   /****************************************************************************
    ********                                                            ********
    ***************************************************************************/
   if (lcrp->threads>1){
      /* macht natuerlich nur Sinn bei mehr als einem thread pro MPI-Prozess */
      for( i = 0; i < N_MULTS+1; i++ ) {
      this_one = i%RevBuf->numvecs;
      rhsVec->val = RevBuf->vec[this_one];
	 if (i==1) {PAS_CYCLE_START;}
	 hybrid_kernel_XV( i, hlpvec_out,  lcrp, hlpvec_in);
      }
      PAS_WRITE_TIME("Hybrid-SpMVM KVXIV");
      if (me==0){ car[15] = (double) asm_cycles; tar[15] = time_it_took; }
   }

}

   /* Loop over all kernel versions */
   for (version=0; version<NUMKERNELS; version++){

      /* Skip rest for versions that make only sense for more than one thread */
      if (version>10 && lcrp->threads==1) continue;

      for( iteration = 0; iteration < N_MULTS+1; iteration++ ) {

         /* choose current input vector from revolving buffer */
	 this_one = iteration%RevBuf->numvecs;
	 rhsVec->val = RevBuf->vec[this_one];
 
         /* Timing starts after the initialisation iteration */
	 if (iteration==1) {PAS_CYCLE_START;}

         /* call to multiplication kernel */
	 HyK[version].kernel( iteration, hlpvec_out, lcrp, hlpvec_in);

      }

      PAS_WRITE_TIME(HyK[version].name);

      if (me==0){ 
	 HyK[version].cycles = (double) asm_cycles;
	 HyK[version].time   = time_it_took; 
      }

      /* Perform correctness check once for each kernel version */ 
      Correctness_check( resCR, lcrp, hlpvec_out );

   }




#endif
   /****************************************************************************
    *****         Perform correctness-check against serial result          *****
    ***************************************************************************/
   PAS_CYCLE_START;

   hlpres_serial =  (double*) allocateMemory( lcrp->lnRows[me] * sizeof( double ), "hlpvec_in" );

   /* Scatter the serial result-vector */
   ierror = MPI_Scatterv ( resCR->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
	 hlpres_serial, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

   error_count = 0;
   for (i=0; i<lcrp->lnRows[me]; i++){
      mytol = EPSILON * (1.0 + fabs(hlpres_serial[i]) ) ;
      if (fabs(hlpres_serial[i]-hlpvec_out[i]) > mytol){
	 printf( "Correctness-check Hybrid:  PE%d: error in row %i:", me, i);
	 printf(" Differences: %e   Value ser: %25.16e Value par: %25.16e\n",
	       hlpres_serial[i]-hlpvec_out[i], hlpres_serial[i], hlpvec_out[i]);
         MPI_ABORT(999, MPI_COMM_WORLD);
	 error_count++;
      }
   }
   ierror = MPI_Reduce ( &error_count, &acc_error_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
   PAS_WRITE_TIME("Correctness-Check");

   if (me==0){
      printf("-------------------------------------------------------\n");
      printf("serial time measurement [us]: %12.3f\n", (1e6*cycles4measurement)/clockfreq);
      printf("global sync & time      [us]: %12.3f\n", (1e6*p_cycles4measurement)/clockfreq);
      printf("Number of iterations        : %12.0f\n", 1.0*N_MULTS);
      printf("-------------------------------------------------------\n");
      printf("----------------   Correctness-Check    ---------------\n");
      if (acc_error_count==0) 
	 printf("----------------    *** SUCCESS ***    ---------------\n");
      else printf("FAILED ---  %d errors\n", acc_error_count);

#ifdef CLOSER
      printf("=======================================================\n");
      printf("Cycles per nze  | Total time per MVM [ms]   |   MFlop/s\n"); 
      printf("%11.3f   %19.5f   %19.3f\n",  
	    acc_cycles/((double)N_MULTS*(double)cr->nEnts), 
	    1000*acc_time/((double)N_MULTS), 
	    2.0e-6*(double)N_MULTS*(double)cr->nEnts/acc_time);
#else
      printf("=======================================================\n");
      printf("Kernel | Cycles per nze | Time per MVM [ms] |   MFlop/s\n"); 
      for (i=0; i<16; i++){
	 //acc_cycles = car[i];
	 //acc_time = tar[i];
	 acc_cycles = HyK[i].cycles;
	 acc_time =   HyK[i].time;
	 if (acc_cycles>0) printf("%4d    %12.3f   %16.5f %15.3f\n", i ,
	       acc_cycles/((double)N_MULTS*(double)cr->nEnts), 
	       1000*acc_time/((double)N_MULTS), 
	       2.0e-6*(double)N_MULTS*(double)cr->nEnts/acc_time);
      }

#endif
      printf("=======================================================\n");

#ifdef INDIVIDUAL

#ifdef KVV
      sprintf(statfilename, "./PerfStat_KVV_p%d_t%d_%s_%s.dat", 
	    n_nodes, numthreads, testcase, mach_name);
#elif defined KVIII
      sprintf(statfilename, "./PerfStat_KIII_p%d_t%d_%s_%s.dat", 
	    n_nodes, numthreads, testcase, mach_name);
#else
      sprintf(statfilename, "./PerfStat_p%d_t%d_%s_%s.dat", 
	    n_nodes, numthreads, testcase, mach_name);
#endif


      if ((STATFILE = fopen(statfilename, "w"))==NULL){
	 printf("Fehler beim Oeffnen von %s\n", statfilename);
	 exit(1);
      }
      fprintf(STATFILE,"#%4d %19.5lf\n", 0, perf_array[0]);
      fprintf(STATFILE,"#----------------------------------------\n");
      for (i=1;i<N_MULTS+1;i++) fprintf(STATFILE,"%4d %19.5lf\n", i, perf_array[i]); 
      fclose(STATFILE);
#endif
   }



   /*
   //  printf("Debug-free vor allem\n");fflush(stdout);
   freeVector( rhsVec );
   //  printf("Debug-free: freed rhsVec\n");fflush(stdout);
   freeVector( resMM );
   //  printf("Debug-free freed resMM\n");fflush(stdout);
   freeVector( resCR );
   //  printf("Debug-free: freed resCR\n");fflush(stdout);
   freeVector( resJD );
   //  printf("Debug-free: freed resJD\n");fflush(stdout);
   freeCRMatrix( cr );
   // printf("Debug-free: freed cr\n");fflush(stdout);
   freeJDMatrix( jd );
   //  printf("Debug-free: freed jd\n");fflush(stdout);
   free(statistics);
   //   printf("Debug-free: freed statistics\n");fflush(stdout);
   */
   MPI_Finalize();

   return 0;
}
