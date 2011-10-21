#include "matricks.h"
#include <string.h>
#include "timing.h"
#include <math.h>
#include <sys/times.h>
#include <unistd.h>
#include <omp.h>
#include <sched.h>

#include <mpi.h>

/* ########################################################################## */

#ifdef JAN

#include <perfmon.h>
#include <rdtsc.h>

TscCounter start, stop, over;

#define perfmon_event_internal_start \
   RDTSC(start); \
RDTSC(stop); \
over.int64 = stop.int64 - start.int64; \
/*perfmon_markerStartCounters(thread_id);*/ \
perfmon_markerStartCounters(0); \
RDTSC(start);

#define perfmon_event_internal_stop \
   RDTSC(stop); \
perfmon_markerStopCounters(0); \
printf("STORE CYCLES\n"); \
perfmon_markerSetCycles(stop.int64 - start.int64 - over.int64); 

#else

#define perfmon_event_internal_start
#define perfmon_event_internal_stop

#endif


#ifdef hades

#define PAS_CYCLE_START
#define PAS_CYCLE_STOP
#define AS_CYCLE_START
#define AS_CYCLE_STOP
#define EVALUATE_CYCLES(identifier,entries)
#define AS_WRITE_TIME(identifier)

#else


#define PAS_CYCLE_START \
   ierror = MPI_Barrier(MPI_COMM_WORLD);\
   if (me == 0) for_timing_start_asm_( &asm_cyclecounter); \
   ierror = MPI_Barrier(MPI_COMM_WORLD);

#define PAS_CYCLE_STOP \
   ierror = MPI_Barrier(MPI_COMM_WORLD);\
   if (me == 0) for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles); \
   ierror = MPI_Barrier(MPI_COMM_WORLD);

#define PAS_WRITE_TIME(identifier) \
   ierror = MPI_Barrier(MPI_COMM_WORLD);\
   if (me == 0){ \
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles); \
      asm_cycles = asm_cycles - p_cycles4measurement; \
      time_it_took = (1.0*asm_cycles)/clockfreq; \
      printf("%-23s [s] : %12.3f\n", identifier, time_it_took ); } \
   ierror = MPI_Barrier(MPI_COMM_WORLD);

#define PAS_GET_TIME \
  ierror = MPI_Barrier(MPI_COMM_WORLD);			     \
  if (me == 0){						     \
    for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);   \
    asm_cycles = asm_cycles - p_cycles4measurement;	     \
    time_it_took = (1.0*asm_cycles)/clockfreq; }	     \
  ierror = MPI_Barrier(MPI_COMM_WORLD);

#define PAS_EVALUATE_CYCLES(identifier, entries)  \
   ierror = MPI_Barrier(MPI_COMM_WORLD);\
   if (me == 0){ \
      for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles); \
      asm_cycles = asm_cycles - p_cycles4measurement; \
      time_it_took = (1.0*asm_cycles)/clockfreq; \
      printf("\t\t\t\t Cycles per nze  | Total time per MVM [ms]   |   MFlop/s\n"); \
      printf("%-23s:   %15.3f   %20.5f   %18.3f\n", identifier,		\
	     (double)asm_cycles/((double)N_MULTS*(double)entries),	\
	     1000*time_it_took/((double)N_MULTS),			\
	     2.0e-6*(double)N_MULTS*(double)entries/time_it_took);	\
      printf("Gesamtzahl Cycles:  %25.13lg %15.3f\n", (double)asm_cycles, time_it_took); } \
   ierror = MPI_Barrier(MPI_COMM_WORLD);


#define AS_CYCLE_START for_timing_start_asm_( &asm_cyclecounter);

#define AS_CYCLE_STOP for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles );

#define EVALUATE_CYCLES(identifier, entries)	 \
  asm_cycles = asm_cycles - cycles4measurement;	 \
  time_it_took = (1.0*asm_cycles)/clockfreq;				\
  printf("\t\t\t\t Cycles per nze  | Total time per MVM [ms]   |   MFlop/s\n"); \
  printf("%-23s:   %15.3f   %20.5f   %18.3f\n", identifier,		\
	 (double)asm_cycles/((double)N_MULTS*(double)entries),		\
	 1000*time_it_took/((double)N_MULTS),				\
	 2.0e-6*(double)N_MULTS*(double)entries/time_it_took);		\
  printf("Gesamtzahl Cycles:  %25.13lg %15.3f\n", (double)asm_cycles, time_it_took);

#define AS_WRITE_TIME(identifier)		 \
   asm_cycles = asm_cycles - cycles4measurement; \
time_it_took = (1.0*asm_cycles)/clockfreq; \
printf("%-23s [s] : %12.3f\n", identifier, time_it_took );

#endif


#define diagnose_performance(identifier, entries) 
/* #define diagnose_performance(identifier, entries) \
   printf( "Average MVM for %-40s : %8.2e s %8.2f MFLOPs\n", \
   identifier, (double)(stopTime-startTime)/(N_MULTS), \ 
   ( (double)(2.0*(double)(entries)) / \
   (1000.0*1000.0*(double)(stopTime-startTime)/ (N_MULTS)) ));
   */

static void actcrs_(int*, int*, double*, double*, double*, int*, int*);
static uint64 sp_mvm_timing_asm(int*, int*, double*, double*, double*, int*, int*);
static uint64 get_timing_overhead_asm();
static void sp_mvm_asm(int*, int*, double*, double*, double*, int*, int*);
static void do_stats_(int*, double*, int*, double*, double*, double*, double*, double*, double*);

int ierror;
int pseudo_ldim;
double the_cycles, the_time;
double acc_cycles, acc_time;

void myblockjds_resorted_(int*, int*, int*, int*, int*, int*, double*, double*, double*, double*);
//void myblockjds_resorted_(int*, double*, double*, double*, int*, int*);

void myblockjds_asm(int*, double*, double*, double*, int*, int*);

/*void testfomp_(void);*/
extern void readCycles(uint64* );
float myCpuClockFrequency();
unsigned long machname(char* );
unsigned long kernelversion(char* );
unsigned long modelname(char* );
unsigned long cachesize(char* );

int error_count, acc_error_count;

double *hlpvec_in, *hlpvec_out, *hlpres_serial;

double *perf_array;

uint64 cycles, counter;
void myJDS_pure_asm(int*, double*, double*, double*, int*, int*, int*, int*);

const double EPSILON = 1e-6;

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
   char testcase[12];
   char mach_name[50];
   char kernel_version[50];
   char model_name[50];
   char cache_size[12];
   MM_TYPE* mm = NULL;
   CR_TYPE* cr = NULL;
   CR_P_TYPE* cr_parallel = NULL;
   LCRP_TYPE* lcrp = NULL; 
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
   uint64 memcounter, asm_cycles, asm_cyclecounter;
   uint64 tmp_cycles, tmp_cyclecounter;

   double time_it_took;
   double total_mem;
   uint64 num_fill;

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


   char tmp_string[50];

   double* tmparray;
   double* statistics;
   int* essential_flag;
   void test_omp_();
   void test_comp();
   void analyse_matrix_(int*, int*, int*, int*, int*);
   double my_amount_of_mem();
 
   unsigned int cpusetsize;
   cpu_set_t* cpumask;

   int required_threading_level;
   int provided_threading_level;

     //   MPI_Init(&nArgs, &arg);

   required_threading_level = MPI_THREAD_MULTIPLE;

   ierr = MPI_Init_thread(&nArgs, &arg, required_threading_level, &provided_threading_level );

  
   ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );
   ierr = MPI_Comm_size ( MPI_COMM_WORLD, &n_nodes );


#ifdef CLEAN_MEM
   IF_DEBUG(1) printf("cleaning up memory on all nodes...\n"); 
   total_mem = my_amount_of_mem();
   IF_DEBUG(1) printf("Total memory as read from /cpu/meminfo: %10.0f kB\n", total_mem/(1024.0));
   num_fill = (uint64)(0.5*total_mem/(8.0));
   IF_DEBUG(1) printf("Allocate %llu doubles to fill 90%% of the total memory\n", num_fill);
   tmparray = (double*) malloc(num_fill * sizeof( double ));
   for (memcounter=0; memcounter<num_fill; memcounter++) tmparray[memcounter]=0.0;
   IF_DEBUG(1) printf("Freeing memory again ...\n"); 
   free (tmparray);
   IF_DEBUG(1) printf("... done\n"); 
#endif

   mypid = getpid();
   //int sched_getaffinity(pid_t pid, unsigned int cpusetsize, cpu_set_t *mask);
   // ierr = sched_getaffinity(mypid, cpusetsize, cpumask);
 
   //   printf("PE %d: PID: %d, cpusetsize: %d, cpumask: %d", me, mypid, cpusetsize, cpumask);
  

#ifdef _OPENMP
#pragma omp parallel 
   numthreads = omp_get_num_threads();
#else
   numthreads = 1;
#endif


   if (me == 0){

      if ( nArgs != 3 ) mypabort("expect input: [N_MULTS] [testcase]"); 

      N_MULTS  = atoi(arg[1]);

      mystringlength = strlen(arg[2]);
      if (mystringlength > 11) mypabort("testcase longer than field"); 
      strcpy(testcase, arg[2]);


      if ( (machname(mach_name)   > 49) || (kernelversion(kernel_version) > 49) ||
	    (modelname(model_name) > 49) || (cachesize(cache_size) > 11)   ) 
	 mypabort("some system variable longer than field"); 


      /* Get overhead of cycles measurement */
      AS_CYCLE_START;
      AS_CYCLE_STOP;
      cycles4measurement = asm_cycles;
      clockfreq = myCpuClockFrequency(); 

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
      printf("Cache size [kB]             : %12s\n", cache_size); 
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

      perf_array = (double*) allocateMemory( (N_MULTS+1) *sizeof( double ), "perf_array" );
      
      acc_cycles = 0.0;
      acc_time = 0.0;

   }
   else{
     AS_CYCLE_START;
     AS_CYCLE_STOP;
     cycles4measurement = asm_cycles;
     clockfreq = myCpuClockFrequency(); 
   }

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

   /**************************************************************************** 
    *****  Paralleles Einlesen der Matrix und Aufsetzen der Kommunikation  *****
    ***************************************************************************/
   PAS_CYCLE_START
     lcrp = parallel_MatRead(testcase);
   PAS_CYCLE_STOP;
   PAS_WRITE_TIME("Parallel reading of CR & setup");
   
 
 
    
   /* Verifiere Loesung mittels serieller CRS. Hierfuer globales gather etc noetig...
      PAS_CYCLE_START;
      cr = convertMMToCRMatrix( mm );
      PAS_CYCLE_STOP;
      PAS_WRITE_TIME("Setup of CR");
      
      for (i=0; i<cr->nCols; i++) rhsVec->val[i] = i+1;
      
      fortrancrs_(&(cr->nRows), &(cr->nEnts), 
      resCR->val, rhsVec->val, cr->val , cr->col, cr->rowOffset);*/
  
     

   
   PAS_CYCLE_START;
   //lcrp = setup_communication_parRead(lmmp);
   PAS_WRITE_TIME("Setup of Communication");

   lcrp->threads = numthreads; 
   
   /* Diagnostic output corresponding to given output level */
   check_lcrp(me, lcrp);



   PAS_CYCLE_START;
  
   pseudo_ldim = lcrp->lnRows[me]+lcrp->halo_elements ;
   hlpvec_out = (double*) allocateMemory( lcrp->lnRows[me] * sizeof( double ), "hlpvec_out" );
   hlpvec_in =  (double*) allocateMemory( pseudo_ldim      * sizeof( double ), "hlpvec_in" );

   /* NUMA-Placement fuer in & outvec mittels first touch */
#ifdef PLACE
#pragma omp parallel for schedule(runtime)
#endif
   for (i=0; i<pseudo_ldim; i++) hlpvec_in[i] = 0.0;
#ifdef PLACE
#pragma omp parallel for schedule(runtime)
#endif
   for (i=0; i<lcrp->lnRows[me]; i++) hlpvec_out[i] = 0.0;

   ierr = MPI_Barrier(MPI_COMM_WORLD);
    printf(" PE %d: invec allocation and placement\n", me);
   ierr = MPI_Barrier(MPI_COMM_WORLD);

   /* Scatter the input vector from the master node to all others */
   //ierror = MPI_Scatterv ( rhsVec->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
   //	 hlpvec_in, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );
   // hier muss ich mir wohl was anderes einfallen lassen...
   // wohl am sinnvollsten auch verteilt einlesen
   for (i=0; i<lcrp->lnRows[me]; i++) hlpvec_in[i] = 1.0 + (1.0*i)/(1.0*lcrp->lnRows[me]);


   /* Fill up halo with some markers */
   for (i=lcrp->lnRows[me]; i< pseudo_ldim; i++) hlpvec_in[i] = 77.0;
  
  PAS_WRITE_TIME("Setup of invec");

   ierr = MPI_Barrier(MPI_COMM_WORLD);
    printf(" PE %d: setup of invec completed\n", me);
   ierr = MPI_Barrier(MPI_COMM_WORLD);


#ifdef INDIVIDUAL
  for( i = 0; i < N_MULTS+1; i++ ) {
     PAS_CYCLE_START;
     hybrid_kernel( i, hlpvec_out,  lcrp, hlpvec_in);

     printf("PE: %d: bin wieder rausgekommen\n", me); fflush(stdout);

     PAS_GET_TIME;
     if (me==0){
	the_cycles = (double) asm_cycles;
	the_time = time_it_took;
	perf_array[i] = 2.0e-6*(double)mm->nEnts/the_time;
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



   //PAS_EVALUATE_CYCLES("Hybrid",mm->nEnts);




   PAS_CYCLE_START;

   hlpres_serial =  (double*) allocateMemory( lcrp->lnRows[me] * sizeof( double ), "hlpvec_in" );

#ifdef HAVE_REFERENCE
   /* Scatter the serial result-vector to perform correctness check on all PEs */
   ierror = MPI_Scatterv ( resCR->val, lcrp->lnRows, lcrp->lfRow, MPI_DOUBLE, 
	 hlpres_serial, lcrp->lnRows[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );

   error_count = 0;
   for (i=0; i<lcrp->lnRows[i]; i++){
      if (fabs(hlpres_serial[i]-hlpvec_out[i]) > EPSILON){
	 error_count++;
	 printf("Correctness-Check: PE %d: error for i=%d: %25.13g  (%25.13g <-> %25.13g)\n",
	       me, i, hlpres_serial[i]-hlpvec_out[i], hlpres_serial[i], hlpvec_out[i]); 
         fflush(stdout);
      }
   }
   ierror = MPI_Reduce ( &error_count, &acc_error_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
   PAS_WRITE_TIME("Correctness-Check");
#endif

   if (me==0){
     printf("-------------------------------------------------------\n");
     printf("serial time measurement [ms]: %12.3f\n", (1e3*cycles4measurement)/clockfreq);
     printf("global sync & time      [ms]: %12.3f\n", (1e3*p_cycles4measurement)/clockfreq);
     printf("Number of iterations        : %12.0f\n", 1.0*N_MULTS);
     printf("-------------------------------------------------------\n");
      printf("----------------   Correctness-Check    ---------------\n");
      if (acc_error_count==0) 
	 printf("----------------    *** SUCCESS ***    ---------------\n");
      else printf("FAILED ---  %d errors\n", acc_error_count);
      printf("=======================================================\n");
      printf("Cycles per nze  | Total time per MVM [ms]   |   MFlop/s\n"); 
      printf("%11.3f   %19.5f   %19.3f\n",  
	     acc_cycles/((double)N_MULTS*(double)lcrp->nEnts), 
	     1000*acc_time/((double)N_MULTS), 
	     2.0e-6*(double)N_MULTS*(double)lcrp->nEnts/acc_time);
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
