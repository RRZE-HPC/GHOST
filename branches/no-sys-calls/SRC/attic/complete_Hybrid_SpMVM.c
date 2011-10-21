#include "matricks.h"
#include <string.h>
#include "timing.h"
#include <math.h>
#include <sys/times.h>
#include <unistd.h>
#include <omp.h>


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


/* #define EVALUATE_CYCLES(identifier, entries)  \
   asm_cycles = asm_cycles - cycles4measurement; \
   time_it_took = (1.0*asm_cycles)/clockfreq; \
   printf("%-20s: Cycles total          : %llu\n", identifier, asm_cycles); \
   printf("%-20s: Cycles per MVM        : %10.1f\n", identifier, (double)asm_cycles/((double)N_MULTS)); \
   printf("%-20s: Cycles per element    : %10.3f\n", identifier, (double)asm_cycles/((double)N_MULTS*(double)entries)); \
   printf("%-20s: Time total [s]        : %10.3g\n", identifier, time_it_took); \
   printf("%-20s: Time per MVM [s]      : %10.3g\n", identifier, time_it_took/((double)N_MULTS)); \
   printf("%-20s: Time per element [s]  : %10.3g\n", identifier, time_it_took/((double)N_MULTS*(double)entries)); \
   printf("%-20s: Performance [MFlop/s] : %10.3f\n", identifier, 2.0e-6*(double)N_MULTS*(double)entries/time_it_took); \
 */

#ifdef hades

#define AS_CYCLE_START
#define AS_CYCLE_STOP
#define EVALUATE_CYCLES(identifier,entries)
#define AS_WRITE_TIME(identifier)

#else

#define AS_CYCLE_START for_timing_start_asm_( &asm_cyclecounter);

#define AS_CYCLE_STOP for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles );

#define EVALUATE_CYCLES(identifier, entries)  \
   asm_cycles = asm_cycles - cycles4measurement; \
time_it_took = (1.0*asm_cycles)/clockfreq; \
printf("%-23s:   %15.3f   %20.5f   %18.3f\n", identifier, \
      (double)asm_cycles/((double)N_MULTS*(double)entries), \
      time_it_took/((double)N_MULTS), 2.0e-6*(double)N_MULTS*(double)entries/time_it_took);

#define AS_WRITE_TIME(identifier) \
   asm_cycles = asm_cycles - cycles4measurement; \
time_it_took = (1.0*asm_cycles)/clockfreq; \
printf("%-23s [s] : %10.3f\n", identifier, time_it_took );

#endif


#define diagnose_performance(identifier, entries) 
/* #define diagnose_performance(identifier, entries) \
   printf( "Average MVM for %-40s : %8.2e s %8.2f MFLOPs\n", \
   identifier, (double)(stopTime-startTime)/(N_MULTS), \ 
   ( (double)(2.0*(double)(entries)) / \
   (1000.0*1000.0*(double)(stopTime-startTime)/ (N_MULTS)) ));
 */

typedef unsigned long long uint64;
static void actcrs_(int*, int*, double*, double*, double*, int*, int*);
static uint64 sp_mvm_timing_asm(int*, int*, double*, double*, double*, int*, int*);
static uint64 get_timing_overhead_asm();
static void sp_mvm_asm(int*, int*, double*, double*, double*, int*, int*);
static void do_stats_(int*, double*, int*, double*, double*, double*, double*, double*, double*);



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

uint64 cycles, counter;
void myJDS_pure_asm(int*, double*, double*, double*, int*, int*, int*, int*);
void for_timing_start_asm_(uint64*);
void for_timing_stop_asm_(uint64*, uint64*);


int main( const int nArgs, const char* arg[] ) {
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
   JD_TYPE* jd = NULL;
   JD_RESORTED_TYPE* jdr = NULL;
   JD_OFFDIAGONAL_TYPE* jdo = NULL;
   VIP_TYPE* vip = NULL;
   VECTOR_TYPE* rhsVec = NULL;
   VECTOR_TYPE* resMM  = NULL;
   VECTOR_TYPE* resCR  = NULL;
   VECTOR_TYPE* resJD  = NULL;
   uint64 mystart=0ULL, mystop=0ULL, overhead;
   uint64 sumcycles=0ULL, cycles=0ULL, test=0ULL, min=1000000000ULL;
   uint64 mincycles=100000000000000ULL;
   uint64 memcounter, asm_cycles, asm_cyclecounter, cycles4measurement;
   double time_it_took, clockfreq;
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


   /* Number of treads per node */
   int n_threads;

   /* Number of nodes */
   int n_nodes;

   char tmp_string[50];

   double* tmparray;
   double* statistics;
   int* essential_flag;
   void test_omp_();
   void test_comp();
   void analyse_matrix_(int*, int*, int*, int*, int*);
   double my_amount_of_mem();


   if ( nArgs != 5 ) myabort("expect input: [N_MULTS] [blocklen] [READ/WRITE] [testcase]"); 

   N_MULTS  = atoi(arg[1]);
   blocklen = atoi(arg[2]);
   rw_flag  = atoi(arg[3]);

   mystringlength = strlen(arg[4]);
   if (mystringlength > 11) myabort("testcase longer than field"); 
   strcpy(testcase, arg[4]);


   if ( (machname(mach_name)   > 49) || (kernelversion(kernel_version) > 49) ||
	 (modelname(model_name) > 49) || (cachesize(cache_size) > 11)   ) 
      myabort("some system variable longer than field"); 

   if ( (statistics = malloc ((MAX(N_TRIES,N_MULTS))*sizeof(double)) ) == 0)
      myabort("Allokieren des Statistik-arrays fehlgeschlagen\n");

   if ( (essential_flag = malloc ((MAX(N_TRIES,N_MULTS))*sizeof(int)) ) == 0)
      myabort("Allokieren des flag-arrays fehlgeschlagen\n");

   /* Get overhead of cycles measurement */
   AS_CYCLE_START;
      AS_CYCLE_STOP;
      cycles4measurement = asm_cycles;
   clockfreq = myCpuClockFrequency(); 

   n_nodes = 4;


   total_mem = my_amount_of_mem();
   mypid = getpid();

#ifdef _OPENMP
#pragma omp parallel 
   numthreads = omp_get_num_threads();
#else
   numthreads = 1;
#endif

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
   printf("Number of OpenMP threads    : %12i\n", numthreads); 
   printf("Value of $KMP_AFFINITY      : %12s\n", getenv("KMP_AFFINITY")); 
   printf("Value of $OMP_SCHEDULE      : %12s\n", getenv("OMP_SCHEDULE")); 
   printf("Value of $LD_PRELOAD        : %12s\n", getenv("LD_PRELOAD")); 


#ifdef CLEAN_MEM
   IF_DEBUG(1) printf("cleaning up memory...\n"); 
   IF_DEBUG(1) printf("Total memory as read from /cpu/meminfo: %10.0f kB\n", total_mem/(1024.0));
   num_fill = (uint64)(0.5*total_mem/(8.0));
   IF_DEBUG(1) printf("Allocate %llu doubles to fill 90%% of the total memory\n", num_fill);
   tmparray = (double*) malloc(num_fill * sizeof( double ));
   for (memcounter=0; memcounter<num_fill; memcounter++) tmparray[memcounter]=0.0;
   IF_DEBUG(1) printf("Freeing memory again ...\n"); 
   free (tmparray);
   IF_DEBUG(1) printf("... done\n"); 
#endif

      if (rw_flag == 1){

      /* Kein restart-file in CRS und JDS-Format vorhanden: Lese Matrix im 
       * Standardformat von MM ein und konvertiere in die entsprechenden 
       * anderen Formate. Danach Ausgabe der entsprechenden restart files */

      printf("-----------------------------------------------------\n");
      printf("------   Setup matrices in different formats   ------\n");
      printf("-----------------------------------------------------\n");

      AS_CYCLE_START
	 sprintf(restartfilename, "./daten/%s.mtx", testcase);
      mm = readMMFile( restartfilename, 0.0 );
      AS_CYCLE_STOP
	 AS_WRITE_TIME("Reading of MM");

      if( ! mm ) {
	 printf( "main: couldn't load matrix file\n" );
	 return 1;
      }

      IF_DEBUG(1){
	 printf( "main: Anzahl der Matrixeintraege = %i \n", mm->nEnts );
	 printf("blocklen = %i , N_MULTS = %i \n", blocklen, N_MULTS);
      }

      rhsVec = newVector( mm->nCols );
      resMM  = newVector( mm->nCols );
      resCR  = newVector( mm->nCols );
      resJD  = newVector( mm->nCols );


#ifdef ODJDS
      AS_CYCLE_START;
      jdo = convertMMToODJDMatrix(mm, blocklen);
      AS_CYCLE_STOP;
      AS_WRITE_TIME("Setup of JDO");
#endif

      AS_CYCLE_START;
      cr = convertMMToCRMatrix( mm );
      AS_CYCLE_STOP;
      AS_WRITE_TIME("Setup of CR");

      AS_CYCLE_START;
      jd = convertMMToJDMatrix( mm, blocklen);
      AS_CYCLE_STOP;
      AS_WRITE_TIME("Setup of JD");

#ifdef RSJDS
      AS_CYCLE_START;
      jdr = resort_JDS(jd, blocklen);
      AS_CYCLE_STOP;
      AS_WRITE_TIME("Setup of JDR");
#endif

      /* Skip the writing of a restart-file since placement does not work correctly for it yet
       *  Furthermore it is questionable if it really makes sense since depending on the blocksize
       *  the placement might differ (?)
       bin_write_cr(cr, testcase); 
       bin_write_jd(jd, testcase); */
   }
   else{     

      /*******   Allocate memory for CRS- and JDS-matrix structure   *******/
      cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
      jd = (JD_TYPE*) allocateMemory( sizeof( JD_TYPE ), "jd" );

      /* Read matrix in CRS and JDS from packed binary files */
      bin_read_cr(cr, testcase);
      bin_read_jd(jd, blocklen, testcase);

      /* Allocate input and result vectors */
      rhsVec = newVector( jd->nCols );
      resMM  = newVector( jd->nCols );
      resCR  = newVector( jd->nCols );
      resJD  = newVector( jd->nCols );
   }


      setup_communication(mm, cr_parallel, n_nodes);


#ifdef PLACE_JDS
#pragma omp parallel for schedule(runtime)
#endif
   for( i = 0; i < rhsVec->nRows; i++ ) rhsVec->val[jd->rowPerm[i]] = 0.0;
   IF_DEBUG(1) printf("Done NUMA-placement for rhsVec->val[Perm]\n");
   // at least do _some_ placement for RHS although access will (probably) be random

   /* Initialise right-hand side vector */
   for( i = 0; i < rhsVec->nRows; i++ ) rhsVec->val[i] = i+1.0;
   //for( i = 0; i < rhsVec->nRows; i++ ) rhsVec->val[i] = 5.0+3.0*i;



   if (rw_flag == 1){/* We have the original matrix, so we can calculate a reference result */

      IF_DEBUG(2) if (rhsVec->nRows < 100) for (i=0; i < rhsVec->nRows; i++)
	 printf("%i %f\n", i, rhsVec->val[i]);

      /* reference run (correctness check) */
      multiplyMMWithVector( resMM, mm, rhsVec );

      IF_DEBUG(2) if (resMM->nRows < 100) for (i=0; i < rhsVec->nRows; i++)
	 printf("Result%i %f\n", i, resMM->val[i]);

      /* GH: free up space for NUMA-placed arrays */
      freeMMMatrix( mm );
   }


#ifdef PLACE_CRS
#pragma omp parallel for schedule(runtime)
#endif
   for( i=0; i< resCR->nRows; ++i) resCR->val[i] = 0.0;
   IF_DEBUG(1) printf("Done NUMA-placement for resCR->val\n");

#ifdef PLACE_JDS
#pragma omp parallel for schedule(runtime)
#endif
   for( i=0; i< resJD->nRows; ++i) resJD->val[i] = 0.0;
   IF_DEBUG(1) printf("Done NUMA-placement for resJD->val\n");



#ifdef ANALYSE

   printf("Starte Analyse der Matrix im CRS-Format\n");
   hlpi = 0;
   analyse_matrix_( &(cr->nEnts), &(cr->nRows), cr->col, &blocklen, &hlpi );

   printf("Starte Analyse der Matrix im JDS-Format\n");
   hlpi = 1;
   analyse_matrix_( &(jd->nEnts), &(jd->nRows), jd->col, &blocklen, &hlpi );

   printf("Starte Analyse der Matrix im RSJDS-Format\n");
   hlpi = 2;
   analyse_matrix_( &(jdr->nEnts), &(jdr->nRows), jdr->resorted_col, &blocklen, &hlpi );

   printf("Starte Analyse der Matrix im ODJDS-Format\n");
   hlpi = 3;
   analyse_matrix_( &(jdo->nEnts), &(jdo->nRows), jdo->resorted_col, &blocklen, &hlpi );



   exit(0);
#endif


   num_flops = 2.0*(double)(cr->nEnts);
   ws = (cr->nRows*20.0 + cr->nEnts*12.0)/(1024*1024);
   nnz1 = (double)cr->nEnts;
   wdim1 = (double)cr->nRows;
   printf("-----------------------------------------------------\n");
   printf("-------         Statistics about matrix       -------\n");
   printf("-----------------------------------------------------\n");
   printf("Investigated matrix         : %12s\n", testcase); 
   printf("Dimension of matrix         : %12.0f\n", wdim1); 
   printf("Non-zero elements           : %12.0f\n", nnz1); 
   printf("Average elements per row    : %12.3f\n", nnz1/wdim1); 
   printf("Working set [MB]            : %12.3f\n", ws); 





   /****************************************************************************
    **********************         CRS - multiplication          ***************
    ***************************************************************************/


   /*printf("%8.2f  %8.2f\n", resCR->val[0], rhsVec->val[0]); fflush(stdout);*/
   /* multiplyCRWithVector( resCR, cr, rhsVec ); */
   /*#define POWER_CRS*/
#ifdef POWER_CRS	

   printf("Entering my own CRS-multiplication branch\n");

#ifdef CYCLES_EXT
   readCycles(&mystart);
   readCycles(&mystop);
   overhead = mystop-mystart;

   for( i = 0; i < N_MULTS+1; i++ ) {
      if (i==1){
	 readCycles(&mystart);
      }
      sp_mvm_asm(&(cr->nRows), &(cr->nEnts), resCR->val, rhsVec->val,
	    cr->val, cr->col, cr->rowOffset);
   }

   readCycles(&stop);

   cycles = (mystop-mystart-overhead);
   min = MIN(cycles,min);
   mycycles = cycles/(1.0*N_MULTS);
#endif

#ifdef CYCLES_INT


   /****************************************************************************
    *******  Determine overhead of timing routine and gather statistics  *******
    ***************************************************************************/

   cycles = 100000000000ULL;

#ifdef FSTAT
   for(i=0;i<N_TRIES+1;i++) {
      overhead = get_timing_overhead_asm();
      statistics[i] = (double) overhead;
      cycles = MIN(cycles,overhead);
   }
   overhead = cycles;

   sprintf(statfilename, "./daten/statistics/overheads_%d.dat", mypid);
   if ((STATFILE = fopen(statfilename, "w"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", statfilename);
      exit(1);
   }

   do_stats_(&N_TRIES, statistics, essential_flag, &mean, &stdev, &runaway,
	 &ess_mean, &ess_stdev, &ess_run); 

   fprintf(STATFILE, "###################################################\n");
   fprintf(STATFILE, "# Minimum overhead of timing function   = %llu\n", overhead); 
   fprintf(STATFILE, "# Average overhead of timing function   = %3.0f +/- %2.0f\n", mean, stdev); 
   fprintf(STATFILE, "# Percentage of runaways (2*sigma)      = %4.1f\n", runaway*100); 
   fprintf(STATFILE, "# Essential overhead of timing function = %3.0f +/- %2.0f\n", ess_mean, ess_stdev); 
   fprintf(STATFILE, "# Non-essential runaways (2*sigma_new)  = %4.1f\n", ess_run*100); 
   fprintf(STATFILE, "###################################################\n");
   fprintf(STATFILE, "# Invidual overhead of timing function meausrement:\n");
   fprintf(STATFILE, "# Iteration   |    cycles    |    essential_flag\n");
   for (i=0;i<N_TRIES;i++){
      fprintf(STATFILE, "%7i  %15.0f   %15i\n", i, statistics[i], essential_flag[i]);
   }
   fclose(STATFILE);
#endif

   /****************************************************************************
    *******  Measure required times for MVM in CRS and gather statistics *******
    ***************************************************************************/

   for( i = 0; i < N_MULTS; i++ ) {
      cycles = sp_mvm_timing_asm(&(cr->nRows), &(cr->nEnts), 
	    resCR->val, rhsVec->val, cr->val, cr->col, cr->rowOffset);
      cycles = cycles - overhead; 
      if (outlev > 2 ) printf("Benoetigte Cycles fuer SpMVM = %llu\n", cycles);
      statistics[i] = (double) cycles;
      mincycles = MIN(mincycles, cycles);
   }

   cycles = mincycles;
   mycycles = cycles;

#ifdef FSTAT
   sprintf(statfilename, "./daten/statistics/cycles_%d.dat", mypid);
   if ((STATFILE = fopen(statfilename, "w"))==NULL){
      printf("Fehler beim Oeffnen von %s\n", statfilename);
      exit(1);
   }

   do_stats_(&N_MULTS, statistics, essential_flag, &mean, &stdev, &runaway, 
	 &ess_mean, &ess_stdev, &ess_run); 
   mycycles = ess_mean;

   fprintf(STATFILE, "###################################################\n");
   fprintf(STATFILE, "# Minimum cycles for MVM in CRS        = %llu\n", mincycles); 
   fprintf(STATFILE, "# Average cycles for MVM in CRS        = %3.0f +/- %2.0f\n", mean, stdev); 
   fprintf(STATFILE, "# Percentage of runaways (2*sigma)     = %4.1f\n", runaway*100); 
   fprintf(STATFILE, "# Essential cycles for MVM in CRS      = %3.0f +/- %2.0f\n", ess_mean, ess_stdev); 
   fprintf(STATFILE, "# Non-essential runaways (2*sigma_new) = %4.1f\n", ess_run*100); 
   fprintf(STATFILE, "###################################################\n");
   fprintf(STATFILE, "# Required cycles for an individual SpMVM in CRS:\n");
   fprintf(STATFILE, "# Iteration   |    cycles     |    essential_flag\n");
   for (i=0;i<N_MULTS;i++){
      fprintf(STATFILE, "%7i  %15.0f  %15i\n", i, statistics[i], essential_flag[i]);
   }
   fclose(STATFILE);
#endif


#endif

   req_time = mycycles/myCpuClockFrequency();

   printf("-----------------------------------------------------\n");
   printf("-------            Statistics per MVM         -------\n");
   printf("-----------------------------------------------------\n");
   printf("Performed FLOPs             : %12.0f\n", num_flops);
   printf("Required cycles             : %12.0f\n", mycycles);
   printf("Required time [ms]          : %12.5f\n", req_time*1000);
   printf("Performance [MFLOPS/s]      : %12.3f\n", (num_flops*1e-6) / req_time); 
   printf("-----------------------------------------------------\n");
   printf("--- Statistics relative to matrix characteristics ---\n");
   printf("----------------------------------------------------\n");
   printf("Cycles per non-zero-element : %12.3f\n", mycycles/nnz1);
   printf("Cycles per matrix row       : %12.3f\n", mycycles/wdim1);
   printf("=====================================================\n");


#else

   IF_DEBUG(1){
      printf("CRS-multiplication with Gerhard-Routine\n");
      printf("%i %i \n", cr->nRows, cr->nEnts);
   }

   printf("---------------------------------------------------");
   printf("-------------------------------------------\n");
   printf("Multiplication scheme  |  cycles per element update");
   printf(" | time per MVM [s] | Performance [MFlop/s]\n");
   printf("---------------------------------------------------");
   printf("-------------------------------------------\n");

   for( i = 0; i < N_MULTS+1; i++ ){
      if (i==1){
	 timing( &startTime, &ct );
	 AS_CYCLE_START;
      } 
      fortrancrs_(&(cr->nRows), &(cr->nEnts), 
	    resCR->val, rhsVec->val, cr->val , cr->col, cr->rowOffset);
   }
   AS_CYCLE_STOP;
   timing( &stopTime, &ct );

   diagnose_performance("CRS", jdo->nEnts);
   EVALUATE_CYCLES("CRS" ,jdo->nEnts);

#endif


   IF_DEBUG(2) if (resMM->nRows < 100) for (i=0; i < rhsVec->nRows; i++)
      printf("Result CR %i %f\n", i, resCR->val[i]);

   /* Check CRS result against plain one */
   if (rw_flag == 1) {
      for( i = 0; i < cr->nCols; i++ ) {
	 if( fabs( resMM->val[i]-resCR->val[i] ) > EPSILON ) {
	    printf( "Correctness-check CRS: error in row %i:", i);
	    printf(" Differences: %e   Value MM: %25.16e Value CRS: %25.16e\n",
		  resMM->val[i] - resCR->val[i], resMM->val[i], resCR->val[i] );

	 }
      } 
      IF_DEBUG(1) printf( "Correctness-check CRS passed without error\n"); 
   }
   else	{
      printf("no correctness-check for CRS\n"); 
   }


#ifdef ODJDS

   /* rechte Seite nicht permutiert falls nur eine Permutation! */
   /* for( i = 0; i < rhsVec->nRows; i++ ) rhsVec->val[i] = i+1.0; */

   /* rechte Seite doch permutiert falls zwei Permutationen! */
   for( i = 0; i < rhsVec->nRows; i++ ) rhsVec->val[i] = jd->rowPerm[i]+1.0; 

   IF_DEBUG(2) printf("Entering off-diagnoal based JDS_SpMVM\n");fflush(stdout);

   IF_DEBUG(2) if (resJD->nRows < 100) for (i=0; i < jdo->nRows; i++)
      printf("rhs_invec ODJDS %i %f %p\n", i, rhsVec->val[i], &(rhsVec->val[i]));

   for( i = 0; i < N_MULTS+1; i++ ){
      if (i==1){
	 timing( &startTime, &ct );
	 AS_CYCLE_START;
      }
#ifdef BLOCKJDS_ASM
      //printf("Adresse von resJD vorher %p\n", resJD->val);
      myblockjds_asm( jdo->resorted_col, jdo->resorted_val, rhsVec->val, 
	    resJD->val, &(jdo->totalblocks), jdo->blockinfo );
      //printf("Adresse von resJD nachher:%p\n", resJD->val);
      //for(i=0;i<jdo->nRows;i++) printf("Adresse von resJD[%i] nachher:%f %p\n", i, resJD->val[i], &(resJD->val[i]));
#else
      /*myblockjds_resorted_(jdo->resorted_col, jdo->resorted_val, rhsVec->val, 
	resJD->val, &(jdo->totalblocks), jdo->tbi);*/
      myblockjds_resorted_(&(jdo->nEnts), 
	    &(jdo->nRows), &(jdo->totalblocks), jdo->blockdim_rows, jdo->blockdim_cols,
	    jdo->resorted_col, jdo->resorted_val, rhsVec->val, resJD->val, &clockfreq);
#endif
   }
   AS_CYCLE_STOP;
   timing( &stopTime, &ct );

   diagnose_performance("ODJDS", jdo->nEnts);
   EVALUATE_CYCLES("ODJDS", jdo->nEnts);

   errcount = 0;
   for( i = 0; i < jd->nCols; i++ ) {
      if( fabs( resMM->val[jd->rowPerm[i]]-resJD->val[i] ) > EPSILON ) {
	 printf( "error in row %i: Differences:  %e \n",
	       i, resMM->val[jd->rowPerm[i]] - resJD->val[i] );
	 errcount++;
      }
   } 
   IF_DEBUG(1){
      if (errcount==0) 
	 printf( "Correctness-check for ODJDS passed without error\n"); 
      else
	 printf( ">>>>> Correctness-check for ODJDS yields %d errors <<<<<\n", errcount); 
   }

   IF_DEBUG(1) if (resJD->nRows < 100) for (i=0; i < jdo->nRows; i++)
      printf("Result ODJDS %i %f\n", i, resJD->val[i]);

#endif


   /* In the classical JDS schemes rows and colums are permuted. This means that both
    * the input and resulting vector are in the transformed (permuted) basis. Therefort
    * we have to first permute the input vector prior to the application of the SpMVM */

   //for( i = 0; i < rhsVec->nRows; i++ ) rhsVec->val[i] = 5.0+3.0*jd->rowPerm[i]; 
   for( i = 0; i < rhsVec->nRows; i++ ) rhsVec->val[i] = jd->rowPerm[i]+1.0; 


#ifdef RSJDS

   IF_DEBUG(1) printf("Entering resorted blocked SpMVM\n");fflush(stdout);

   IF_DEBUG(2) if (resJD->nRows < 100) for (i=0; i < jd->nRows; i++)
      printf("rhs_invec RSJDS %i %f\n", i, rhsVec->val[i]);

   for( i = 0; i < N_MULTS+1; i++ ){
      if (i==1){
	 timing( &startTime, &ct );
	 AS_CYCLE_START;
      }
#ifdef BLOCKJDS_ASM
      myblockjds_asm( jdr->resorted_col, jdr->resorted_val, rhsVec->val, 
	    resJD->val, &(jdr->totalblocks), jdr->blockinfo );
#else
      /*myblockjds_resorted_(jdr->resorted_col, jdr->resorted_val, rhsVec->val, 
	resJD->val, &(jdr->totalblocks), jdr->tbi);*/
      myblockjds_resorted_(&(jdr->nEnts), &(jdr->nRows), 
	    &(jdr->totalblocks), jdr->blockdim_rows, jdr->blockdim_cols, 
	    jdr->resorted_col, jdr->resorted_val, rhsVec->val, resJD->val, &clockfreq);
#endif
   }
   AS_CYCLE_STOP;
   timing( &stopTime, &ct );

   diagnose_performance("RSJDS", jd->nEnts);
   EVALUATE_CYCLES("RSJDS", jd->nEnts);

   IF_DEBUG(2) if (resJD->nRows < 100) for (i=0; i < jd->nRows; i++)
      printf("Result RSJDS %i %f\n", i, resJD->val[i]);

   IF_DEBUG(2) if (resJD->nRows < 100) for( i = 0; i < jd->nCols; i++ ) 
      printf("Results: %i %f %e\n", i, resMM->val[i], resJD->val[i]);

   errcount = 0;
   for( i = 0; i < jd->nCols; i++ ) {
      if( fabs( resMM->val[jd->rowPerm[i]]-resJD->val[i] ) > EPSILON ) {
	 printf( "error in row %i: Differences:  %e \n",
	       i, resMM->val[jd->rowPerm[i]] - resJD->val[i] );
	 errcount++;
      }
   } 
   IF_DEBUG(1){
      if (errcount==0) 
	 printf( "Correctness-check for RSJDS passed without error\n"); 
      else
	 printf( ">>>>> Correctness-check for RSJDS yields %d errors <<<<<\n", errcount); 
   }

#endif


#ifdef myJDS
   printf("in\n");

   for_timing_start_asm_(&counter);
   perfmon_event_internal_start;

   myJDS_pure_asm(jd->diagOffset, rhsVec->val, resJD->val, jd->val, jd->col, 
	 &(jd->nDiags), &(jd->nRows), &(jd->nEnts));

   perfmon_event_internal_stop;

   for_timing_stop_asm_(&counter, &cycles);
   printf("Cycles required: %llu", cycles);
   exit(0);



#else

   for(mode=1; mode < 3; mode++) {

      IF_DEBUG(2) if (rhsVec->nRows < 100) for (i=0; i < rhsVec->nRows; i++)
	 printf("rhs_invec JDS %i %f\n", i, rhsVec->val[i]);



      for( i = 0; i < N_MULTS+1; i++ ) {
	 if (i==1){
	    timing( &startTime, &ct );
	    AS_CYCLE_START;
	 }
	 fortranjds_(&(jd->nRows), &(jd->nDiags), &(jd->nEnts), resJD->val, rhsVec->val, 
	       jd->diagOffset, jd->val,jd->col , &mode , &blocklen);
      }
      AS_CYCLE_STOP;
      timing( &stopTime, &ct );

      sprintf(tmp_string, "JDS (%d)", mode);
      diagnose_performance(tmp_string, jd->nEnts);
      EVALUATE_CYCLES(tmp_string, jd->nEnts);

      errcount = 0;
      for( i = 0; i < jd->nCols; i++ ) {
	 if( fabs( resMM->val[i             ]-resCR->val[i] ) > EPSILON ||
	       fabs( resMM->val[jd->rowPerm[i]]-resJD->val[i] ) > EPSILON ) {
	    printf( "error in row %i: Differences: %e  %e \n",
		  i, resMM->val[i] -  resCR->val[i], resMM->val[jd->rowPerm[i]] - resJD->val[i] );
	    errcount++;
	 }
      }
      IF_DEBUG(1){
	 if (errcount ==0)
	    printf( "Correctness-check for JDS (%d) passed without error\n", mode); 
	 else
	    printf( ">>>>> Correctness-check for JDS (%d) yields %d errors <<<<<\n", mode, errcount);
      } 

      IF_DEBUG(2) if (rhsVec->nRows < 100) for (i=0; i < rhsVec->nRows; i++) 
	 printf("Result JD %i %i %f %f %f\n", i, jd->rowPerm[i], resCR->val[i], resJD->val[i], resCR->val[jd->rowPerm[i]]);
   }



#endif

   mode=3;
   fblocklen = blocklen;
   oldblocklen = blocklen-1;
   /*for(; fblocklen < 80000; fblocklen*=1.1) {

     blocklen= (int)fblocklen;
     if (blocklen == oldblocklen) continue;
    */


   for( i = 0; i < N_MULTS+1; i++ ) {
      if (i==1){
	 timing( &startTime, &ct );
	 AS_CYCLE_START;
	 perfmon_event_internal_start;
      }
      fortranjds_(&(jd->nRows), &(jd->nDiags), &(jd->nEnts), resJD->val, rhsVec->val, 
	    jd->diagOffset, jd->val,jd->col , &mode , &blocklen);
   }

   perfmon_event_internal_stop;
   AS_CYCLE_STOP;
   timing( &stopTime, &ct );


   sprintf(tmp_string, "JDS (%d,%d)",mode, blocklen);
   diagnose_performance(tmp_string, jd->nEnts);
   EVALUATE_CYCLES(tmp_string, jd->nEnts);
   printf("===================================================");
   printf("===========================================\n");

   if (rw_flag == 1) {
      errcount = 0;
      for( i = 0; i < jd->nCols; i++ ) {
	 if( fabs( resMM->val[i             ]-resCR->val[i] ) > EPSILON ||
	       fabs( resMM->val[jd->rowPerm[i]]-resJD->val[i] ) > EPSILON ) {
	    printf( "error in row %i: Differences: %e  %e \n",
		  i, resMM->val[i] -  resCR->val[i], resMM->val[jd->rowPerm[i]] - resJD->val[i] );
	    errcount++;
	 }
      } 
      IF_DEBUG(1){
	 if (errcount ==0) 
	    printf( "Correctness-check for JDS (%d,%d) passed without error\n", mode, blocklen);
	 else 
	    printf( ">>>>> Correctness-check for JDS (%d,%d) yields %d errors <<<<<\n", mode, blocklen, errcount);
      }
   }
   else	{
      for( i = 0; i < jd->nCols; i++ ) {
	 if( fabs( resJD->val[i]-resCR->val[jd->rowPerm[i]] ) > EPSILON ) {
	    printf( "error in row %i: %d %e %e \n", i, jd->rowPerm[i], 
		  resJD->val[i], resCR->val[jd->rowPerm[i]] );
	    exit (1);
	 }
      } 
   }
   oldblocklen = blocklen;
   /*}*/
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
   return 0;
}
