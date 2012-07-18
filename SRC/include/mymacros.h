#ifndef _MY_MACROS_H_
#define _MY_MACROS_H_

#define SPM_KERNEL_FULL 0
#define SPM_KERNEL_LOCAL 1
#define SPM_KERNEL_REMOTE 2
#define AXPY_KERNEL 3
#define DOTPROD_KERNEL 4
#define VECSCAL_KERNEL 5


#define DEBUG 0 
#define xMAIN_DIAGONAL_FIRST

#define PJDS_CHUNK_HEIGHT 32

#define GLOBAL 0

#define IS_DAXPY 1
#define IS_AX 0

#define EPSILON 1e-6

#define LIKWID_MARKER

#define EQUAL_NZE  1
#define EQUAL_LNZE 2

#define IF_DEBUG(level) if( DEBUG >= level )

#define BOOL  int
#define TRUE  (1==1)
#define FALSE (1==0)


#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)<(y)?(y):(x))
#endif

/*******************************************************************************
 **************                Makros fuer Zeitmessung           ***************
 ******************************************************************************/

#ifdef hades

#define PAS_CYCLE_START
#define PAS_CYCLE_STOP
#define AS_CYCLE_START
#define AS_CYCLE_STOP
#define EVALUATE_CYCLES(identifier,entries)
#define AS_WRITE_TIME(identifier)

#else


#define PAS_CYCLE_START \
   ierr = MPI_Barrier(MPI_COMM_WORLD);\
if (me == 0) for_timing_start_asm_( &asm_cyclecounter); \
ierr = MPI_Barrier(MPI_COMM_WORLD);

#define PAS_CYCLE_STOP \
   ierr = MPI_Barrier(MPI_COMM_WORLD);\
if (me == 0) for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles); \
ierr = MPI_Barrier(MPI_COMM_WORLD);

#define PAS_WRITE_TIME(identifier) {\
   ierr = MPI_Barrier(MPI_COMM_WORLD);\
if (me == 0){ \
   for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles); \
   asm_cycles = asm_cycles - p_cycles4measurement; \
   time_it_took = (1.0*asm_cycles)/clockfreq; \
   printf("%-23s [s] : %12.3f\n", identifier, time_it_took ); }} \
ierr = MPI_Barrier(MPI_COMM_WORLD);

#define PAS_GET_TIME \
   ierr = MPI_Barrier(MPI_COMM_WORLD);			     \
if (me == 0){						     \
   for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles);   \
   asm_cycles = asm_cycles - p_cycles4measurement;	     \
   time_it_took = (1.0*asm_cycles)/clockfreq; }	     \
ierr = MPI_Barrier(MPI_COMM_WORLD);

#define PAS_EVALUATE_CYCLES(identifier, entries)  \
   ierr = MPI_Barrier(MPI_COMM_WORLD);\
if (me == 0){ \
   for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles); \
   asm_cycles = asm_cycles - p_cycles4measurement; \
   time_it_took = (1.0*asm_cycles)/clockfreq; \
   printf("\t\t\t\t Cycles per nze  | Total time per MVM [ms]   |   MFlop/s\n"); \
   printf("%-23s:   %15.3f   %20.5f   %18.3f\n", identifier,		\
	 (real)asm_cycles/((real)N_MULTS*(real)entries),	\
	 1000*time_it_took/((real)N_MULTS),			\
	 2.0e-6*(real)N_MULTS*(real)entries/time_it_took);	\
   printf("Gesamtzahl Cycles:  %25.13lg %15.3f\n", (real)asm_cycles, time_it_took); } \
ierr = MPI_Barrier(MPI_COMM_WORLD);


#define AS_CYCLE_START for_timing_start_asm_( &asm_cyclecounter);

#define AS_CYCLE_STOP for_timing_stop_asm_( &asm_cyclecounter, &asm_cycles );

#define EVALUATE_CYCLES(identifier, entries)	 \
   asm_cycles = asm_cycles - cycles4measurement;	 \
time_it_took = (1.0*asm_cycles)/clockfreq;				\
printf("\t\t\t\t Cycles per nze  | Total time per MVM [ms]   |   MFlop/s\n"); \
printf("%-23s:   %15.3f   %20.5f   %18.3f\n", identifier,		\
      (real)asm_cycles/((real)N_MULTS*(real)entries),		\
      1000*time_it_took/((real)N_MULTS),				\
      2.0e-6*(real)N_MULTS*(real)entries/time_it_took);		\
printf("Gesamtzahl Cycles:  %25.13lg %15.3f\n", (real)asm_cycles, time_it_took);

#define AS_WRITE_TIME(identifier){		 \
   asm_cycles = asm_cycles - cycles4measurement; \
time_it_took = (1.0*asm_cycles)/clockfreq; \
printf("%-23s [s] : %12.3f\n", identifier, time_it_took );}

#endif


#define diagnose_performance(identifier, entries) 
/* #define diagnose_performance(identifier, entries) \
   printf( "Average MVM for %-40s : %8.2e s %8.2f MFLOPs\n", \
   identifier, (real)(stopTime-startTime)/(N_MULTS), \ 
   ( (real)(2.0*(real)(entries)) / \
   (1000.0*1000.0*(real)(stopTime-startTime)/ (N_MULTS)) ));
   */

#define NUMA_CHECK(identifier)                                                 \
         {                                                                     \
         real naccmem;                                                       \
         int ierr, me, me_node, coreId;                                        \
         int ns0=0;                                                            \
         int ns1=0;                                                            \
         int nf0=0;                                                            \
         int nf1=0;                                                            \
         real individual_mem;                                                \
         individual_mem = (real)allocatedMem;                                \
         ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );                         \
	 ierr = MPI_Comm_rank ( single_node_comm, &me_node );                  \
	 ierr = MPI_Reduce ( &individual_mem, &naccmem, 1, MPI_DOUBLE,         \
                             MPI_SUM, 0, single_node_comm);                    \
         coreId = likwid_processGetProcessorId();                              \
         if (coreId==0){                                                       \
	    IF_DEBUG(1) printf("PE:%d acc_mem=%6.3f\n", me, naccmem/(1024.0*1024.0));      \
	    if ( get_NUMA_info(&ns0, &nf0, &ns1, &nf1) != 0 )                  \
	       mypabort("failed to retrieve NUMA-info");                       \
	    IF_DEBUG(1) printf("PE%d: %23s: NUMA-LD-0: %5d (%5d )MB free\n",               \
                    me, identifier, nf0, ns0);                                 \
	    IF_DEBUG(1) printf("PE%d: %23s: NUMA-LD-1: %5d (%5d )MB free\n",               \
                    me, identifier, nf1, ns1);                                 \
            fflush(stdout);                                                    \
            }                                                                  \
         } 

#define NUMA_CHECK_SERIAL(identifier)                                          \
         {                                                                     \
         real naccmem;                                                       \
         int ierr, me;                                                         \
         int ns0=0;                                                            \
         int ns1=0;                                                            \
         int nf0=0;                                                            \
         int nf1=0;                                                            \
         naccmem = (real) allocatedMem;                                      \
         ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );                         \
	 IF_DEBUG(1) printf("PE:%d acc_mem=%6.3f\n", me, naccmem/(1024.0*1024.0));         \
	 if ( get_NUMA_info(&ns0, &nf0, &ns1, &nf1) != 0 )                     \
	    mypabort("failed to retrieve NUMA-info");                          \
	 IF_DEBUG(1) printf("PE%d: %23s: NUMA-LD-0: %5d (%5d )MB free\n",                  \
                 me, identifier, nf0, ns0);                                    \
	 IF_DEBUG(1) printf("PE%d: %23s: NUMA-LD-1: %5d (%5d )MB free\n",                  \
                 me, identifier, nf1, ns1);                                    \
         fflush(stdout);                                                       \
         } 



#endif // _MY_MACROS_H_
