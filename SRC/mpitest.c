#include <omp.h>
#include <math.h>
#include <sys/times.h>
//#include <unistd.h>
//#include <sched.h>

#include <likwid.h>
#include "matricks.h"
#include <mpi.h>

/* Global variables */
int error_count, acc_error_count;

int coreId=2;
#ifdef LIKWID
int RegionId;
int numberOfThreads = 1;
int numberOfRegions = 1;
int threadId = 0;
#endif


void checkMPIerror(int ierr, int rnk) {
  int len;
  char message[100];
  if( MPI_SUCCESS != ierr ) {
    MPI_Error_string(ierr, message, &len);
    printf("PE%i: %s\n", rnk, message);
  }
}


int main( int nArgs, char* arg[] ) {

   int i, N;

   int numthreads;
   char hostname[50];

   /* Number of nodes */
   int n_nodes;

   /* Error-code for MPI */
   int ierr;

   /* Rank of this node */
   int me;

   /* Memory page size in bytes*/
   const int pagesize=4096;

   int homelength;
   int remotelength;
   int* lengths;
   int* offsets;

   int required_threading_level;
   int provided_threading_level;

   VECTOR_TYPE* sendvec;
   VECTOR_TYPE* recvec;

   required_threading_level = MPI_THREAD_MULTIPLE;

   ierr = MPI_Init_thread(&nArgs, &arg, required_threading_level, 
	 &provided_threading_level );
   checkMPIerror( ierr, me );

   ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );
   ierr = MPI_Comm_size ( MPI_COMM_WORLD, &n_nodes );

#if 1 
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
         printf ("Rank %d/%d Thread (%d/%d) running on node %s core %d \n",
                 me,n_nodes, omp_get_thread_num(), numthreads, hostname, coreId);
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
#endif

   /* setup MPI scatterv testcase */
   lengths = (int*)malloc(n_nodes*sizeof(int)); // allocateMemory( n_nodes*sizeof(int), "lengths");
   offsets = (int*)malloc(n_nodes*sizeof(int)); // allocateMemory( n_nodes*sizeof(int), "offsetss");
   for(i=0; i < n_nodes; ++i) {
     lengths[i] = offsets[i] = 22;
   }

   if( 0 == me ) {
      if ( nArgs < 3 ) { 
         printf("expected input: [Hlength] [Rlength]\n");
         exit(1);
      }

      homelength = atoi( arg[1] );
      remotelength = atoi( arg[2] );


      lengths[0] = homelength;
      offsets[0] = 0;
      for(i=1; i < n_nodes; ++i) { 
        lengths[i] = remotelength;
        offsets[i] = offsets[i-1] + lengths[i-1];
      }

      N = homelength + (n_nodes-1)*remotelength;



      sendvec = newVector( N );
      for( i=0; i < N; ++i) sendvec->val[i] = (double)i*3.3 + 1.3;

      printf("PE%i; home: %i, remote: %i, total: %i\n", me, homelength, remotelength, N);
   } else {
      sendvec = newVector( 1 );
   }

   ierr = MPI_Barrier(MPI_COMM_WORLD);
   checkMPIerror( ierr, me );
      
      printf("PE%i: bcast lengths \n",me);
      ierr = MPI_Bcast(lengths, n_nodes, MPI_INT, 0, MPI_COMM_WORLD);
      checkMPIerror( ierr, me );
      ierr = MPI_Bcast(offsets, n_nodes, MPI_INT, 0, MPI_COMM_WORLD);
      checkMPIerror( ierr, me );

   for(i=0; i < n_nodes; ++i) {
        printf("PE%i: lengths[%i] = %i\n", me, i, lengths[i]);
        printf("PE%i: offsets[%i] = %i\n", me, i, offsets[i]);
   }

   recvec = newVector( lengths[me] );

   ierr = MPI_Barrier(MPI_COMM_WORLD);
   checkMPIerror( ierr, me );
   
   printf("PE%i: scattering\n", me); fflush(stdout);

   ierr = MPI_Scatterv ( sendvec->val, lengths, offsets, MPI_DOUBLE, 
    recvec->val, lengths[me], MPI_DOUBLE, 0, MPI_COMM_WORLD );
   
   checkMPIerror( ierr, me );
   printf("PE%i: scattered\n", me); fflush(stdout);
   
   freeVector( sendvec );
   freeVector( recvec );
   
   free( lengths );
   free( offsets );
   
   ierr = MPI_Finalize();

   return 0;
}

