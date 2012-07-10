#include "mpihelper.h"
#include "mymacros.h"
#include <stdio.h>
#include <sys/param.h>

void setupSingleNodeComm( char* hostname, MPI_Comm* single_node_comm, int* me_node) {

   /* return MPI communicator between nodal MPI processes single_node_comm
    * and process rank me_node on local node */

   int i, coreId, me, n_nodes, ierr;
   char **all_hostnames;
   char *all_hn_mem;

   size_t size_ahnm, size_ahn, size_nint;
   int *mymate, *acc_mates;
   
   ierr = MPI_Comm_size ( MPI_COMM_WORLD, &n_nodes );
   ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &me );
   coreId = likwid_processGetProcessorId();

   size_ahnm = (size_t)( MAXHOSTNAMELEN*n_nodes * sizeof(char) );
   size_ahn  = (size_t)( n_nodes    * sizeof(char*) );
   size_nint = (size_t)( n_nodes    * sizeof(int) );

   mymate        = (int*)      malloc( size_nint);
   acc_mates     = (int*)      malloc( size_nint );
   all_hn_mem    = (char*)     malloc( size_ahnm );
   all_hostnames = (char**)    malloc( size_ahn );

   for (i=0; i<n_nodes; i++){
      all_hostnames[i] = &all_hn_mem[i*MAXHOSTNAMELEN];
      mymate[i] = 0;
   }

   /* write local hostname to all_hostnames and share */
   ierr = MPI_Allgather ( hostname, MAXHOSTNAMELEN, MPI_CHAR, 
	 &all_hostnames[0][0], MAXHOSTNAMELEN, MPI_CHAR, MPI_COMM_WORLD );

   /* one process per node writes its global id to all its mates' fields */ 
   if (coreId==0){
      for (i=0; i<n_nodes; i++){
    	 if ( strcmp (hostname, all_hostnames[i]) == 0) mymate[i]=me;
      }
   }   

   ierr = MPI_Allreduce( mymate, acc_mates, n_nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
   /* all processes should now have the rank of their coreId 0 process in their acc_mate field;
    * split into comm groups with this rank as communicator ID */
   ierr = MPI_Comm_split ( MPI_COMM_WORLD, acc_mates[me], me, single_node_comm );
   ierr = MPI_Comm_rank ( *single_node_comm, me_node);

   IF_DEBUG(0) printf("PE%d hat in single_node_comm den rank %d\n", me, *me_node);

   free( mymate );
   free( acc_mates );
   free( all_hn_mem );
   free( all_hostnames );
}

