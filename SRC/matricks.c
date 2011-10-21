#include "matricks.h"
#include "cudafun.h"
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <likwid.h>

//#define _XOPEN_SOURCE 600
#include <errno.h>

#ifdef __sun
#include <sys/processor.h>
#include <sys/procset.h>
#include <sun_prefetch.h>
#endif

#ifdef CUDAKERNEL
#include "my_ellpack.h"
#endif


#define min(A,B) ((A)<(B) ? (A) : (B))

/* ########################################################################## */


static size_t maxMem=0;
static int free0 = 0;
static int free1 = 0;
static int size0 = 0;
static int size1 = 0;
static int now0  = 0;
static int now1  = 0;


/* ########################################################################## */
#ifdef CUDAKERNEL
void vectorDeviceCopyCheck( VECTOR_TYPE* testvec, int me ) {

  /* copy val to gpuval on device in testvec, copy back to temporary and check for consistency*/

  int i;
  double* tmp = NULL;
  size_t bytesize = sizeof(double) * testvec->nRows;
  printf("PE %d: vectorDeviceCopyCheck: size = %lu (%i)\n", me, bytesize, testvec->nRows);
  tmp = (double*) allocateMemory( bytesize, "copycheck");
  for( i = 0; i < testvec->nRows; ++i) tmp[i] = -77.3;

  printf("copying to device...");
  copyHostToDevice( testvec->val_gpu, testvec->val, bytesize );
  printf("done\n");
  printf("copying back to host...");
  copyDeviceToHost( tmp, testvec->val_gpu, bytesize );
  printf("done\n");
  
  for( i=0; i < testvec->nRows; ++i) {
    if( testvec->val[i] != tmp[i] )
      printf("PE %d: error: \tcpu %e , \t gpu %e\n", me, testvec->val[i], tmp[i]);
  }
  free( tmp );
  printf("PE %d: completed copycheck\n", me);
}
#endif


void* allocateMemory( const size_t size, const char* desc ) {

   /* allocate size bytes of posix-aligned memory;
    * check for success and increase global counter */

   size_t boundary = 4096;
   int me, ierr; 

   maxMem = (size_t)(total_mem);
   //printf("Gesamtspeicher [bytes]: %llu\n", maxMem); 

   void* mem;

   MPI_Comm_rank(MPI_COMM_WORLD, &me);

   /*if( allocatedMem + size > maxMem ) {
      fprintf( stderr, "PE%d: allocateMemory: exceeded maximum memory of %llu bytes"
	    "when allocating %-23s\n", me, (uint64)(maxMem), desc );
      printf("PE%d: tried to allocate %llu bytes for %s\n", me, (uint64)(size), desc);
      mypabort("exceeded memory on allocation");
   }*/

   IF_DEBUG(2){
      //if (size>1024.0*1024.0){  
	 printf("PE%d: Allocating %8.2f MB of memory for %-18s  -- %6.3f\n ", 
	       me, size/(1024.0*1024.0), desc, (1.0*allocatedMem)/(1024.0*1024.0) );
         fflush(stdout);
      //}
   }


   if (  (ierr = posix_memalign(  (void**) &mem, boundary, size)) != 0 ) {
      printf("PE%d: Fehler beim Speicherallokieren using posix_memalign\n", me);
      printf("Array to be allocated: %s\n", desc);
      printf("Error ENOMEM: allocated Mem war %6.3f MB\n", (1.0*allocatedMem)/(1024.0*1024.0));
      printf("Errorcode: %s\n", strerror(ierr));
      exit(1);
   }

   if( ! mem ) {
      fprintf( stderr, "allocateMemory: could not allocate %lu bytes of memory "
	    "for %s\n", size, desc );
      abort();
   }
//   if ( get_NUMA_info(&size0, &now0, &size1, &now1) != 0 ) 
//      myabort("failed to retrieve NUMA-info");


   allocatedMem += size;
   IF_DEBUG(2) printf("PE%d: Gegenwaerig allokierter Speicher: %8.2f MB in LD 0/1: (%d /  %d) \n", 
	 me, allocatedMem/(1024.0*1024.0), size0-now0, size1-now1);
   return mem;
}


/* ########################################################################## */

void freeMemory( size_t size, const char* desc, void* this_array ) {

   IF_DEBUG(1) if (size>1024.0*1024.0) printf("Freeing %8.2f MB of memory for %s -- \n", 
	 size/(1024.0*1024.0), desc);

   allocatedMem -= size;
   free (this_array);

}

/* ########################################################################## */

void sweepMemory(int range) {

   int ierr, coreId, me; 
   size_t elements, memcounter;
   size_t size;
   double *tmparray;
   int me_node;
   double individual_mem, acc_mem; 


   maxMem = (size_t)(total_mem);
   coreId = likwid_processGetProcessorId();

   if (range == GLOBAL){
      individual_mem = (double)allocatedMem;
      ierr = MPI_Comm_rank (MPI_COMM_WORLD, &me);
      ierr = MPI_Comm_rank ( single_node_comm, &me_node );
      ierr = MPI_Reduce ( &individual_mem, &acc_mem, 1, MPI_DOUBLE, MPI_SUM, 0, single_node_comm);
      IF_DEBUG(1){
	 printf("Global memsweep\n");
	 printf("PE%d: allokierter Speicher: %6.3f MB\n", me, individual_mem/(1024.0*1024.0));
	 printf("PE:%d -- single-node-id:%d -- core-id:%d\n", me, me_node, coreId);
      }   
   }
   else{
      /* serial call: no global MPI-calls permitted!
       * assume for this case that the used memory on the other PE is negligble */
      IF_DEBUG(1) printf("Serial memsweep\n");
      acc_mem = (double)allocatedMem;
   }

   NUMA_CHECK("before memsweep");

   if (coreId == 0) {

      /* Beruecksichtigt nicht den allokierten Speicher der anderen PEs */
      //elements = ( 0.99*( (double)(maxMem) - acc_mem) ) / (sizeof(double));
      //elements = ( 0.9*( (double)(maxMem) - acc_mem) ) / (sizeof(double));
      elements = ( 0.8*( (double)(maxMem) - acc_mem) ) / (sizeof(double));

      IF_DEBUG(1) printf("PE%d: Sweeping memory with %llu doubles\n", me, (uint64)elements);
      size = (size_t)( elements*sizeof(double) );
      tmparray = (double*) allocateMemory( size, "tmparray" );
      //tmparray = (double*) malloc( size );

      for (memcounter=0; memcounter<elements; memcounter++) 
	 tmparray[memcounter]=0.0;

      IF_DEBUG(1) printf("Freeing memory again ...\n"); 
      freeMemory(size, "sweepMemory tmparray", tmparray);
      //free(tmparray);
      IF_DEBUG(1) printf("... done\n"); 

   }
      NUMA_CHECK("after memsweep");
}

/* ########################################################################## */


MM_TYPE* readMMFile( const char* filename, const double epsilon ) {

   /* allocate and read matrix-market format ascii file;
    * discard values smaller than epsilon;
    * row and col index assumed one-based, converted to zero-based */

   BOOL skippingComments = TRUE, readUntilEndOfLine = FALSE;
   MM_TYPE* mm = (MM_TYPE*) allocateMemory( sizeof( MM_TYPE ), "mm" );
   int e;
   FILE* file;
   size_t size;

   file = fopen( filename, "r" );

   if( ! file ) {
      fprintf( stderr, "readMMFile: could not open file '%s' for reading\n", filename );
      free( mm );
      //mypaborts("readMMFile: could not open file for reading:", filename);
      return NULL;
   }

   /* skip comments ########################################################## */
   while( skippingComments ) {
      char c;
      if( fread( &c, 1, 1, file ) != 1 ) {
	 fprintf( stderr, "readMMFile: error while skipping comments\n" );
	 fclose( file );
	 free( mm );
	 return NULL;
      }

      if( readUntilEndOfLine ) {
	 if( c == '\n' ) readUntilEndOfLine = FALSE;
      }
      else {
	 if( c == '%' ) readUntilEndOfLine = TRUE;
	 else {
	    ungetc( c, file );
	    skippingComments = FALSE;
	 }
      }
   }
   IF_DEBUG(1) printf( "readMMFile: skipping comments done\n" );

   /* read header ############################################################ */
   if( fscanf( file, "%i %i %i\n", &mm->nRows, &mm->nCols, &mm->nEnts ) != 3 ) {
      fprintf( stderr, "readMMFile: error while reading header\n" );
      fclose( file );
      free( mm );
      return NULL;
   }
   IF_DEBUG(1) printf( "readMMFile: nRows %i; nCols %i; nEnts: %i\n", mm->nRows, mm->nCols, mm->nEnts );

   NUMA_CHECK_SERIAL("before placement of MM");

   /* allocate memory for entries */
   size = (size_t)( mm->nEnts * sizeof(NZE_TYPE)  );
   IF_DEBUG(1) printf("Allocating %llu bytes for mm->nze\n", (uint64)size);
   mm->nze = (NZE_TYPE*) allocateMemory(size, "mm->nze" );
   IF_DEBUG(1) printf("...finished\n"); fflush(stdout);

   /* read entries ########################################################### */
   for( e = 0; e < mm->nEnts; e++ ) {
      IF_DEBUG(1) if (e%1000000==0) printf("e=%d\n", e);
      /* mtx format should be one-based (fortran style) ###################### */
      if( fscanf( file, "%i %i %le\n", &mm->nze[e].row, &mm->nze[e].col,
	       &mm->nze[e].val ) != 3 ||
	    mm->nze[e].row < 1 || mm->nze[e].row > mm->nRows ||
	    mm->nze[e].col < 1 || mm->nze[e].col > mm->nCols ) {
	 fprintf( stderr, "readMMFile: error while reading entries:\n" );
   fprintf( stderr, " entry %i: row %i/%i, col %i/%i\n", 
    e, mm->nze[e].row, mm->nRows, mm->nze[e].col, mm->nCols );
	 fclose( file );
	 free( mm->nze );
	 free( mm );
	 return NULL;
      }
      /* row and column index should be zero-based ############################ */
      mm->nze[e].row -= 1;
      mm->nze[e].col -= 1;
      IF_DEBUG(2) printf( "%i %i %e\n", mm->nze[e].row, mm->nze[e].col, mm->nze[e].val );

      /* value smaller than threshold epsilon? ################################ */
      if( fabs( mm->nze[e].val ) < epsilon ) {
        IF_DEBUG(1) printf("entry %i: %i %i %e smaller than eps, skipping...\n", 
          e, mm->nze[e].row, mm->nze[e].col, mm->nze[e].val );
	      e--;
	      mm->nEnts--;
      }
   }
   IF_DEBUG(1) printf( "readMMFile: nEnts (after applying threshold): %i\n", mm->nEnts );fflush(stdout);

   fclose( file );

   IF_DEBUG(1) printf( "readMMFile: done\n" );

   NUMA_CHECK_SERIAL("after placement of MM");

   return mm;
}


/* ########################################################################## */


int compareNZEPos( const void* a, const void* b ) {

   /* comparison function for sorting of matrix entries;
    * sort lesser row id first, then lesser column id first;
    * if MAIN_DIAGONAL_FIRST is defined sort diagonal 
    * before lesser column id */

   int aRow = ((NZE_TYPE*)a)->row,
       bRow = ((NZE_TYPE*)b)->row,
       aCol = ((NZE_TYPE*)a)->col,
       bCol = ((NZE_TYPE*)b)->col;

   if( aRow == bRow ) {
#ifdef MAIN_DIAGONAL_FIRST
      if( aRow == aCol ) aCol = -1;
      if( bRow == bCol ) bCol = -1;
#endif /* MAIN_DIAGONAL_FIRST */
      return aCol - bCol;
   }
   else return aRow - bRow;
}


/* ########################################################################## */


REVBUF_TYPE* revolvingBuffer( const uint64 cachesize, const int pagesize, const int vec_dim ) {

   /* set up buffer for vector of length vec_dim to avoid unrealistic loads from cache;
    * buffer can hold at least as many (numvecs) copies of vector as required to fill cachesize;
    * starting index (i*offset, i<numvecs) of each copy of vector in buffer is aligned to pagesize*/

   int i, me, ierr;
   REVBUF_TYPE* rb;
   size_t size_mem, size_vec;

   ierr = MPI_Comm_rank( MPI_COMM_WORLD, &me );

   rb = (REVBUF_TYPE*) allocateMemory( sizeof( REVBUF_TYPE ), "rb");

   rb->pagesize  = pagesize;
   rb->cachesize = cachesize;
   rb->vecdim    = vec_dim;
   rb->ppvec     = (int)( sizeof(double) * rb->vecdim / rb->pagesize) + 1;
   rb->offset    = ( (int)(rb->ppvec*rb->pagesize) )/sizeof(double);
   rb->numvecs   = (int)( (1024.0*rb->cachesize) / (rb->ppvec*rb->pagesize) ) + 2;
   rb->globdim   = rb->numvecs*rb->offset;

   IF_DEBUG(1){ 
      printf("-----------------------------------------------------\n");
      printf("------   Setup of revolving buffer for PE %2.2d   -----\n", me);
      printf("-----------------------------------------------------\n");
      printf("Elements per vector         : %12d\n", rb->vecdim);
      printf("Cache size per socket [kB]  : %12d\n", rb->cachesize);
      printf("Memorypage size [bytes]     : %12d\n", rb->pagesize);
      printf("Memorypage in cache         : %12d\n", 1024*rb->cachesize/rb->pagesize);
      printf("Memory pages per vector     : %12d\n", rb->ppvec); 
      printf("Vectors in RevBuf           : %12d\n", rb->numvecs);
      printf("Elements in RevBuf          : %12d\n", rb->globdim);
      printf("Memory for RevBuf [MB]      : %12.3f\n", (rb->globdim*sizeof(double))/(1024.0*1024.0));
      printf("Offset between vectors (el) : %12d\n", rb->offset);
      printf("-----------------------------------------------------\n");
   }

   size_mem = (size_t)( rb->globdim * sizeof(double) );
   size_vec = (size_t)( rb->numvecs * sizeof(double*) );

   rb->mem = (double*)   allocateMemory(size_mem, "rb->mem");
   rb->vec = (double **) allocateMemory(size_vec, "rb->vec");

   for (i=0; i<rb->numvecs; i++){
      rb->vec[i] = &rb->mem[i*rb->offset];
   }
   IF_DEBUG(1) {
     printf("----   Finished revolving buffer for PE %2.2d   ----\n", me);
     printf("-----------------------------------------------------\n");
   }
   return rb;
}


/* ########################################################################## */


CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm ) {

   /* allocate and fill CRS-format matrix from MM-type;
    * row and col indices have same base as MM (0-based);
    * elements in row are sorted according to column*/

   int* nEntsInRow;
   int ierr, i, e, pos, nthr=1;
   uint64 hlpaddr;
   int me;

   size_t size_rowOffset, size_col, size_val, size_nEntsInRow;

   double total_mem;
   int coreId;

   /* allocate memory ######################################################## */
   IF_DEBUG(1) printf("Entering convertMMToCRMatrix\n");

   ierr = MPI_Comm_rank (MPI_COMM_WORLD, &me);
   coreId = likwid_processGetProcessorId();
   total_mem = my_amount_of_mem();

#ifdef CMEM
   if (allocatedMem > 0.02*total_mem){
      IF_DEBUG(1) printf("CR setup: Large matrix -- allocated mem=%8.3f MB\n",
	    (float)(allocatedMem)/(1024.0*1024.0));
      sweepMemory(SINGLE);
      IF_DEBUG(1) printf("Nach memsweep\n"); fflush(stdout);
   }
#endif

   size_rowOffset  = (size_t)( (mm->nRows+1) * sizeof( int ) );
   size_col        = (size_t)( mm->nEnts     * sizeof( int ) );
   size_val        = (size_t)( mm->nEnts     * sizeof( double) );
   size_nEntsInRow = (size_t)(  mm->nRows    * sizeof( int ) );

   NUMA_CHECK_SERIAL("before placement of CR");

   CR_TYPE* cr   = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
   cr->rowOffset = (int*)     allocateMemory( size_rowOffset,    "rowOffset" );
   cr->col       = (int*)     allocateMemory( size_col,          "col" );
   cr->val       = (double*)  allocateMemory( size_val,          "val" );
   nEntsInRow    = (int*)     allocateMemory( size_nEntsInRow,   "nEntsInRow" );

   IF_DEBUG(1){
      printf("in convert\n");
      printf("\n mm: %i %i\n\n", mm->nEnts, mm->nRows);

      printf("Anfangsaddresse cr %p\n", cr);
      printf("Anfangsaddresse &cr %p\n", &cr);
      printf("Anfangsaddresse &cr->nEnts %p\n", &(cr->nEnts));
      printf("Anfangsaddresse &cr->nCols %p\n", &(cr->nCols));
      printf("Anfangsaddresse &cr->nRows %p\n", &(cr->nRows));
      printf("Anfangsaddresse &cr->rowOffset %p\n", &(cr->rowOffset));
      printf("Anfangsaddresse &cr->col %p\n", &(cr->col));
      printf("Anfangsaddresse &cr->val %p\n", &(cr->val));
      printf("Anfangsaddresse cr->rowOffset %p\n", cr->rowOffset);
      printf("Anfangsaddresse &(cr->rowOffset[0]) %p\n", &(cr->rowOffset[0]));
   }	

   /* initialize values ###################################################### */
   cr->nRows = mm->nRows;
   cr->nCols = mm->nCols;
   cr->nEnts = mm->nEnts;
   for( i = 0; i < mm->nRows; i++ ) nEntsInRow[i] = 0;

   IF_DEBUG(2){
      hlpaddr = (uint64) ((long)8 * (long)(cr->nEnts-1));
      printf("\ncr->val %p -- %p\n", (&(cr->val))[0], 
	    (void*) ( (uint64)(&(cr->val))[0] + hlpaddr) );
      printf("Anfangsaddresse cr->col   %p\n\n", cr->col);
      fflush(stdout);
   }


   /* sort NZEs with ascending column index for each row ##################### */
   //qsort( mm->nze, mm->nEnts, sizeof( NZE_TYPE ), compareNZEPos );
   IF_DEBUG(1) printf("direkt vor  qsort\n"); fflush(stdout);
   qsort( mm->nze, (size_t)(mm->nEnts), sizeof( NZE_TYPE ), compareNZEPos );
   IF_DEBUG(1) printf("Nach qsort\n"); fflush(stdout);

   /* count entries per row ################################################## */
   for( e = 0; e < mm->nEnts; e++ ) nEntsInRow[mm->nze[e].row]++;

   /* set offsets for each row ############################################### */
   pos = 0;
   cr->rowOffset[0] = pos;
   /*  !!! SUN
   #ifdef _OPENMP
   omp_set_num_threads(128);
   // bind to lower 64
   #pragma omp parallel
   {
     if(processor_bind(P_LWPID,P_MYID,omp_get_thread_num(),NULL)) exit(1);
   }
   #endif
   */
   #ifdef PLACE_CRS
   // NUMA placement for rowOffset
   #pragma omp parallel for schedule(runtime)
   for( i = 0; i < mm->nRows; i++ ) {
      cr->rowOffset[i] = 0;
   }
   #endif

   for( i = 0; i < mm->nRows; i++ ) {
      cr->rowOffset[i] = pos;
      pos += nEntsInRow[i];
   }
   cr->rowOffset[mm->nRows] = pos;

   for( i = 0; i < mm->nRows; i++ ) nEntsInRow[i] = 0;

   #ifdef PLACE_CRS
   // NUMA placement for cr->col[] and cr->val []
   #pragma omp parallel for schedule(runtime)
   for(i=0; i<cr->nRows; ++i) {
     int start = cr->rowOffset[i];
     int end = cr->rowOffset[i+1];
     int j;
     for(j=start; j<end; j++) {
        cr->val[j] = 0.0;
        cr->col[j] = 0;
     }
   }
   #endif //PLACE_CRS
   /* !!! SUN
   #ifdef _OPENMP
   omp_set_num_threads(128);
   // bind to lower 64
   #pragma omp parallel
   {
   if(processor_bind(P_LWPID,P_MYID,omp_get_thread_num(),NULL)) exit(1);
   }
   #endif
   */
   
   /* store values in compressed row data structure ########################## */
   for( e = 0; e < mm->nEnts; e++ ) {
      const int row = mm->nze[e].row,
   	 col = mm->nze[e].col;
      const double val = mm->nze[e].val;
      pos = cr->rowOffset[row] + nEntsInRow[row];
      /* GW 
      cr->col[pos] = col;
      */
      cr->col[pos] = col;
   
      cr->val[pos] = val;
   
      nEntsInRow[row]++;
   }
   /* clean up ############################################################### */
   free( nEntsInRow );
   
   IF_DEBUG(2) {
      for( i = 0; i < mm->nRows+1; i++ ) printf( "rowOffset[%2i] = %3i\n", i, cr->rowOffset[i] );
      for( i = 0; i < mm->nEnts; i++ ) printf( "col[%2i] = %3i, val[%2i] = %e\n", i, cr->col[i], i, cr->val[i] );
   }
   
   IF_DEBUG(1) printf( "convertMMToCRMatrix: done\n" );
   
      NUMA_CHECK_SERIAL("after placement of CR");
   
   return cr;
}


/* ########################################################################## */


int compareNZEPerRow( const void* a, const void* b ) {
   /* comparison function for JD_SORT_TYPE; 
    * sorts rows with higher number of non-zero elements first */

   return  ((JD_SORT_TYPE*)b)->nEntsInRow - ((JD_SORT_TYPE*)a)->nEntsInRow;
}


/* ########################################################################## */


static int* invRowPerm;

int compareNZEForJD( const void* a, const void* b ) {
   const int aRow = invRowPerm[((NZE_TYPE*)a)->row],
	 bRow = invRowPerm[((NZE_TYPE*)b)->row],

	 /*  GeWe
	     aCol = ((NZE_TYPE*)a)->col,
	     bCol = ((NZE_TYPE*)b)->col; 
	     */

	 aCol = invRowPerm[((NZE_TYPE*)a)->col],
	 bCol = invRowPerm[((NZE_TYPE*)b)->col];

   if( aRow == bRow )
      return aCol - bCol;
   else
      return aRow - bRow;
}


/* ########################################################################## */


JD_TYPE* convertMMToJDMatrix( MM_TYPE* mm, int blocklen) {
   /* convert matrix-market format to blocked jagged-diagonal format*/

   JD_SORT_TYPE* rowSort;
   int i, e, pos, oldRow, nThEntryInRow,ib;
   uint64 hlpaddr;

   int block_start, block_end; 
   int diag,diagLen,offset;
   size_t size_rowPerm, size_col, size_val, size_invRowPerm, size_rowSort;
   size_t size_diagOffset;

   FILE *STATFILE;
   char statfilename[50];

   /* allocate memory ######################################################## */
   size_rowPerm    = (size_t)( mm->nRows * sizeof( int ) );
   size_col        = (size_t)( mm->nEnts * sizeof( int ) );
   size_val        = (size_t)( mm->nEnts * sizeof( double) );
   size_invRowPerm = (size_t)( mm->nRows * sizeof( int ) );
   size_rowSort    = (size_t)( mm->nRows * sizeof( JD_SORT_TYPE ) );

   JD_TYPE* jd = (JD_TYPE*)      allocateMemory( sizeof( JD_TYPE ), "jd" );
   jd->rowPerm = (int*)          allocateMemory( size_rowPerm,      "rowPerm" );
   jd->col     = (int*)          allocateMemory( size_col,          "col" );
   jd->val     = (double*)       allocateMemory( size_val,          "val" );
   invRowPerm  = (int*)          allocateMemory( size_invRowPerm,   "invRowPerm" );
   rowSort     = (JD_SORT_TYPE*) allocateMemory( size_rowSort,      "rowSort" );

   /* initialize values ###################################################### */
   jd->nRows = mm->nRows;
   jd->nCols = mm->nCols;
   jd->nEnts = mm->nEnts;
   for( i = 0; i < mm->nRows; i++ ) {
      rowSort[i].row = i;
      rowSort[i].nEntsInRow = 0;
   }


   IF_DEBUG(2){
      hlpaddr = (uint64) ((long)8 * (long)(jd->nEnts-1));
      printf("\njd->val %p -- %p\n", (&(jd->val))[0], 
	    (void*) ( (uint64)(&(jd->val))[0] + hlpaddr) );
      hlpaddr = (uint64) ((long)4 * (long)(jd->nEnts-1));
      printf("Anfangsaddresse jd->col   %p   -- %p \n\n", jd->col, 
	    (void*) ( (uint64)(&(jd->col))[0] + hlpaddr) );
      fflush(stdout);
   }


   /* count entries per row ################################################## */
   for( e = 0; e < mm->nEnts; e++ ) rowSort[mm->nze[e].row].nEntsInRow++;

   /* sort rows with desceding number of NZEs ################################ */
   //qsort( rowSort, mm->nRows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );
   qsort( rowSort, (size_t)(mm->nRows), sizeof( JD_SORT_TYPE  ), compareNZEPerRow );

   /* allocate memory for diagonal offsets */
   jd->nDiags = rowSort[0].nEntsInRow;
   size_diagOffset = (size_t)( (jd->nDiags+1) * sizeof( int ) );
   jd->diagOffset = (int*) allocateMemory(size_diagOffset, "diagOffset" );

   hlpaddr = (uint64) ((long)4 * (long)(jd->nDiags+1));
   IF_DEBUG(2) printf("Anfangsaddresse jd->diagOffset   %p   -- %p \n\n", jd->diagOffset, 
	 (void*) ( (uint64)(&(jd->diagOffset))[0] + hlpaddr) );

   /* set permutation vector for rows ######################################## */
   for( i = 0; i <  mm->nRows; i++ ) {
      invRowPerm[rowSort[i].row] = i;
      jd->rowPerm[i] = rowSort[i].row;
   }

   IF_DEBUG(2) {
      for( i = 0; i <  mm->nRows; i++ ) {
	 printf( "rowPerm[%6i] = %6i; invRowPerm[%6i] = %6i\n", i, jd->rowPerm[i],
	       i, invRowPerm[i] );
      }
   }

   /* sort NZEs with ascending column index for each row ##################### */
   qsort( mm->nze, (size_t)(mm->nEnts), sizeof( NZE_TYPE ), compareNZEForJD );
   //qsort( mm->nze, mm->nEnts, sizeof( NZE_TYPE ), compareNZEForJD );
   IF_DEBUG(2) {
      for( i = 0; i < mm->nEnts; i++ ) {
	 printf( "%i %i %e\n", mm->nze[i].row, mm->nze[i].col, mm->nze[i].val );
      }
   }

   /* number entries per row ################################################# */
   oldRow = -1;
   pos = 0;
   for( e = 0; e < mm->nEnts; e++ ) {
      if( oldRow != mm->nze[e].row ) {
	 pos = 0;
	 oldRow = mm->nze[e].row;
      }
      mm->nze[e].nThEntryInRow = pos;
      pos++;
   }

   /* !!! SUN
   #ifdef _OPENMP
   omp_set_num_threads(128);
      // bind to lower 64
   #pragma omp parallel
   {
   if(processor_bind(P_LWPID,P_MYID,omp_get_thread_num(),NULL)) exit(1);
   }
   #endif
   */
   
   /* store values in jagged diagonal format */
   #ifdef PLACE_JDS
   /* GH: first evaluate shape */
   pos = 0;
   for( nThEntryInRow = 0; nThEntryInRow < jd->nDiags; nThEntryInRow++ ) {
      jd->diagOffset[nThEntryInRow] = pos;
      for( e = 0; e < mm->nEnts; e++ ) {
         if( mm->nze[e].nThEntryInRow == nThEntryInRow ) {
   	 // GH jd->val[pos] = mm->nze[e].val;
   
   	 // GH jd->col[pos] = invRowPerm[mm->nze[e].col]+1;
   	 pos++;
         }
      }
   }
   jd->diagOffset[jd->nDiags] = pos;
   
   
   /* GH: then place jd->col[] and jd->val[] */
   #pragma omp parallel for schedule(runtime) private(block_start, block_end, diag, diagLen, offset, i)
   for(ib=0; ib<jd->nRows; ib+=blocklen) {
      block_start = ib;
      block_end   = min(ib+blocklen-2, jd->nRows-1);
   
      for(diag=0; diag<jd->nDiags; diag++) {
   
         diagLen = jd->diagOffset[diag+1]-jd->diagOffset[diag];
         offset  = jd->diagOffset[diag];
   
         if(diagLen >= block_start) {
   
   	 for(i=block_start; i<= min(block_end,diagLen-1); ++i) {
   	    jd->val[offset+i]=0.0;
   	    jd->col[offset+i]=0;
   	 }
         }
      }
   } 
   /* GH: then fill matrix */
   #endif   // PLACE_JDS
   
   /* !!! SUN
   #ifdef _OPENMP
   omp_set_num_threads(128);
   // bind to lower 64
   #pragma omp parallel
   {
   if(processor_bind(P_LWPID,P_MYID,omp_get_thread_num(),NULL)) exit(1);
   }
   #endif
   */
   pos = 0;
   for( nThEntryInRow = 0; nThEntryInRow < jd->nDiags; nThEntryInRow++ ) {
      jd->diagOffset[nThEntryInRow] = pos;
      for( e = 0; e < mm->nEnts; e++ ) {
         if( mm->nze[e].nThEntryInRow == nThEntryInRow ) {
      	 if (pos > jd->nEnts) printf("wahhhh\n");
      	 jd->val[pos] = mm->nze[e].val;
      
      	 /*  GeWe
      	     jd->col[pos] = mm->nze[e].col; 
      	     */
      
      	 jd->col[pos] = invRowPerm[mm->nze[e].col]+1;
      	 pos++;
            }
      }
   }
   jd->diagOffset[jd->nDiags] = pos;
   
   /* clean up ############################################################### */
   free( rowSort );
   free( invRowPerm );
   IF_DEBUG(1) printf( "convertMMToJDMatrix: done with FORTRAN numbering in jd->col\n" );
   IF_DEBUG(2) {
   for( i = 0; i < mm->nRows;    i++ ) printf( "rowPerm[%6i] = %3i\n", i, jd->rowPerm[i] );
   for( i = 0; i < mm->nEnts;    i++ ) printf( "col[%6i] = %6i, val[%6i] = %e\n", i, jd->col[i], i, jd->val[i] );
   for( i = 0; i < jd->nDiags+1; i++ ) printf( "diagOffset[%6i] = %6i\n", i, jd->diagOffset[i] );
   }
   
   IF_DEBUG(1) printf( "convertMMToJDMatrix: done\n" );
   /*  sprintf(statfilename, "./intermediate3.dat");
       if ((STATFILE = fopen(statfilename, "w"))==NULL){
       printf("Fehler beim Oeffnen von %s\n", statfilename);
       exit(1);
       }
       for (i = 0 ; i < cr->nEnts ; i++) fprintf(STATFILE,"%i %25.16g\n",i, (cr->val)[i]);
       fclose(STATFILE);
       */
   return jd;
}


/* ########################################################################## */
   
   
BOOL multiplyMMWithVector( VECTOR_TYPE* res, const MM_TYPE* mm, const VECTOR_TYPE* vec ) {
   /* perform reference SpMVM with matrix-market type matrix;
    * requires indices to be 0-based */

   int i;

   IF_DEBUG(1) {
      if( res->nRows != vec->nRows ||
	    res->nRows != mm->nCols ) {
	 fprintf( stderr, "multiplyMMWithVector: operand sizes do not match\n" );
	 return FALSE;
      }
   }

   for( i = 0; i < res->nRows; i++ ) res->val[i] = 0.0;

   for( i = 0; i < mm->nEnts; i++ ) {
      res->val[mm->nze[i].row] += mm->nze[i].val * vec->val[mm->nze[i].col];
   }

   return TRUE;
}


/* ########################################################################## */


BOOL multiplyCRWithVector( VECTOR_TYPE* res, const CR_TYPE* cr, const VECTOR_TYPE* vec ) {
   /* perform reference SpMVM with CRS type matrix;
    * requires indices to be 0-based (unlike fortran_crs)*/

   int i, j;

   IF_DEBUG(1) {
      if( res->nRows != vec->nRows ||
	    res->nRows != cr->nCols ) {
	 fprintf( stderr, "multiplyCRWithVector: operand sizes do not match\n" );
	 return FALSE;
      }
   }

#pragma omp parallel for private(j)
   for( i = 0; i < res->nRows; i++ ) {
      res->val[i] = 0.0;
      for( j = cr->rowOffset[i]; j < cr->rowOffset[i+1]; j++ ) {
	 res->val[i] += cr->val[j] * vec->val[cr->col[j]];
      }
   }

   return TRUE;
}


/* ########################################################################## */


BOOL multiplyJDWithVector( VECTOR_TYPE* res, const JD_TYPE* jd, const VECTOR_TYPE* vec ) {
   /* perform reference SoMVM with JDS type matrix;
    * requires indices to be 0-based */

   int d, i;

   IF_DEBUG(1) {
      if( res->nRows != vec->nRows ||
	    res->nRows != jd->nCols ) {
	 fprintf( stderr, "multiplyJDWithVector: operand sizes do not match\n" );
	 return FALSE;
      }
   }

   for( i = 0; i < res->nRows; i++ ) res->val[i] = 0.0;

   for( d = 0; d < jd->nDiags; d++ ) {
      const int diagOffset = jd->diagOffset[d],
	    diagLen    = jd->diagOffset[d+1] - diagOffset;
#pragma ivdep
#pragma vector always
#pragma vector aligned
      for( i = 0; i < diagLen; i++ ) {
	 res->val[i] += jd->val[diagOffset+i] * vec->val[jd->col[diagOffset+i]];
      }
   }

   return TRUE;
}


/* ########################################################################## */


void crColIdToFortran( CR_TYPE* cr ) {
   /* increase column index of CRS matrix by 1;
    * check index after conversion */

   int i;
   fprintf(stdout,"CR to Fortran: for %i entries in %i rows\n",
    cr->rowOffset[cr->nRows], cr->nRows); fflush(stdout);
   for( i = 0; i < cr->rowOffset[cr->nRows]; ++i) {
     cr->col[i] += 1;
     if( cr->col[i] < 1 || cr->col[i] > cr->nCols) {
       fprintf(stderr, "error in crColIdToFortran: index out of bounds\n");
       exit(1);
     }
   }
   fprintf(stdout,"CR to Fortran: completed %i entries\n",
    i); fflush(stdout);
}


/* ########################################################################## */


void crColIdToC( CR_TYPE* cr ) {
   /* decrease column index of CRS matrix by 1;
    * check index after conversion */

   int i;

   for( i = 0; i < cr->rowOffset[cr->nRows]; ++i) {
     cr->col[i] -= 1;
     if( cr->col[i] < 0 || cr->col[i] > cr->nCols-1) {
       fprintf(stderr, "error in crColIdToC: index out of bounds\n");
       exit(1);
     }
   }
}


/* ########################################################################## */


VECTOR_TYPE* newVector( const int nRows ) {
   /* allocate VECTOR_TYPE to hold nRows double entries;
    * if CUDAKERNEL is defined also allocate matching array on device */

   VECTOR_TYPE* vec;
   size_t size_val;

   size_val = (size_t)( nRows * sizeof(double) );
   vec = (VECTOR_TYPE*) allocateMemory( sizeof( VECTOR_TYPE ), "vec");

   #ifdef CUDAKERNEL
   vec->val_gpu = allocDeviceMemory( size_val );
   #endif

   vec->val = (double*) allocateMemory( size_val, "vec->val");
   vec->nRows = nRows;

   return vec;
}


/* ########################################################################## */


void freeVector( VECTOR_TYPE* const vec ) {
   if( vec ) {
      freeMemory( (size_t)(vec->nRows*sizeof(double)), "vec->val",  vec->val );
      #ifdef CUDAKERNEL
      freeDeviceMemory( vec->val_gpu );
      #endif
      free( vec );
   }
}


/* ########################################################################## */

void freeMMMatrix( MM_TYPE* const mm ) {
   if( mm ) {
      freeMemory( (size_t)(mm->nEnts*sizeof(NZE_TYPE)), "mm->nze", mm->nze );
      freeMemory( (size_t)sizeof(MM_TYPE), "mm", mm );
   }
}

/* ########################################################################## */


void freeCRMatrix( CR_TYPE* const cr ) {

   size_t size_rowOffset, size_col, size_val;

   if( cr ) {
 
      size_rowOffset  = (size_t)( (cr->nRows+1) * sizeof( int ) );
      size_col        = (size_t)( cr->nEnts     * sizeof( int ) );
      size_val        = (size_t)( cr->nEnts     * sizeof( double) );

      freeMemory( size_rowOffset,  "cr->rowOffset", cr->rowOffset );
      freeMemory( size_col,        "cr->col",       cr->col );
      freeMemory( size_val,        "cr->val",       cr->val );
      freeMemory( sizeof(CR_TYPE), "cr",            cr );

   }
}


/* ########################################################################## */


void freeJDMatrix( JD_TYPE* const jd ) {
   if( jd ) {
      free( jd->rowPerm );
      free( jd->diagOffset );
      free( jd->col );
      free( jd->val );
      free( jd );
   }
}


void freeLcrpType( LCRP_TYPE* const lcrp ) {
  if( lcrp ) {
    free( lcrp->lnEnts );
    free( lcrp->lnRows );
    free( lcrp->lfEnt );
    free( lcrp->lfRow );
    free( lcrp->wishes );
    free( lcrp->wishlist_mem );
    free( lcrp->wishlist );
    free( lcrp->dues );
    free( lcrp->duelist_mem );
    free( lcrp->duelist );
    free( lcrp->due_displ );
    free( lcrp->wish_displ );
    free( lcrp->hput_pos );
    free( lcrp->val );
    free( lcrp->col );
    free( lcrp->row_ptr );
    free( lcrp->lrow_ptr );
    free( lcrp->lrow_ptr_l );
    free( lcrp->lrow_ptr_r );
    free( lcrp->lcol );
    free( lcrp->rcol );
    free( lcrp->lval );
    free( lcrp->rval );
#ifdef CUDAKERNEL
    freeCUDAELRMatrix( lcrp->celr );
    freeCUDAELRMatrix( lcrp->lcelr );
    freeCUDAELRMatrix( lcrp->rcelr );
#endif
    free( lcrp );
  }
}
