#define _XOPEN_SOURCE 600
#include "matricks.h"
#include "spmvm_util.h"

#include <string.h>
#include <libgen.h>
#include <complex.h>
#include <mmio.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


#define DIAG_NEW (char)0
#define DIAG_OK (char)1
#define DIAG_INVALID (char)2
static int allocatedMem;

static int compareNZEPos( const void* a, const void* b ) 
{

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

static int compareNZEOrgPos( const void* a, const void* b ) 
{
	return  ((JD_SORT_TYPE*)a)->row - ((JD_SORT_TYPE*)b)->row;
}

int compareNZEPerRow( const void* a, const void* b ) 
{
	/* comparison function for JD_SORT_TYPE; 
	 * sorts rows with higher number of non-zero elements first */

	return  ((JD_SORT_TYPE*)b)->nEntsInRow - ((JD_SORT_TYPE*)a)->nEntsInRow;
}

int isMMfile(const char *filename) 
{

	FILE *file = fopen( filename, "r" );

	if( ! file ) {
		ABORT("Could not open file in isMMfile!");
	}

	const char *keyword="%%MatrixMarket";
	char *readkw = (char *)allocateMemory((strlen(keyword)+1)*sizeof(char),"readkw");
	if (NULL == fgets(readkw,strlen(keyword)+1,file))
		return 0;

	int cmp = strcmp(readkw,keyword);

	free(readkw);
	return cmp==0?1:0;
}

void* allocateMemory( const size_t size, const char* desc ) 
{

	/* allocate size bytes of posix-aligned memory;
	 * check for success and increase global counter */

	size_t boundary = 1024;
	int ierr;

	void* mem;

	DEBUG_LOG(2,"Allocating %8.2f MB of memory for %-18s  -- %6.3f", 
			size/(1024.0*1024.0), desc, (1.0*allocatedMem)/(1024.0*1024.0));

	if (  (ierr = posix_memalign(  (void**) &mem, boundary, size)) != 0 ) {
		ABORT("Error while allocating using posix_memalign: %s",strerror(ierr));
	}

	if( ! mem ) {
		ABORT("Error in memory allocation of %lu bytes for %s",size,desc);
	}

	allocatedMem += size;
	return mem;
}

void freeMemory( size_t size, const char* desc, void* this_array ) 
{

	DEBUG_LOG(2,"Freeing %8.2f MB of memory for %s", size/(1024.*1024.), desc);

	allocatedMem -= size;
	free (this_array);

}

MM_TYPE * readMMFile(const char* filename ) 
{

	MM_typecode matcode;
	FILE *f;
	int i;
	MM_TYPE* mm = (MM_TYPE*) malloc( sizeof( MM_TYPE ) );

	if ((f = fopen(filename, "r")) == NULL) 
		exit(1);

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

#ifdef COMPLEX
	if (!mm_is_complex(matcode))
		DEBUG_LOG(0,"Warning! The library has been built for complex data "
				"but the MM file contains mat_data_t data. Casting...");
#else
	if (mm_is_complex(matcode))
		DEBUG_LOG(0,"Warning! The library has been built for real data "
				"but the MM file contains complex data. Casting...");
#endif



	if ((mm_read_mtx_crd_size(f, &mm->nRows, &mm->nCols, &mm->nEnts)) !=0)
		exit(1);


	mm->nze = (NZE_TYPE *)malloc(mm->nEnts*sizeof(NZE_TYPE));

	if (!mm_is_complex(matcode)) {
		for (i=0; i<mm->nEnts; i++)
		{
#ifdef DOUBLE
			double re;
			fscanf(f, "%d %d %lg\n", &mm->nze[i].row, &mm->nze[i].col, &re);
#else
			float re;
			fscanf(f, "%d %d %g\n", &mm->nze[i].row, &mm->nze[i].col, &re);
#endif
#ifdef COMPLEX	
			mm->nze[i].val = re+I*0;
#else
			mm->nze[i].val = re;
#endif
			mm->nze[i].col--;  /* adjust from 1-based to 0-based */
			mm->nze[i].row--;
		}
	} else {

		for (i=0; i<mm->nEnts; i++)
		{
#ifdef DOUBLE
			double re,im;
			fscanf(f, "%d %d %lg %lg\n", &mm->nze[i].row, &mm->nze[i].col, &re,
					&im);
#else
			float re,im;
			fscanf(f, "%d %d %g %g\n", &mm->nze[i].row, &mm->nze[i].col, &re,
					&im);
#endif
#ifdef COMPLEX	
			mm->nze[i].val = re+I*im;
#else
			mm->nze[i].val = re;
#endif
			mm->nze[i].col--;  /* adjust from 1-based to 0-based */
			mm->nze[i].row--;
		}

	}


	if (f !=stdin) fclose(f);
	return mm;
}


CR_TYPE * readCRbinFile(const char* path, int rowPtrOnly, int detectDiags)
{

	CR_TYPE *cr;
	int i, j;
	int datatype;
	FILE* RESTFILE;

	cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
	DEBUG_LOG(1,"Reading binary CRS matrix %s",path);

	if ((RESTFILE = fopen(path, "rb"))==NULL){
		ABORT("Could not open binary CRS file %s",path);
	}

	fread(&datatype, sizeof(int), 1, RESTFILE);
	fread(&cr->nRows, sizeof(int), 1, RESTFILE);
	fread(&cr->nCols, sizeof(int), 1, RESTFILE);
	fread(&cr->nEnts, sizeof(int), 1, RESTFILE);

	if (datatype != DATATYPE_DESIRED) {
		DEBUG_LOG(0,"Warning in %s:%d! The library has been built for %s data but"
				" the file contains %s data. Casting...\n",__FILE__,__LINE__,
				DATATYPE_NAMES[DATATYPE_DESIRED],DATATYPE_NAMES[datatype]);
	}

	DEBUG_LOG(2,"Allocate memory for cr->rowOffset");
	cr->rowOffset = (int*)    allocateMemory( (cr->nRows+1)*sizeof(int), "rowOffset" );

	DEBUG_LOG(1,"NUMA-placement for cr->rowOffset");
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < cr->nRows+1; i++ ) {
		cr->rowOffset[i] = 0;
	}

	DEBUG_LOG(2,"Reading array with row-offsets");
	fread(&cr->rowOffset[0],        sizeof(int),    cr->nRows+1, RESTFILE);


	if (!rowPtrOnly) {
		cr->constDiags = NULL;

		if (detectDiags) {
			int bandwidth = 2;//cr->nCols/2;
			int nDiags = 2*bandwidth + 1;

			mat_data_t *diagVals = (mat_data_t *)allocateMemory(nDiags*sizeof(mat_data_t),"diagVals");

			char *diagStatus = (char *)allocateMemory(nDiags*sizeof(char),"diagStatus");
			for (i=0; i<nDiags; i++) diagStatus[i] = DIAG_NEW;

			int *diagEnts = (int *)allocateMemory(nDiags*sizeof(int),"diagEnts");
			for (i=0; i<nDiags; i++) diagEnts[i] = 0;

			DEBUG_LOG(1,"Detecting constant subdiagonals within a band of width %d",bandwidth);
			int *tmpcol = (int *)allocateMemory(cr->nEnts*sizeof(int),"tmpcol");
			mat_data_t *tmpval = (mat_data_t *)allocateMemory(cr->nEnts*sizeof(mat_data_t),"tmpval");

			int pfile;
			pfile = open(path,O_RDONLY);
			int offs = 4*sizeof(int)+(cr->nRows+1)*sizeof(int);
			int idx = 0;
			for (i=0; i<cr->nRows; ++i) {
				for(j = cr->rowOffset[i] ; j < cr->rowOffset[i+1] ; j++) {
					pread(pfile,&tmpcol[idx],sizeof(int),offs+idx*sizeof(int));
					pread(pfile,&tmpval[idx],sizeof(mat_data_t),offs+cr->nEnts*sizeof(int)+idx*sizeof(mat_data_t));
					if (ABS(tmpcol[idx]-i) <= bandwidth) { // in band

						int didx = tmpcol[idx]-i+bandwidth; // index of diagonal
						if (diagStatus[didx] == DIAG_NEW) { // first time
							diagVals[didx] =  tmpval[idx];
							diagStatus[didx] = (char)1;
							diagEnts[didx] = 1;
							DEBUG_LOG(2,"Diag %d initialized with %f",didx,diagVals[didx]);
						} else if (diagStatus[didx] == DIAG_OK) { // diag initialized
							if (EQUALS(diagVals[didx],tmpval[idx])) {
								diagEnts[didx]++;
							} else {
								DEBUG_LOG(2,"Diag %d discontinued in row %d: %f (was %f)",didx,i,tmpval[idx],diagVals[didx]);
								diagStatus[didx] = DIAG_INVALID;
							}
						}
					}
					idx++;
				}
			}

			cr->nConstDiags = 0;

			for (i=0; i<bandwidth+1; i++) { // lower subdiagonals AND diagonal
				if (diagStatus[i] == DIAG_OK && diagEnts[i] == cr->nCols-bandwidth+i) {
					DEBUG_LOG(1,"The %d-th subdiagonal is constant with %f",bandwidth-i,diagVals[i]);
					cr->nConstDiags++;
					cr->constDiags = realloc(cr->constDiags,sizeof(CONST_DIAG)*cr->nConstDiags);
					cr->constDiags[cr->nConstDiags-1].idx = bandwidth-i;
					cr->constDiags[cr->nConstDiags-1].val = diagVals[i];
					cr->constDiags[cr->nConstDiags-1].len = diagEnts[i];
					cr->constDiags[cr->nConstDiags-1].minRow = i-bandwidth;
					cr->constDiags[cr->nConstDiags-1].maxRow = cr->nRows-1;
					DEBUG_LOG(1,"range: %d..%d",i-bandwidth,cr->nRows-1);
				}
			}
			for (i=bandwidth+1; i<nDiags ; i++) { // upper subdiagonals
				if (diagStatus[i] == DIAG_OK && diagEnts[i] == cr->nCols+bandwidth-i) {
					DEBUG_LOG(1,"The %d-th subdiagonal is constant with %f",-bandwidth+i,diagVals[i]);
					cr->nConstDiags++;
					cr->constDiags = realloc(cr->constDiags,sizeof(CONST_DIAG)*cr->nConstDiags);
					cr->constDiags[cr->nConstDiags-1].idx = -bandwidth+i;
					cr->constDiags[cr->nConstDiags-1].val = diagVals[i];
					cr->constDiags[cr->nConstDiags-1].len = diagEnts[i];
					cr->constDiags[cr->nConstDiags-1].minRow = 0;
					cr->constDiags[cr->nConstDiags-1].maxRow = cr->nRows-1-i+bandwidth;
					DEBUG_LOG(1,"range: %d..%d",0,cr->nRows-1-i+bandwidth);
				}
			}

			if (cr->nConstDiags == 0) 
			{ // no constant diagonals found
				cr->val = tmpval;
				cr->col = tmpcol;
			} 
			else {
				int d = 0;

				DEBUG_LOG(1,"Adjusting the number of matrix entries, old: %d",cr->nEnts);
				for (d=0; d<cr->nConstDiags; d++) {
					cr->nEnts -= cr->constDiags[d].len;
				}
				DEBUG_LOG(1,"Adjusting the number of matrix entries, new: %d",cr->nEnts);

				DEBUG_LOG(2,"Allocate memory for cr->col and cr->val");
				cr->col       = (int*)    allocateMemory( cr->nEnts * sizeof(int),  "col" );
				cr->val       = (mat_data_t*) allocateMemory( cr->nEnts * sizeof(mat_data_t),  "val" );

				//TODO NUMA
				int *newRowOffset = (int *)allocateMemory((cr->nRows+1)*sizeof(int),"newRowOffset");

				idx = 0;
				int oidx = 0; // original idx in tmp arrays
				for (i=0; i<cr->nRows; ++i) {
					newRowOffset[i] = idx;
					for(j = cr->rowOffset[i] ; j < cr->rowOffset[i+1]; j++) {
						if (ABS(tmpcol[oidx]-i) <= bandwidth) { // in band
							int diagFound = 0;
							for (d=0; d<cr->nConstDiags; d++) {
								if (tmpcol[oidx]-i == cr->constDiags[d].idx) { // do not store constant diagonal
									oidx++;
									diagFound = 1;
									break;
								} 
							}
							if (!diagFound) {
								cr->col[idx] = tmpcol[oidx];
								cr->val[idx] = tmpval[oidx];
								idx++;
								oidx++;
							}

						} else {
							cr->col[idx] = tmpcol[oidx];
							cr->val[idx] = tmpval[oidx];
							idx++;
							oidx++;

						}
					}
				}
				free(cr->rowOffset);
				free(tmpval);
				free(tmpcol);

				newRowOffset[cr->nRows] = cr->nEnts;
				cr->rowOffset = newRowOffset;
			}
			close(pfile);
		} 
		else {

			DEBUG_LOG(2,"Allocate memory for cr->col and cr->val");
			cr->col       = (int*)    allocateMemory( cr->nEnts * sizeof(int),  "col" );
			cr->val       = (mat_data_t*) allocateMemory( cr->nEnts * sizeof(mat_data_t),  "val" );

			DEBUG_LOG(1,"NUMA-placement for cr->val and cr->col");
#pragma omp parallel for schedule(runtime)
			for(i = 0 ; i < cr->nRows; ++i) {
				for(j = cr->rowOffset[i] ; j < cr->rowOffset[i+1] ; j++) {
					cr->val[j] = 0.0;
					cr->col[j] = 0;
				}
			}


			DEBUG_LOG(2,"Reading array with column indices");

			// TODO fread => pread
			fread(&cr->col[0], sizeof(int), cr->nEnts, RESTFILE);
			DEBUG_LOG(2,"Reading array with values");
			if (datatype == DATATYPE_DESIRED)
			{
				fread(&cr->val[0], sizeof(mat_data_t), cr->nEnts, RESTFILE);
			} 
			else 
			{
				switch (datatype) {
					case DATATYPE_FLOAT:
						{
							float *tmp = (float *)allocateMemory(
									cr->nEnts*sizeof(float), "tmp");
							fread(tmp, sizeof(float), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case DATATYPE_DOUBLE:
						{
							double *tmp = (double *)allocateMemory(
									cr->nEnts*sizeof(double), "tmp");
							fread(tmp, sizeof(double), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case DATATYPE_COMPLEX_FLOAT:
						{
							_Complex float *tmp = (_Complex float *)allocateMemory(
									cr->nEnts*sizeof(_Complex float), "tmp");
							fread(tmp, sizeof(_Complex float), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case DATATYPE_COMPLEX_DOUBLE:
						{
							_Complex double *tmp = (_Complex double *)allocateMemory(
									cr->nEnts*sizeof(_Complex double), "tmp");
							fread(tmp, sizeof(_Complex double), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
				}
			}
		}
	}
	fclose(RESTFILE);

	return cr;
}

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm ) 
{

	/* allocate and fill CRS-format matrix from MM-type;
	 * row and col indices have same base as MM (0-based);
	 * elements in row are sorted according to column*/

	int* nEntsInRow;
	int i, e, pos;
	uint64 hlpaddr;

	size_t size_rowOffset, size_col, size_val, size_nEntsInRow;


	/* allocate memory ######################################################## */
	IF_DEBUG(1) printf("Entering convertMMToCRMatrix\n");


	size_rowOffset  = (size_t)( (mm->nRows+1) * sizeof( int ) );
	size_col        = (size_t)( mm->nEnts     * sizeof( int ) );
	size_val        = (size_t)( mm->nEnts     * sizeof( mat_data_t) );
	size_nEntsInRow = (size_t)(  mm->nRows    * sizeof( int ) );


	CR_TYPE* cr   = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
	cr->rowOffset = (int*)     allocateMemory( size_rowOffset,    "rowOffset" );
	cr->col       = (int*)     allocateMemory( size_col,          "col" );
	cr->val       = (mat_data_t*)  allocateMemory( size_val,          "val" );
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

	/* store values in compressed row data structure ########################## */
	for( e = 0; e < mm->nEnts; e++ ) {
		const int row = mm->nze[e].row,
			  col = mm->nze[e].col;
		const mat_data_t val = mm->nze[e].val;
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
		for( i = 0; i < mm->nEnts; i++ ) printf( "col[%2i] = %3i, val[%2i] = %e+i%e\n", i, cr->col[i], i, REAL(cr->val[i]),IMAG(cr->val[i]) );
	}

	IF_DEBUG(1) printf( "convertMMToCRMatrix: done\n" );


	return cr;
}

void crColIdToFortran( CR_TYPE* cr ) 
{
	/* increase column index of CRS matrix by 1;
	 * check index after conversion */

	int i;
	IF_DEBUG(1) {
		printf("CR to Fortran: for %i entries in %i rows\n",
				cr->rowOffset[cr->nRows], cr->nRows); 
		fflush(stdout);
	}

	for( i = 0; i < cr->rowOffset[cr->nRows]; ++i) {
		cr->col[i] += 1;
		if( cr->col[i] < 1 || cr->col[i] > cr->nCols) {
			fprintf(stderr, "error in crColIdToFortran: index out of bounds\n");
			exit(1);
		}
	}
	IF_DEBUG(1) {
		printf("CR to Fortran: completed %i entries\n",
				i); 
		fflush(stdout);
	}
}

void crColIdToC( CR_TYPE* cr ) 
{
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

void freeMMMatrix( MM_TYPE* const mm ) 
{
	if( mm ) {
		freeMemory( (size_t)(mm->nEnts*sizeof(NZE_TYPE)), "mm->nze", mm->nze );
		freeMemory( (size_t)sizeof(MM_TYPE), "mm", mm );
	}
}

int pad(int nRows, int padding) 
{
	int nRowsPadded;

	if(  nRows % padding != 0) {
		nRowsPadded = nRows + padding - nRows % padding;
	} else {
		nRowsPadded = nRows;
	}
	return nRowsPadded;
}

BJDS_TYPE * CRStoBJDS(CR_TYPE *cr) 
{
	int i,j,c;
	BJDS_TYPE *mv;

	mv = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"mv");

	mv->nRows = cr->nRows;
	mv->nNz = cr->nEnts;
	mv->nEnts = 0;
	mv->nRowsPadded = pad(mv->nRows,BJDS_LEN);


	int nChunks = mv->nRowsPadded/BJDS_LEN;
	mv->chunkStart = (int *)allocateMemory((nChunks+1)*sizeof(int),"mv->chunkStart");
	mv->chunkMin = (int *)allocateMemory((nChunks)*sizeof(int),"mv->chunkMin");
	mv->rowLen = (int *)allocateMemory((mv->nRowsPadded)*sizeof(int),"mv->chunkMin");
	mv->chunkStart[0] = 0;




	int chunkMax = 0;
	int chunkMin = cr->nCols;
	int curChunk = 1;
	int rowLen;

	for (i=0; i<mv->nRowsPadded; i++) {
		if (i<cr->nRows)
			rowLen = cr->rowOffset[i+1]-cr->rowOffset[i];
		else
			rowLen = 0;

		mv->rowLen[i] = rowLen;
		chunkMax = rowLen>chunkMax?rowLen:chunkMax;
		chunkMin = rowLen<chunkMin?rowLen:chunkMin;
#if defined(MIC) && BJDS_LEN==8
		/* The gather instruction is only available on MIC. Therefore, the
		   access to the index vector has to be 512bit-aligned only on MIC.
		   Also, the innerloop in the BJDS-kernel has to be 2-way unrolled
		   only on this case. ==> The number of columns of one chunk does
		   not have to be a multiple of two in the other cases. */
		chunkMax = chunkMax%2==0?chunkMax:chunkMax+1;
#endif

		if ((i+1)%BJDS_LEN == 0) {
			mv->nEnts += BJDS_LEN*chunkMax;
			mv->chunkStart[curChunk] = mv->chunkStart[curChunk-1]+BJDS_LEN*chunkMax;
			mv->chunkMin[curChunk-1] = chunkMin;

			chunkMax = 0;
			chunkMin = cr->nCols;
			curChunk++;
		}
	}

	mv->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*mv->nEnts,"mv->val");
	mv->col = (int *)allocateMemory(sizeof(int)*mv->nEnts,"mv->col");

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<mv->nRowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<(mv->chunkStart[c+1]-mv->chunkStart[c])/BJDS_LEN; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				mv->val[mv->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				mv->col[mv->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
	}



	for (c=0; c<nChunks; c++) {
		int chunkLen = (mv->chunkStart[c+1]-mv->chunkStart[c])/BJDS_LEN;

		for (j=0; j<chunkLen; j++) {

			for (i=0; i<BJDS_LEN; i++) {
				if (j<mv->rowLen[c*BJDS_LEN+i]) {

					mv->val[mv->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rowOffset[c*BJDS_LEN+i]+j];
					mv->col[mv->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rowOffset[c*BJDS_LEN+i]+j];
				} else {
					mv->val[mv->chunkStart[c]+j*BJDS_LEN+i] = 0.0;
					mv->col[mv->chunkStart[c]+j*BJDS_LEN+i] = 0;
				}
				//printf("%f ",mv->val[mv->chunkStart[c]+j*BJDS_LEN+i]);


			}
		}
	}


	return mv;
}

BJDS_TYPE * CRStoSTBJDS(CR_TYPE *cr, int rowWise, unsigned int sortBlock, int **rowPerm, int **invRowPerm, mat_flags_t flags) 
{
	int i,j,c;
	BJDS_TYPE *tbjds;


	tbjds = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"mv");

	tbjds->nRows = cr->nRows;
	tbjds->nNz = cr->nEnts;
	tbjds->nEnts = 0;
	tbjds->nRowsPadded = pad(tbjds->nRows,BJDS_LEN);


	int nChunks = tbjds->nRowsPadded/BJDS_LEN;
	tbjds->chunkStart = (int *)allocateMemory((nChunks+1)*sizeof(int),"tbjds->chunkStart");
	tbjds->chunkMin = (int *)allocateMemory((nChunks)*sizeof(int),"tbjds->chunkMin");
	tbjds->chunkLen = (int *)allocateMemory((nChunks)*sizeof(int),"tbjds->chunkMin");
	tbjds->rowLen = (int *)allocateMemory((tbjds->nRowsPadded)*sizeof(int),"tbjds->chunkMin");
	tbjds->chunkStart[0] = 0;


	int chunkMin = cr->nCols;
	int chunkLen = 0;
	int curChunk = 1;
	int rowLen;
	tbjds->nu = 0.;

	JD_SORT_TYPE* rowSort;
	/* get max number of entries in one row ###########################*/
	rowSort = (JD_SORT_TYPE*) allocateMemory( cr->nRows * sizeof( JD_SORT_TYPE ),
			"rowSort" );

	for (c=0; c<cr->nRows/(int)sortBlock; c++) // TODO signed vs unsigned 
	{
		for( i = c*(int)sortBlock; i < (c+1)*(int)sortBlock; i++ ) 
		{
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = cr->rowOffset[i+1] - cr->rowOffset[i];
		} 

		qsort( rowSort+c*sortBlock, sortBlock, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );
	}
	for( i = c*sortBlock; i < cr->nRows; i++ ) 
	{ // remainder
		rowSort[i].row = i;
		rowSort[i].nEntsInRow = cr->rowOffset[i+1] - cr->rowOffset[i];
	}


	/* sort within same rowlength with asceding row number #################### */
/*	i=0;
	while(i < cr->nRows) {
		int start = i;

		j = rowSort[start].nEntsInRow;
		while( i<cr->nRows && rowSort[i].nEntsInRow >= j ) 
			++i;

		DEBUG_LOG(1,"sorting over %i rows (%i): %i - %i\n",i-start,j, start, i-1);
		qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
	}

	for(i=1; i < cr->nRows; ++i) {
		if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
			printf("Error in row %i: descending row number\n",i);
	}*/

	*rowPerm = (int *)allocateMemory(cr->nRows*sizeof(int),"sbjds->rowPerm");
	*invRowPerm = (int *)allocateMemory(cr->nRows*sizeof(int),"sbjds->invRowPerm");

	for(i=0; i < cr->nRows; ++i) {
		/* invRowPerm maps an index in the permuted system to the original index,
		 * rowPerm gets the original index and returns the corresponding permuted position.
		 */
		if( rowSort[i].row >= cr->nRows ) DEBUG_LOG(0,"error: invalid row number %i in %i\n",rowSort[i].row, i); 

		(*invRowPerm)[i] = rowSort[i].row;
		(*rowPerm)[rowSort[i].row] = i;
	}
//	for(i=0; i < cr->nRows; ++i) printf("%d\n",(*invRowPerm)[i]);


	for (i=0; i<tbjds->nRowsPadded; i++) {
		if (i<cr->nRows)
			rowLen = rowSort[i].nEntsInRow;
		else
			rowLen = 0;

		tbjds->rowLen[i] = rowLen;
		tbjds->nEnts += rowLen;

		chunkMin = rowLen<chunkMin?rowLen:chunkMin;
		chunkLen = rowLen>chunkLen?rowLen:chunkLen;

		if ((i+1)%BJDS_LEN == 0) {
			tbjds->chunkStart[curChunk] = tbjds->nEnts;
			tbjds->chunkMin[curChunk-1] = chunkMin;
			tbjds->chunkLen[curChunk-1] = chunkLen;

			tbjds->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->nCols;
			chunkLen = 0;
			curChunk++;
		}
	}
	tbjds->nu /= (double)nChunks;

	tbjds->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*tbjds->nEnts,"tbjds->val");
	tbjds->col = (int *)allocateMemory(sizeof(int)*tbjds->nEnts,"tbjds->col");

	//printf("nEnts: %d\n",tbjds->nEnts);

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<tbjds->nRowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<tbjds->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				tbjds->val[tbjds->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				tbjds->col[tbjds->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
		int rem = tbjds->chunkStart[c] + tbjds->chunkMin[c]*BJDS_LEN;
		for (i=0; i<BJDS_LEN; i++)
		{
			for (j=tbjds->chunkMin[c]; j<tbjds->rowLen[c*BJDS_LEN+i]; j++)
			{
				tbjds->val[rem] = 0.;
				tbjds->col[rem++] = 0;
			}
		}
	}
	for (c=0; c<tbjds->nRowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		// store block
		for (j=0; j<tbjds->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				tbjds->val[tbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j];
if (flags & SPM_PERMUTECOLUMNS)
				tbjds->col[tbjds->chunkStart[c]+j*BJDS_LEN+i] = (*rowPerm)[cr->col[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j]];
else
				tbjds->col[tbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j];
			}
		}

		// store remainder
		int rem = tbjds->chunkStart[c] + tbjds->chunkMin[c]*BJDS_LEN;
		if (rowWise) 
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				for (j=tbjds->chunkMin[c]; j<tbjds->rowLen[c*BJDS_LEN+i]; j++)
				{
					tbjds->val[rem] = cr->val[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j];
if (flags & SPM_PERMUTECOLUMNS)
					tbjds->col[rem++] = (*rowPerm)[cr->col[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j]];
else
					tbjds->col[rem++] = cr->col[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j];
				}
			}
		} else 
		{
			for (j=tbjds->chunkMin[c]; j<tbjds->chunkLen[c]; j++)
			{
				for (i=0; i<BJDS_LEN; i++)
				{
					if (j<tbjds->rowLen[c*BJDS_LEN+i] ) {
						tbjds->val[rem] = cr->val[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j];
if (flags & SPM_PERMUTECOLUMNS)
						tbjds->col[rem++] = (*rowPerm)[cr->col[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j]];
else
						tbjds->col[rem++] = cr->col[cr->rowOffset[(*invRowPerm)[c*BJDS_LEN+i]]+j];
					}
				}
			}
		}
	}


	return tbjds;
}

BJDS_TYPE * CRStoTBJDS(CR_TYPE *cr, int rowWise) 
{
	int i,j,c;
	BJDS_TYPE *mv;

	mv = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"mv");

	mv->nRows = cr->nRows;
	mv->nNz = cr->nEnts;
	mv->nEnts = 0;
	mv->nRowsPadded = pad(mv->nRows,BJDS_LEN);


	int nChunks = mv->nRowsPadded/BJDS_LEN;
	mv->chunkStart = (int *)allocateMemory((nChunks+1)*sizeof(int),"mv->chunkStart");
	mv->chunkMin = (int *)allocateMemory((nChunks)*sizeof(int),"mv->chunkMin");
	mv->chunkLen = (int *)allocateMemory((nChunks)*sizeof(int),"mv->chunkMin");
	mv->rowLen = (int *)allocateMemory((mv->nRowsPadded)*sizeof(int),"mv->chunkMin");
	mv->chunkStart[0] = 0;


	int chunkMin = cr->nCols;
	int chunkLen = 0;
	int curChunk = 1;
	int rowLen;
	mv->nu = 0.;

	for (i=0; i<mv->nRowsPadded; i++) {
		if (i<cr->nRows)
			rowLen = cr->rowOffset[i+1]-cr->rowOffset[i];
		else
			rowLen = 0;

		mv->rowLen[i] = rowLen;
		mv->nEnts += rowLen;

		chunkMin = rowLen<chunkMin?rowLen:chunkMin;
		chunkLen = rowLen>chunkLen?rowLen:chunkLen;

		if ((i+1)%BJDS_LEN == 0) {
			mv->chunkStart[curChunk] = mv->nEnts;
			mv->chunkMin[curChunk-1] = chunkMin;
			mv->chunkLen[curChunk-1] = chunkLen;

			mv->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->nCols;
			chunkLen = 0;
			curChunk++;
		}
	}
	mv->nu /= (double)nChunks;

	mv->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*mv->nEnts,"mv->val");
	mv->col = (int *)allocateMemory(sizeof(int)*mv->nEnts,"mv->col");

	//printf("nEnts: %d\n",mv->nEnts);

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<mv->nRowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<mv->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				mv->val[mv->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				mv->col[mv->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
		int rem = mv->chunkStart[c] + mv->chunkMin[c]*BJDS_LEN;
		for (i=0; i<BJDS_LEN; i++)
		{
			for (j=mv->chunkMin[c]; j<mv->rowLen[c*BJDS_LEN+i]; j++)
			{
				mv->val[rem] = 0.;
				mv->col[rem++] = 0;
			}
		}
	}
	for (c=0; c<mv->nRowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		// store block
		for (j=0; j<mv->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				mv->val[mv->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rowOffset[c*BJDS_LEN+i]+j];
				mv->col[mv->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rowOffset[c*BJDS_LEN+i]+j];
			}
		}

		// store remainder
		int rem = mv->chunkStart[c] + mv->chunkMin[c]*BJDS_LEN;
		if (rowWise) 
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				for (j=mv->chunkMin[c]; j<mv->rowLen[c*BJDS_LEN+i]; j++)
				{
					mv->val[rem] = cr->val[cr->rowOffset[c*BJDS_LEN+i]+j];
					mv->col[rem++] = cr->col[cr->rowOffset[c*BJDS_LEN+i]+j];
				}
			}
		} else 
		{
			for (j=mv->chunkMin[c]; j<mv->chunkLen[c]; j++)
			{
				for (i=0; i<BJDS_LEN; i++)
				{
					if (j<mv->rowLen[c*BJDS_LEN+i] ) {
						mv->val[rem] = cr->val[cr->rowOffset[c*BJDS_LEN+i]+j];
						mv->col[rem++] = cr->col[cr->rowOffset[c*BJDS_LEN+i]+j];
					}
				}
			}
		}
	}


	return mv;
}

BJDS_TYPE * CRStoSBJDS(CR_TYPE *cr, int **rowPerm, int **invRowPerm, mat_flags_t flags) 
{
	int i,j,c;
	BJDS_TYPE *sbjds;
	JD_SORT_TYPE* rowSort;
	/* get max number of entries in one row ###########################*/
	rowSort = (JD_SORT_TYPE*) allocateMemory( cr->nRows * sizeof( JD_SORT_TYPE ),
			"rowSort" );

	for( i = 0; i < cr->nRows; i++ ) {
		rowSort[i].row = i;
		rowSort[i].nEntsInRow = cr->rowOffset[i+1] - cr->rowOffset[i];
	} 

	qsort( rowSort, cr->nRows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );

	/* sort within same rowlength with asceding row number #################### */
	i=0;
	while(i < cr->nRows) {
		int start = i;

		j = rowSort[start].nEntsInRow;
		while( i<cr->nRows && rowSort[i].nEntsInRow >= j ) 
			++i;

		DEBUG_LOG(1,"sorting over %i rows (%i): %i - %i\n",i-start,j, start, i-1);
		qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
	}

	for(i=1; i < cr->nRows; ++i) {
		if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
			printf("Error in row %i: descending row number\n",i);
	}



	sbjds = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"sbjds");

	sbjds->nRows = cr->nRows;
	sbjds->nNz = cr->nEnts;
	sbjds->nEnts = 0;
	sbjds->nRowsPadded = pad(sbjds->nRows,BJDS_LEN);

	*rowPerm = (int *)allocateMemory(cr->nRows*sizeof(int),"sbjds->rowPerm");
	*invRowPerm = (int *)allocateMemory(cr->nRows*sizeof(int),"sbjds->invRowPerm");

	for(i=0; i < cr->nRows; ++i) {
		/* invRowPerm maps an index in the permuted system to the original index,
		 * rowPerm gets the original index and returns the corresponding permuted position.
		 */
		if( rowSort[i].row >= cr->nRows ) DEBUG_LOG(0,"error: invalid row number %i in %i\n",rowSort[i].row, i); 

		(*invRowPerm)[i] = rowSort[i].row;
		(*rowPerm)[rowSort[i].row] = i;
	}

	int nChunks = sbjds->nRowsPadded/BJDS_LEN;
	sbjds->chunkStart = (int *)allocateMemory((nChunks+1)*sizeof(int),"sbjds->chunkStart");
	sbjds->chunkStart[0] = 0;




	int chunkMax = 0;
	int curChunk = 1;

	for (i=0; i<sbjds->nRows; i++) {
		int rowLen = rowSort[i].nEntsInRow;
		chunkMax = rowLen>chunkMax?rowLen:chunkMax;
#ifdef MIC
		/* The gather instruction is only available on MIC. Therefore, the
		   access to the index vector has to be 512bit-aligned only on MIC.
		   Also, the innerloop in the BJDS-kernel has to be 2-way unrolled
		   only on this case. ==> The number of columns of one chunk does
		   not have to be a multiple of two in the other cases. */
		chunkMax = chunkMax%2==0?chunkMax:chunkMax+1;
#endif

		if ((i+1)%BJDS_LEN == 0) {
			sbjds->nEnts += BJDS_LEN*chunkMax;
			sbjds->chunkStart[curChunk] = sbjds->chunkStart[curChunk-1]+BJDS_LEN*chunkMax;

			chunkMax = 0;
			curChunk++;
		}
	}

	sbjds->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*sbjds->nEnts,"sbjds->val");
	sbjds->col = (int *)allocateMemory(sizeof(int)*sbjds->nEnts,"sbjds->val");

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<sbjds->nRowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<(sbjds->chunkStart[c+1]-sbjds->chunkStart[c])/BJDS_LEN; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				sbjds->val[sbjds->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				sbjds->col[sbjds->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
	}



	for (c=0; c<nChunks; c++) {
		int chunkLen = (sbjds->chunkStart[c+1]-sbjds->chunkStart[c])/BJDS_LEN;

		for (j=0; j<chunkLen; j++) {

			for (i=0; i<BJDS_LEN; i++) {
				int row = c*BJDS_LEN+i;
				int rowLen = rowSort[row].nEntsInRow;
				if (j<rowLen) {

					sbjds->val[sbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rowOffset[(*invRowPerm)[row]]+j];
if (flags & SPM_PERMUTECOLUMNS)
					sbjds->col[sbjds->chunkStart[c]+j*BJDS_LEN+i] = (*rowPerm)[cr->col[cr->rowOffset[(*invRowPerm)[row]]+j]];
else
					sbjds->col[sbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rowOffset[(*invRowPerm)[row]]+j];
				} else {
					sbjds->val[sbjds->chunkStart[c]+j*BJDS_LEN+i] = 0.0;
					sbjds->col[sbjds->chunkStart[c]+j*BJDS_LEN+i] = 0;
				}
				//	printf("%f ",sbjds->val[sbjds->chunkStart[c]+j*BJDS_LEN+i]);


			}
		}
	}


	return sbjds;
}
