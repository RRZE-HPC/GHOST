#define _XOPEN_SOURCE 600
#include "matricks.h"
#include "ghost.h"
#include "ghost_util.h"

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


int compareNZEPos( const void* a, const void* b ) 
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

int compareNZEOrgPos( const void* a, const void* b ) 
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
		ABORT("Could not open file in isMMfile: %s",filename);
	}

	const char *keyword="%%MatrixMarket";
	char *readkw = (char *)allocateMemory((strlen(keyword)+1)*sizeof(char),"readkw");
	if (NULL == fgets(readkw,strlen(keyword)+1,file))
		return 0;

	int cmp = strcmp(readkw,keyword);

	free(readkw);
	return cmp==0?1:0;
}

/*
MM_TYPE * readMMFile(const char* filename ) 
{
	MM_typecode matcode;
	FILE *f;
	ghost_midx_t i;
	MM_TYPE* mm = (MM_TYPE*) malloc( sizeof( MM_TYPE ) );

	if ((f = fopen(filename, "r")) == NULL) 
		exit(1);

	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

#ifdef GHOST_MAT_COMPLEX
	if (!mm_is_complex(matcode))
		DEBUG_LOG(0,"Warning! The library has been built for complex data "
				"but the MM file contains real data. Casting...");
#else
	if (mm_is_complex(matcode))
		DEBUG_LOG(0,"Warning! The library has been built for real data "
				"but the MM file contains complex data. Casting...");
#endif



	if ((mm_read_mtx_crd_size(f, &mm->nrows, &mm->ncols, &mm->nEnts)) !=0)
		exit(1);


	mm->nze = (NZE_TYPE *)malloc(mm->nEnts*sizeof(NZE_TYPE));

	if (!mm_is_complex(matcode)) {
		for (i=0; i<mm->nEnts; i++)
		{
#ifdef GHOST_MAT_DP
			double re;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %lg\n", &mm->nze[i].row, &mm->nze[i].col, &re);
#else
			float re;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %g\n", &mm->nze[i].row, &mm->nze[i].col, &re);
#endif
#ifdef GHOST_MAT_COMPLEX
			mm->nze[i].val = re+I*0;
#else
			mm->nze[i].val = re;
#endif
			mm->nze[i].col--;  // adjust from 1-based to 0-based 
			mm->nze[i].row--;
		}
	} else {

		for (i=0; i<mm->nEnts; i++)
		{
#ifdef GHOST_MAT_DP
			double re,im;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %lg %lg\n", &mm->nze[i].row, &mm->nze[i].col, &re,
					&im);
#else
			float re,im;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %g %g\n", &mm->nze[i].row, &mm->nze[i].col, &re,
					&im);
#endif
#ifdef GHOST_MAT_COMPLEX	
			mm->nze[i].val = re+I*im;
#else
			mm->nze[i].val = re;
#endif
			mm->nze[i].col--; //  adjust from 1-based to 0-based
			mm->nze[i].row--;
		}

	}


	if (f !=stdin) fclose(f);
	return mm;
}


CR_TYPE * readCRbinFile(const char* path, int rowPtrOnly, int detectDiags)
{

	CR_TYPE *cr;
	ghost_midx_t i, j;
	int datatype;
	FILE* RESTFILE;

	cr = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
	DEBUG_LOG(1,"Reading binary CRS matrix %s",path);

	if ((RESTFILE = fopen(path, "rb"))==NULL){
		ABORT("Could not open binary CRS file %s",path);
	}

	fread(&datatype, sizeof(int), 1, RESTFILE);
	fread(&cr->nrows, sizeof(int), 1, RESTFILE);
	fread(&cr->ncols, sizeof(int), 1, RESTFILE);
	fread(&cr->nEnts, sizeof(int), 1, RESTFILE);

	DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",cr->nrows,cr->ncols,cr->nEnts);

	if (datatype != GHOST_MY_MDATATYPE) {
		DEBUG_LOG(0,"Warning in %s:%d! The library has been built for %s data but"
				" the file contains %s data. Casting...\n",__FILE__,__LINE__,
				ghost_datatypeName(GHOST_MY_MDATATYPE),ghost_datatypeName(datatype));
	}

	DEBUG_LOG(2,"Allocate memory for cr->rpt");
	cr->rpt = (ghost_midx_t *)    allocateMemory( (cr->nrows+1)*sizeof(ghost_midx_t), "rpt" );

	DEBUG_LOG(1,"NUMA-placement for cr->rpt");
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < cr->nrows+1; i++ ) {
		cr->rpt[i] = 0;
	}

	DEBUG_LOG(2,"Reading array with row-offsets");
	fread(&cr->rpt[0],        sizeof(int),    cr->nrows+1, RESTFILE);


	if (!rowPtrOnly) {
		cr->constDiags = NULL;

		if (detectDiags) {
			ghost_midx_t bandwidth = 2;//cr->ncols/2;
			ghost_midx_t nDiags = 2*bandwidth + 1;

			ghost_mdat_t *diagVals = (ghost_mdat_t *)allocateMemory(nDiags*sizeof(ghost_mdat_t),"diagVals");

			char *diagStatus = (char *)allocateMemory(nDiags*sizeof(char),"diagStatus");
			for (i=0; i<nDiags; i++) diagStatus[i] = DIAG_NEW;

			int *diagEnts = (int *)allocateMemory(nDiags*sizeof(int),"diagEnts");
			for (i=0; i<nDiags; i++) diagEnts[i] = 0;

			DEBUG_LOG(1,"Detecting constant subdiagonals within a band of width %"PRmatIDX,bandwidth);
			ghost_midx_t *tmpcol = (ghost_midx_t *)allocateMemory(cr->nEnts*sizeof(ghost_midx_t),"tmpcol");
			ghost_mdat_t *tmpval = (ghost_mdat_t *)allocateMemory(cr->nEnts*sizeof(ghost_mdat_t),"tmpval");

			int pfile;
			pfile = open(path,O_RDONLY);
			int offs = 4*sizeof(int)+(cr->nrows+1)*sizeof(int);
			int idx = 0;
			for (i=0; i<cr->nrows; ++i) {
				for(j = cr->rpt[i] ; j < cr->rpt[i+1] ; j++) {
					pread(pfile,&tmpcol[idx],sizeof(int),offs+idx*sizeof(int));
					pread(pfile,&tmpval[idx],sizeof(ghost_mdat_t),offs+cr->nEnts*sizeof(int)+idx*sizeof(ghost_mdat_t));
					if ((ghost_midx_t)llabs(tmpcol[idx]-i) <= bandwidth) { // in band

						int didx = tmpcol[idx]-i+bandwidth; // index of diagonal
						if (diagStatus[didx] == DIAG_NEW) { // first time
							diagVals[didx] =  tmpval[idx];
							diagStatus[didx] = (char)1;
							diagEnts[didx] = 1;
							DEBUG_LOG(2,"Diag %d initialized with %f",didx,MREAL(diagVals[didx]));
						} else if (diagStatus[didx] == DIAG_OK) { // diag initialized
							if (MEQUALS(diagVals[didx],tmpval[idx])) {
								diagEnts[didx]++;
							} else {
								DEBUG_LOG(2,"Diag %d discontinued in row %"PRmatIDX": %f (was %f)",didx,i,MREAL(tmpval[idx]),MREAL(diagVals[didx]));
								diagStatus[didx] = DIAG_INVALID;
							}
						}
					}
					idx++;
				}
			}

			cr->nConstDiags = 0;

			for (i=0; i<bandwidth+1; i++) { // lower subdiagonals AND diagonal
				if (diagStatus[i] == DIAG_OK && diagEnts[i] == cr->ncols-bandwidth+i) {
					DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f",bandwidth-i,MREAL(diagVals[i]));
					cr->nConstDiags++;
					cr->constDiags = realloc(cr->constDiags,sizeof(CONST_DIAG)*cr->nConstDiags);
					cr->constDiags[cr->nConstDiags-1].idx = bandwidth-i;
					cr->constDiags[cr->nConstDiags-1].val = diagVals[i];
					cr->constDiags[cr->nConstDiags-1].len = diagEnts[i];
					cr->constDiags[cr->nConstDiags-1].minRow = i-bandwidth;
					cr->constDiags[cr->nConstDiags-1].maxRow = cr->nrows-1;
					DEBUG_LOG(1,"range: %"PRmatIDX"..%"PRmatIDX,i-bandwidth,cr->nrows-1);
				}
			}
			for (i=bandwidth+1; i<nDiags ; i++) { // upper subdiagonals
				if (diagStatus[i] == DIAG_OK && diagEnts[i] == cr->ncols+bandwidth-i) {
					DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f",-bandwidth+i,MREAL(diagVals[i]));
					cr->nConstDiags++;
					cr->constDiags = realloc(cr->constDiags,sizeof(CONST_DIAG)*cr->nConstDiags);
					cr->constDiags[cr->nConstDiags-1].idx = -bandwidth+i;
					cr->constDiags[cr->nConstDiags-1].val = diagVals[i];
					cr->constDiags[cr->nConstDiags-1].len = diagEnts[i];
					cr->constDiags[cr->nConstDiags-1].minRow = 0;
					cr->constDiags[cr->nConstDiags-1].maxRow = cr->nrows-1-i+bandwidth;
					DEBUG_LOG(1,"range: 0..%"PRmatIDX,cr->nrows-1-i+bandwidth);
				}
			}

			if (cr->nConstDiags == 0) 
			{ // no constant diagonals found
				cr->val = tmpval;
				cr->col = tmpcol;
			} 
			else {
				ghost_midx_t d = 0;

				DEBUG_LOG(1,"Adjusting the number of matrix entries, old: %"PRmatNNZ,cr->nEnts);
				for (d=0; d<cr->nConstDiags; d++) {
					cr->nEnts -= cr->constDiags[d].len;
				}
				DEBUG_LOG(1,"Adjusting the number of matrix entries, new: %"PRmatNNZ,cr->nEnts);

				DEBUG_LOG(2,"Allocate memory for cr->col and cr->val");
				cr->col       = (ghost_midx_t*)    allocateMemory( cr->nEnts * sizeof(ghost_midx_t),  "col" );
				cr->val       = (ghost_mdat_t*) allocateMemory( cr->nEnts * sizeof(ghost_mdat_t),  "val" );

				//TODO NUMA
				ghost_midx_t *newRowOffset = (ghost_midx_t *)allocateMemory((cr->nrows+1)*sizeof(ghost_midx_t),"newRowOffset");

				idx = 0;
				ghost_midx_t oidx = 0; // original idx in tmp arrays
				for (i=0; i<cr->nrows; ++i) {
					newRowOffset[i] = idx;
					for(j = cr->rpt[i] ; j < cr->rpt[i+1]; j++) {
						if ((ghost_midx_t)llabs(tmpcol[oidx]-i) <= bandwidth) { // in band
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
				free(cr->rpt);
				free(tmpval);
				free(tmpcol);

				newRowOffset[cr->nrows] = cr->nEnts;
				cr->rpt = newRowOffset;
			}
			close(pfile);
		} 
		else {

			DEBUG_LOG(2,"Allocate memory for cr->col and cr->val");
			cr->col       = (ghost_midx_t *)    allocateMemory( cr->nEnts * sizeof(ghost_midx_t),  "col" );
			cr->val       = (ghost_mdat_t *) allocateMemory( cr->nEnts * sizeof(ghost_mdat_t),  "val" );

			DEBUG_LOG(1,"NUMA-placement for cr->val and cr->col");
#pragma omp parallel for schedule(runtime)
			for(i = 0 ; i < cr->nrows; ++i) {
				for(j = cr->rpt[i] ; j < cr->rpt[i+1] ; j++) {
					cr->val[j] = 0.0;
					cr->col[j] = 0;
				}
			}


			DEBUG_LOG(2,"Reading array with column indices");

			// TODO fread => pread
			fread(&cr->col[0], sizeof(int), cr->nEnts, RESTFILE);
			DEBUG_LOG(2,"Reading array with values");
			if (datatype == GHOST_MY_MDATATYPE)
			{
				fread(&cr->val[0], sizeof(ghost_mdat_t), cr->nEnts, RESTFILE);
			} 
			else 
			{
				switch (datatype) {
					case GHOST_DATATYPE_S:
						{
							float *tmp = (float *)allocateMemory(
									cr->nEnts*sizeof(float), "tmp");
							fread(tmp, sizeof(float), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (ghost_mdat_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_D:
						{
							double *tmp = (double *)allocateMemory(
									cr->nEnts*sizeof(double), "tmp");
							fread(tmp, sizeof(double), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (ghost_mdat_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_C:
						{
							_Complex float *tmp = (_Complex float *)allocateMemory(
									cr->nEnts*sizeof(_Complex float), "tmp");
							fread(tmp, sizeof(_Complex float), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (ghost_mdat_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_Z:
						{
							_Complex double *tmp = (_Complex double *)allocateMemory(
									cr->nEnts*sizeof(_Complex double), "tmp");
							fread(tmp, sizeof(_Complex double), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (ghost_mdat_t) tmp[i];
							free(tmp);
							break;
						}
				}
			}
		}
	}
	fclose(RESTFILE);

	DEBUG_LOG(1,"Matrix read in successfully");

	return cr;
}

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm ) 
{


	ghost_midx_t* nEntsInRow;
	ghost_midx_t i, e, pos;

	size_t size_rpt, size_col, size_val, size_nEntsInRow;


	IF_DEBUG(1) printf("Entering convertMMToCRMatrix\n");


	size_rpt  = (size_t)( (mm->nrows+1) * sizeof( ghost_midx_t ) );
	size_col        = (size_t)( mm->nEnts     * sizeof( ghost_midx_t ) );
	size_val        = (size_t)( mm->nEnts     * sizeof( ghost_mdat_t) );
	size_nEntsInRow = (size_t)(  mm->nrows    * sizeof( ghost_midx_t) );


	CR_TYPE* cr   = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
	cr->rpt = (ghost_midx_t*)     allocateMemory( size_rpt,    "rpt" );
	cr->col = (ghost_midx_t*)     allocateMemory( size_col,          "col" );
	cr->val = (ghost_mdat_t*)  allocateMemory( size_val,          "val" );
	nEntsInRow = (ghost_midx_t*)     allocateMemory( size_nEntsInRow,   "nEntsInRow" );


	cr->nrows = mm->nrows;
	cr->ncols = mm->ncols;
	cr->nEnts = mm->nEnts;
	for( i = 0; i < mm->nrows; i++ ) nEntsInRow[i] = 0;


	//qsort( mm->nze, mm->nEnts, sizeof( NZE_TYPE ), compareNZEPos );
	IF_DEBUG(1) printf("direkt vor  qsort\n"); fflush(stdout);
	qsort( mm->nze, (size_t)(mm->nEnts), sizeof( NZE_TYPE ), compareNZEPos );
	IF_DEBUG(1) printf("Nach qsort\n"); fflush(stdout);

	for( e = 0; e < mm->nEnts; e++ ) nEntsInRow[mm->nze[e].row]++;
	pos = 0;
	cr->rpt[0] = pos;
#ifdef PLACE_CRS
	// NUMA placement for rpt
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < mm->nrows; i++ ) {
		cr->rpt[i] = 0;
	}
#endif

	for( i = 0; i < mm->nrows; i++ ) {
		cr->rpt[i] = pos;
		pos += nEntsInRow[i];
	}
	cr->rpt[mm->nrows] = pos;

	for( i = 0; i < mm->nrows; i++ ) nEntsInRow[i] = 0;

#pragma omp parallel for schedule(runtime)
	for(i=0; i<cr->nrows; ++i) {
		ghost_midx_t start = cr->rpt[i];
		ghost_midx_t end = cr->rpt[i+1];
		ghost_midx_t j;
		for(j=start; j<end; j++) {
			cr->val[j] = 0.0;
			cr->col[j] = 0;
		}
	}

	for( e = 0; e < mm->nEnts; e++ ) {
		const int row = mm->nze[e].row,
			  col = mm->nze[e].col;
		const ghost_mdat_t val = mm->nze[e].val;
		pos = cr->rpt[row] + nEntsInRow[row];
		cr->col[pos] = col;

		cr->val[pos] = val;

		nEntsInRow[row]++;
	}
	free( nEntsInRow );

	IF_DEBUG(1) printf( "convertMMToCRMatrix: done\n" );


	return cr;
}

void freeMMMatrix( MM_TYPE* const mm ) 
{
	if( mm ) {
		freeMemory( (size_t)(mm->nEnts*sizeof(NZE_TYPE)), "mm->nze", mm->nze );
		freeMemory( (size_t)sizeof(MM_TYPE), "mm", mm );
	}
}
*/
int pad(int nrows, int padding) 
{
	int nrowsPadded;

	if(  nrows % padding != 0) {
		nrowsPadded = nrows + padding - nrows % padding;
	} else {
		nrowsPadded = nrows;
	}
	return nrowsPadded;
}
