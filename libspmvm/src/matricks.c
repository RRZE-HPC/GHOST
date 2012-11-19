#define _XOPEN_SOURCE 600
#include "matricks.h"
#include "spmvm.h"
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


MM_TYPE * readMMFile(const char* filename ) 
{

	MM_typecode matcode;
	FILE *f;
	mat_idx_t i;
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



	if ((mm_read_mtx_crd_size(f, &mm->nrows, &mm->ncols, &mm->nEnts)) !=0)
		exit(1);


	mm->nze = (NZE_TYPE *)malloc(mm->nEnts*sizeof(NZE_TYPE));

	if (!mm_is_complex(matcode)) {
		for (i=0; i<mm->nEnts; i++)
		{
#ifdef DOUBLE
			double re;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %lg\n", &mm->nze[i].row, &mm->nze[i].col, &re);
#else
			float re;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %g\n", &mm->nze[i].row, &mm->nze[i].col, &re);
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
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %lg %lg\n", &mm->nze[i].row, &mm->nze[i].col, &re,
					&im);
#else
			float re,im;
			fscanf(f, "%"PRmatIDX" %"PRmatIDX" %g %g\n", &mm->nze[i].row, &mm->nze[i].col, &re,
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
	mat_idx_t i, j;
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

	if (datatype != DATATYPE_DESIRED) {
		DEBUG_LOG(0,"Warning in %s:%d! The library has been built for %s data but"
				" the file contains %s data. Casting...\n",__FILE__,__LINE__,
				DATATYPE_NAMES[DATATYPE_DESIRED],DATATYPE_NAMES[datatype]);
	}

	DEBUG_LOG(2,"Allocate memory for cr->rpt");
	cr->rpt = (mat_idx_t *)    allocateMemory( (cr->nrows+1)*sizeof(mat_idx_t), "rpt" );

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
			mat_idx_t bandwidth = 2;//cr->ncols/2;
			mat_idx_t nDiags = 2*bandwidth + 1;

			mat_data_t *diagVals = (mat_data_t *)allocateMemory(nDiags*sizeof(mat_data_t),"diagVals");

			char *diagStatus = (char *)allocateMemory(nDiags*sizeof(char),"diagStatus");
			for (i=0; i<nDiags; i++) diagStatus[i] = DIAG_NEW;

			int *diagEnts = (int *)allocateMemory(nDiags*sizeof(int),"diagEnts");
			for (i=0; i<nDiags; i++) diagEnts[i] = 0;

			DEBUG_LOG(1,"Detecting constant subdiagonals within a band of width %"PRmatIDX,bandwidth);
			mat_idx_t *tmpcol = (mat_idx_t *)allocateMemory(cr->nEnts*sizeof(mat_idx_t),"tmpcol");
			mat_data_t *tmpval = (mat_data_t *)allocateMemory(cr->nEnts*sizeof(mat_data_t),"tmpval");

			int pfile;
			pfile = open(path,O_RDONLY);
			int offs = 4*sizeof(int)+(cr->nrows+1)*sizeof(int);
			int idx = 0;
			for (i=0; i<cr->nrows; ++i) {
				for(j = cr->rpt[i] ; j < cr->rpt[i+1] ; j++) {
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
								DEBUG_LOG(2,"Diag %d discontinued in row %"PRmatIDX": %f (was %f)",didx,i,tmpval[idx],diagVals[didx]);
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
					DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f",bandwidth-i,diagVals[i]);
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
					DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f",-bandwidth+i,diagVals[i]);
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
				mat_idx_t d = 0;

				DEBUG_LOG(1,"Adjusting the number of matrix entries, old: %"PRmatNNZ,cr->nEnts);
				for (d=0; d<cr->nConstDiags; d++) {
					cr->nEnts -= cr->constDiags[d].len;
				}
				DEBUG_LOG(1,"Adjusting the number of matrix entries, new: %"PRmatNNZ,cr->nEnts);

				DEBUG_LOG(2,"Allocate memory for cr->col and cr->val");
				cr->col       = (mat_idx_t*)    allocateMemory( cr->nEnts * sizeof(mat_idx_t),  "col" );
				cr->val       = (mat_data_t*) allocateMemory( cr->nEnts * sizeof(mat_data_t),  "val" );

				//TODO NUMA
				mat_idx_t *newRowOffset = (mat_idx_t *)allocateMemory((cr->nrows+1)*sizeof(mat_idx_t),"newRowOffset");

				idx = 0;
				mat_idx_t oidx = 0; // original idx in tmp arrays
				for (i=0; i<cr->nrows; ++i) {
					newRowOffset[i] = idx;
					for(j = cr->rpt[i] ; j < cr->rpt[i+1]; j++) {
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
			cr->col       = (mat_idx_t *)    allocateMemory( cr->nEnts * sizeof(mat_idx_t),  "col" );
			cr->val       = (mat_data_t *) allocateMemory( cr->nEnts * sizeof(mat_data_t),  "val" );

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
			if (datatype == DATATYPE_DESIRED)
			{
				fread(&cr->val[0], sizeof(mat_data_t), cr->nEnts, RESTFILE);
			} 
			else 
			{
				switch (datatype) {
					case GHOST_DATATYPE_S:
						{
							float *tmp = (float *)allocateMemory(
									cr->nEnts*sizeof(float), "tmp");
							fread(tmp, sizeof(float), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_D:
						{
							double *tmp = (double *)allocateMemory(
									cr->nEnts*sizeof(double), "tmp");
							fread(tmp, sizeof(double), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_C:
						{
							_Complex float *tmp = (_Complex float *)allocateMemory(
									cr->nEnts*sizeof(_Complex float), "tmp");
							fread(tmp, sizeof(_Complex float), cr->nEnts, RESTFILE);
							for (i = 0; i<cr->nEnts; i++) cr->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_Z:
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

	DEBUG_LOG(1,"Matrix read in successfully");

	return cr;
}

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm ) 
{

	/* allocate and fill CRS-format matrix from MM-type;
	 * row and col indices have same base as MM (0-based);
	 * elements in row are sorted according to column*/

	mat_idx_t* nEntsInRow;
	mat_idx_t i, e, pos;

	size_t size_rpt, size_col, size_val, size_nEntsInRow;


	/* allocate memory ######################################################## */
	IF_DEBUG(1) printf("Entering convertMMToCRMatrix\n");


	size_rpt  = (size_t)( (mm->nrows+1) * sizeof( mat_idx_t ) );
	size_col        = (size_t)( mm->nEnts     * sizeof( mat_idx_t ) );
	size_val        = (size_t)( mm->nEnts     * sizeof( mat_data_t) );
	size_nEntsInRow = (size_t)(  mm->nrows    * sizeof( mat_idx_t) );


	CR_TYPE* cr   = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "cr" );
	cr->rpt = (mat_idx_t*)     allocateMemory( size_rpt,    "rpt" );
	cr->col = (mat_idx_t*)     allocateMemory( size_col,          "col" );
	cr->val = (mat_data_t*)  allocateMemory( size_val,          "val" );
	nEntsInRow = (mat_idx_t*)     allocateMemory( size_nEntsInRow,   "nEntsInRow" );


	/* initialize values ###################################################### */
	cr->nrows = mm->nrows;
	cr->ncols = mm->ncols;
	cr->nEnts = mm->nEnts;
	for( i = 0; i < mm->nrows; i++ ) nEntsInRow[i] = 0;


	/* sort NZEs with ascending column index for each row ##################### */
	//qsort( mm->nze, mm->nEnts, sizeof( NZE_TYPE ), compareNZEPos );
	IF_DEBUG(1) printf("direkt vor  qsort\n"); fflush(stdout);
	qsort( mm->nze, (size_t)(mm->nEnts), sizeof( NZE_TYPE ), compareNZEPos );
	IF_DEBUG(1) printf("Nach qsort\n"); fflush(stdout);

	/* count entries per row ################################################## */
	for( e = 0; e < mm->nEnts; e++ ) nEntsInRow[mm->nze[e].row]++;

	/* set offsets for each row ############################################### */
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
		mat_idx_t start = cr->rpt[i];
		mat_idx_t end = cr->rpt[i+1];
		mat_idx_t j;
		for(j=start; j<end; j++) {
			cr->val[j] = 0.0;
			cr->col[j] = 0;
		}
	}

	/* store values in compressed row data structure ########################## */
	for( e = 0; e < mm->nEnts; e++ ) {
		const int row = mm->nze[e].row,
			  col = mm->nze[e].col;
		const mat_data_t val = mm->nze[e].val;
		pos = cr->rpt[row] + nEntsInRow[row];
		/* GW 
		   cr->col[pos] = col;
		 */
		cr->col[pos] = col;

		cr->val[pos] = val;

		nEntsInRow[row]++;
	}
	/* clean up ############################################################### */
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

void CRStoTBJDS(CR_TYPE *cr, mat_trait_t trait, ghost_mat_t **matrix) 
{
	mat_idx_t i,j,c;
	BJDS_TYPE *tbjds;
	JD_SORT_TYPE* rowSort;
	mat_idx_t *rowPerm = NULL, *invRowPerm = NULL;
	unsigned int flags;

	*matrix = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	tbjds = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"mv");

	tbjds->nrows = cr->nrows;
	tbjds->nnz = cr->nEnts;
	tbjds->nEnts = 0;
	tbjds->nrowsPadded = pad(tbjds->nrows,BJDS_LEN);
	mat_idx_t nChunks = tbjds->nrowsPadded/BJDS_LEN;
		tbjds->chunkStart = (mat_nnz_t *)allocateMemory((nChunks+1)*sizeof(mat_nnz_t),"tbjds->chunkStart");
		tbjds->chunkMin = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"tbjds->chunkMin");
		tbjds->chunkLen = (mat_idx_t *)allocateMemory((nChunks)*sizeof(mat_idx_t),"tbjds->chunkMin");
		tbjds->rowLen = (mat_idx_t *)allocateMemory((tbjds->nrowsPadded)*sizeof(mat_idx_t),"tbjds->chunkMin");
		tbjds->chunkStart[0] = 0;
	flags = trait.flags;
	
	**matrix = (ghost_mat_t)MATRIX_INIT(
			.trait = trait, 
			.nrows = cr->nrows, 
			.ncols = cr->ncols, 
			.nnz = tbjds->nnz,
			.rowPerm = rowPerm,
			.invRowPerm = invRowPerm,	   
			.data = tbjds);
	TBJDS_init(*matrix);

	

	if (trait.flags & GHOST_SPM_SORTED) {
		rowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"sbjds->rowPerm");
		invRowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"sbjds->invRowPerm");

		(*matrix)->rowPerm = rowPerm;
		(*matrix)->invRowPerm = invRowPerm;

		unsigned int sortBlock = *(unsigned int *)(trait.aux);

		/* get max number of entries in one row ###########################*/
		rowSort = (JD_SORT_TYPE*) allocateMemory( cr->nrows * sizeof( JD_SORT_TYPE ),
				"rowSort" );

		for (c=0; c<cr->nrows/sortBlock; c++)  
		{
			for( i = c*sortBlock; i < (c+1)*sortBlock; i++ ) 
			{
				rowSort[i].row = i;
				rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
			} 

			qsort( rowSort+c*sortBlock, sortBlock, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );
		}
		for( i = c*sortBlock; i < cr->nrows; i++ ) 
		{ // remainder
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
		}


		/* sort within same rowlength with asceding row number #################### */
		/*	i=0;
			while(i < cr->nrows) {
			int start = i;

			j = rowSort[start].nEntsInRow;
			while( i<cr->nrows && rowSort[i].nEntsInRow >= j ) 
			++i;

			DEBUG_LOG(1,"sorting over %i rows (%i): %i - %i\n",i-start,j, start, i-1);
			qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
			}

			for(i=1; i < cr->nrows; ++i) {
			if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
			printf("Error in row %i: descending row number\n",i);
			}*/
		for(i=0; i < cr->nrows; ++i) {
			/* invRowPerm maps an index in the permuted system to the original index,
			 * rowPerm gets the original index and returns the corresponding permuted position.
			 */
			if( rowSort[i].row >= cr->nrows ) DEBUG_LOG(0,"error: invalid row number %"PRmatIDX" in %"PRmatIDX,rowSort[i].row, i); 

			(invRowPerm)[i] = rowSort[i].row;
			(rowPerm)[rowSort[i].row] = i;
		}
	}

	//	for(i=0; i < cr->nrows; ++i) printf("%d\n",(*invRowPerm)[i]);

	mat_idx_t chunkMin = cr->ncols;
	mat_idx_t chunkLen = 0;
	mat_idx_t curChunk = 1;
	tbjds->nu = 0.;

	for (i=0; i<tbjds->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				tbjds->rowLen[i] = rowSort[i].nEntsInRow;
			else
				tbjds->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			tbjds->rowLen[i] = 0;
		}

		tbjds->nEnts += tbjds->rowLen[i];

		chunkMin = tbjds->rowLen[i]<chunkMin?tbjds->rowLen[i]:chunkMin;
		chunkLen = tbjds->rowLen[i]>chunkLen?tbjds->rowLen[i]:chunkLen;

		if ((i+1)%BJDS_LEN == 0) {
			tbjds->chunkStart[curChunk] = tbjds->nEnts;
			tbjds->chunkMin[curChunk-1] = chunkMin;
			tbjds->chunkLen[curChunk-1] = chunkLen;

			tbjds->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			curChunk++;
		}
	}
	tbjds->nu /= (double)nChunks;

	tbjds->val = (mat_data_t *)allocateMemory(sizeof(mat_data_t)*tbjds->nEnts,"tbjds->val");
	tbjds->col = (mat_idx_t *)allocateMemory(sizeof(mat_idx_t)*tbjds->nEnts,"tbjds->col");

	//printf("nEnts: %d\n",tbjds->nEnts);

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<tbjds->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<tbjds->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				tbjds->val[tbjds->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				tbjds->col[tbjds->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
		mat_nnz_t rem = tbjds->chunkStart[c] + tbjds->chunkMin[c]*BJDS_LEN;
		for (i=0; i<BJDS_LEN; i++)
		{
			for (j=tbjds->chunkMin[c]; j<tbjds->rowLen[c*BJDS_LEN+i]; j++)
			{
				tbjds->val[rem] = 0.;
				tbjds->col[rem++] = 0;
			}
		}
	}
	for (c=0; c<tbjds->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		// store block
		for (j=0; j<tbjds->chunkMin[c]; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				if (flags & GHOST_SPM_SORTED) {
					tbjds->val[tbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
					if (flags & GHOST_SPM_PERMUTECOLIDX)
						tbjds->col[tbjds->chunkStart[c]+j*BJDS_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
					else
						tbjds->col[tbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
				} else {
					tbjds->val[tbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
					tbjds->col[tbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
				}
			}
		}

		// store remainder
		mat_nnz_t rem = tbjds->chunkStart[c] + tbjds->chunkMin[c]*BJDS_LEN;
		if (flags & GHOST_SPM_COLMAJOR) 
		{
			for (j=tbjds->chunkMin[c]; j<tbjds->chunkLen[c]; j++)
			{
				for (i=0; i<BJDS_LEN; i++)
				{
					if (j<tbjds->rowLen[c*BJDS_LEN+i] ) {
						if (flags & GHOST_SPM_SORTED) {
							tbjds->val[rem] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
							if (flags & GHOST_SPM_PERMUTECOLIDX)
								tbjds->col[rem++] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
							else
								tbjds->col[rem++] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
						} else {
							tbjds->val[rem] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
							tbjds->col[rem++] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
						}
					}
				}
			}
		} else // row major is the default 
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				for (j=tbjds->chunkMin[c]; j<tbjds->rowLen[c*BJDS_LEN+i]; j++)
				{
					if (flags & GHOST_SPM_SORTED) {
						tbjds->val[rem] = cr->val[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							tbjds->col[rem++] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j]];
						else
							tbjds->col[rem++] = cr->col[cr->rpt[(invRowPerm)[c*BJDS_LEN+i]]+j];
					} else {
						tbjds->val[rem] = cr->val[cr->rpt[c*BJDS_LEN+i]+j];
						tbjds->col[rem++] = cr->col[cr->rpt[c*BJDS_LEN+i]+j];
					}
				}
			}
		}
	}
}

void CRStoBJDS(CR_TYPE *cr, mat_trait_t trait, ghost_mat_t **matrix)
{
	DEBUG_LOG(1,"Creating BJDS matrix");
	mat_idx_t i,j,c;
	*matrix = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	BJDS_TYPE *sbjds;
	unsigned int flags = trait.flags;

	mat_idx_t *rowPerm = NULL;
	mat_idx_t *invRowPerm = NULL;

	JD_SORT_TYPE* rowSort;

	sbjds = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"sbjds");
	**matrix = (ghost_mat_t)MATRIX_INIT(
			.trait = trait, 
			.nrows = cr->nrows, 
			.ncols = cr->ncols, 
			.nnz = cr->nEnts,
			.rowPerm = rowPerm,
			.invRowPerm = invRowPerm,	   
			.data = sbjds);
	BJDS_init(*matrix);

	if (trait.flags & GHOST_SPM_SORTED) {
		rowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"sbjds->rowPerm");
		invRowPerm = (mat_idx_t *)allocateMemory(cr->nrows*sizeof(mat_idx_t),"sbjds->invRowPerm");
		
		(*matrix)->rowPerm = rowPerm;
		(*matrix)->invRowPerm = invRowPerm;

		/* get max number of entries in one row ###########################*/
		rowSort = (JD_SORT_TYPE*) allocateMemory( cr->nrows * sizeof( JD_SORT_TYPE ),
				"rowSort" );

		for( i = 0; i < cr->nrows; i++ ) {
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
		} 

		qsort( rowSort, cr->nrows, sizeof( JD_SORT_TYPE  ), compareNZEPerRow );

		/* sort within same rowlength with asceding row number #################### */
		i=0;
		while(i < cr->nrows) {
			mat_idx_t start = i;

			j = rowSort[start].nEntsInRow;
			while( i<cr->nrows && rowSort[i].nEntsInRow >= j ) 
				++i;

			DEBUG_LOG(1,"sorting over %"PRmatIDX" rows (%"PRmatIDX"): %"PRmatIDX" - %"PRmatIDX,i-start,j, start, i-1);
			qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
		}

		for(i=1; i < cr->nrows; ++i) {
			if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
				printf("Error in row %"PRmatIDX": descending row number\n",i);
		}
		for(i=0; i < cr->nrows; ++i) {
			/* invRowPerm maps an index in the permuted system to the original index,
			 * rowPerm gets the original index and returns the corresponding permuted position.
			 */
			if( rowSort[i].row >= cr->nrows ) DEBUG_LOG(0,"error: invalid row number %"PRmatIDX" in %"PRmatIDX,rowSort[i].row, i); 

			(invRowPerm)[i] = rowSort[i].row;
			(rowPerm)[rowSort[i].row] = i;
		}
	}




	sbjds->nrows = cr->nrows;
	sbjds->nnz = cr->nEnts;
	sbjds->nEnts = 0;
	sbjds->nrowsPadded = pad(sbjds->nrows,BJDS_LEN);

	mat_idx_t nChunks = sbjds->nrowsPadded/BJDS_LEN;
	sbjds->chunkStart = (mat_nnz_t *)allocateMemory((nChunks+1)*sizeof(mat_nnz_t),"sbjds->chunkStart");
	sbjds->rowLen = (mat_idx_t *)allocateMemory(sbjds->nrowsPadded*sizeof(mat_idx_t),"sbjds->rowLen");
	sbjds->chunkStart[0] = 0;

	mat_idx_t chunkMax = 0;
	mat_idx_t curChunk = 1;

	for (i=0; i<sbjds->nrows; i++) {

		if (flags & GHOST_SPM_SORTED)
			sbjds->rowLen[i] = rowSort[i].nEntsInRow;
		else
			sbjds->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		chunkMax = sbjds->rowLen[i]>chunkMax?sbjds->rowLen[i]:chunkMax;
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
	sbjds->col = (mat_idx_t *)allocateMemory(sizeof(mat_idx_t)*sbjds->nEnts,"sbjds->val");

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<sbjds->nrowsPadded/BJDS_LEN; c++) 
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
		mat_idx_t chunkLen = (sbjds->chunkStart[c+1]-sbjds->chunkStart[c])/BJDS_LEN;

		for (j=0; j<chunkLen; j++) {

			for (i=0; i<BJDS_LEN; i++) {
				mat_idx_t row = c*BJDS_LEN+i;

				if (j<sbjds->rowLen[row]) {
					if (flags & GHOST_SPM_SORTED) {
						sbjds->val[sbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							sbjds->col[sbjds->chunkStart[c]+j*BJDS_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							sbjds->col[sbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						sbjds->val[sbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[row]+j];
						sbjds->col[sbjds->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					sbjds->val[sbjds->chunkStart[c]+j*BJDS_LEN+i] = 0.0;
					sbjds->col[sbjds->chunkStart[c]+j*BJDS_LEN+i] = 0;
				}
				//	printf("%f ",sbjds->val[sbjds->chunkStart[c]+j*BJDS_LEN+i]);


			}
		}
	}
}

void CRStoCRS(CR_TYPE *cr, mat_trait_t trait, ghost_mat_t **matrix)
{
	DEBUG_LOG(1,"Creating CRS matrix from original CRS type");
	*matrix = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	**matrix = (ghost_mat_t)MATRIX_INIT(.trait = trait, .nrows = cr->nrows, .ncols = cr->ncols, .nnz = cr->nEnts, .data = cr);
	CRS_init(*matrix);


}

ghost_mat_t * SpMVM_createMatrixFromCRS(CR_TYPE *cr, mat_trait_t trait)
{
	ghost_mat_t *matrix;
	switch (trait.format) {
		case GHOST_SPMFORMAT_BJDS:
			CRStoBJDS(cr,trait,&matrix);
			break;
			//case GHOST_SPMFORMAT_SBJDS:
			//	CRStoSBJDS(cr,trait,&matrix);
			//	break;
		case GHOST_SPMFORMAT_TBJDS:
			CRStoTBJDS(cr,trait,&matrix);
			break;
			//case GHOST_SPMFORMAT_STBJDS:
			//	CRStoSTBJDS(cr,trait,&matrix);
			//	break;
		default:
			DEBUG_LOG(0,"Warning!Invalid format for global matrix! Falling back to CRS");
		case GHOST_SPMFORMAT_CRS:
		case GHOST_SPMFORMAT_CRSCD:
			CRStoCRS(cr,trait,&matrix);
			break;
	}
	return matrix;
}

void SpMVM_freeCRS( CR_TYPE* const cr ) 
{

	if (cr) {
		if (cr->rpt)
			free(cr->rpt);
		if (cr->col)
			free(cr->col);
		if (cr->val)
			free(cr->val);
		free(cr);
	}
	//TODO
	/*	size_t size_rpt, size_col, size_val;

		if( cr ) {

		size_rpt  = (size_t)( (cr->nrows+1) * sizeof( int ) );
		size_col        = (size_t)( cr->nEnts     * sizeof( int ) );
		size_val        = (size_t)( cr->nEnts     * sizeof( mat_data_t) );

		freeMemory( size_rpt,  "cr->rpt", cr->rpt );
		freeMemory( size_col,        "cr->col",       cr->col );
		freeMemory( size_val,        "cr->val",       cr->val );
		freeMemory( sizeof(CR_TYPE), "cr",            cr );

		}*/
}

