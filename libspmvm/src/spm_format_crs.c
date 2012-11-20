#define _XOPEN_SOURCE 500

#include "spm_format_crs.h"
#include "matricks.h"
#include "ghost_util.h"


#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


static void CRS_fromBin(char *matrixPath, mat_trait_t trait);
static void CRS_printInfo();
static char * CRS_formatName();
static mat_idx_t CRS_rowLen (mat_idx_t i);
static mat_data_t CRS_entry (mat_idx_t i, mat_idx_t j);
static size_t CRS_byteSize (void);
static void CRS_kernel (ghost_vec_t *, ghost_vec_t *, int);

static ghost_mat_t *thisMat;
static CR_TYPE *thisCR;

void CRS_registerFunctions(ghost_mat_t *mat)
{
	thisMat = mat;
	//thisCR = (CR_TYPE *)(mat->data);

	mat->fromBin = &CRS_fromBin;
	mat->printInfo = &CRS_printInfo;
	mat->formatName = &CRS_formatName;
	mat->rowLen   = &CRS_rowLen;
	mat->entry    = &CRS_entry;
	mat->byteSize = &CRS_byteSize;
	mat->kernel   = &CRS_kernel;
}

static void CRS_printInfo()
{
	return;
}

static char * CRS_formatName()
{
	return "CRS";
}

static mat_idx_t CRS_rowLen (mat_idx_t i)
{
	return thisCR->rpt[i+1] - thisCR->rpt[i];
}
	
static mat_data_t CRS_entry (mat_idx_t i, mat_idx_t j)
{
	mat_idx_t e;
	for (e=thisCR->rpt[i]; e<thisCR->rpt[i+1]; e++) {
		if (thisCR->col[e] == j)
			return thisCR->val[e];
	}
	return 0.;
}

static size_t CRS_byteSize (void)
{
	return (size_t)((thisCR->nrows+1)*sizeof(mat_nnz_t) + 
			thisCR->nEnts*(sizeof(mat_idx_t)+sizeof(mat_data_t)));
}

static void CRS_fromBin(char *matrixPath, mat_trait_t trait)
{

	int rowPtrOnly = 0;
	int detectDiags = 0;
	mat_idx_t i, j;
	int datatype;
	FILE* RESTFILE;

	thisCR = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "thisCR" );
	thisMat->data = thisCR;
	DEBUG_LOG(1,"Reading binary CRS matrix %s",matrixPath);

	if ((RESTFILE = fopen(matrixPath, "rb"))==NULL){
		ABORT("Could not open binary CRS file %s",matrixPath);
	}

	fread(&datatype, sizeof(int), 1, RESTFILE);
	fread(&thisCR->nrows, sizeof(int), 1, RESTFILE);
	fread(&thisCR->ncols, sizeof(int), 1, RESTFILE);
	fread(&thisCR->nEnts, sizeof(int), 1, RESTFILE);

	DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",thisCR->nrows,thisCR->ncols,thisCR->nEnts);

	if (datatype != DATATYPE_DESIRED) {
		DEBUG_LOG(0,"Warning in %s:%d! The library has been built for %s data but"
				" the file contains %s data. Casting...\n",__FILE__,__LINE__,
				DATATYPE_NAMES[DATATYPE_DESIRED],DATATYPE_NAMES[datatype]);
	}

	DEBUG_LOG(2,"Allocate memory for thisCR->rpt");
	thisCR->rpt = (mat_idx_t *)    allocateMemory( (thisCR->nrows+1)*sizeof(mat_idx_t), "rpt" );

	DEBUG_LOG(1,"NUMA-placement for thisCR->rpt");
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < thisCR->nrows+1; i++ ) {
		thisCR->rpt[i] = 0;
	}

	DEBUG_LOG(2,"Reading array with row-offsets");
	fread(&thisCR->rpt[0],        sizeof(int),    thisCR->nrows+1, RESTFILE);


	if (!rowPtrOnly) {
		thisCR->constDiags = NULL;

		if (detectDiags) {
			mat_idx_t bandwidth = 2;//thisCR->ncols/2;
			mat_idx_t nDiags = 2*bandwidth + 1;

			mat_data_t *diagVals = (mat_data_t *)allocateMemory(nDiags*sizeof(mat_data_t),"diagVals");

			char *diagStatus = (char *)allocateMemory(nDiags*sizeof(char),"diagStatus");
			for (i=0; i<nDiags; i++) diagStatus[i] = DIAG_NEW;

			int *diagEnts = (int *)allocateMemory(nDiags*sizeof(int),"diagEnts");
			for (i=0; i<nDiags; i++) diagEnts[i] = 0;

			DEBUG_LOG(1,"Detecting constant subdiagonals within a band of width %"PRmatIDX,bandwidth);
			mat_idx_t *tmpcol = (mat_idx_t *)allocateMemory(thisCR->nEnts*sizeof(mat_idx_t),"tmpcol");
			mat_data_t *tmpval = (mat_data_t *)allocateMemory(thisCR->nEnts*sizeof(mat_data_t),"tmpval");

			int pfile;
			pfile = open(matrixPath,O_RDONLY);
			int offs = 4*sizeof(int)+(thisCR->nrows+1)*sizeof(int);
			int idx = 0;
			for (i=0; i<thisCR->nrows; ++i) {
				for(j = thisCR->rpt[i] ; j < thisCR->rpt[i+1] ; j++) {
					pread(pfile,&tmpcol[idx],sizeof(int),offs+idx*sizeof(int));
					pread(pfile,&tmpval[idx],sizeof(mat_data_t),offs+thisCR->nEnts*sizeof(int)+idx*sizeof(mat_data_t));
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

			thisCR->nConstDiags = 0;

			for (i=0; i<bandwidth+1; i++) { // lower subdiagonals AND diagonal
				if (diagStatus[i] == DIAG_OK && diagEnts[i] == thisCR->ncols-bandwidth+i) {
					DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f",bandwidth-i,diagVals[i]);
					thisCR->nConstDiags++;
					thisCR->constDiags = realloc(thisCR->constDiags,sizeof(CONST_DIAG)*thisCR->nConstDiags);
					thisCR->constDiags[thisCR->nConstDiags-1].idx = bandwidth-i;
					thisCR->constDiags[thisCR->nConstDiags-1].val = diagVals[i];
					thisCR->constDiags[thisCR->nConstDiags-1].len = diagEnts[i];
					thisCR->constDiags[thisCR->nConstDiags-1].minRow = i-bandwidth;
					thisCR->constDiags[thisCR->nConstDiags-1].maxRow = thisCR->nrows-1;
					DEBUG_LOG(1,"range: %"PRmatIDX"..%"PRmatIDX,i-bandwidth,thisCR->nrows-1);
				}
			}
			for (i=bandwidth+1; i<nDiags ; i++) { // upper subdiagonals
				if (diagStatus[i] == DIAG_OK && diagEnts[i] == thisCR->ncols+bandwidth-i) {
					DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f",-bandwidth+i,diagVals[i]);
					thisCR->nConstDiags++;
					thisCR->constDiags = realloc(thisCR->constDiags,sizeof(CONST_DIAG)*thisCR->nConstDiags);
					thisCR->constDiags[thisCR->nConstDiags-1].idx = -bandwidth+i;
					thisCR->constDiags[thisCR->nConstDiags-1].val = diagVals[i];
					thisCR->constDiags[thisCR->nConstDiags-1].len = diagEnts[i];
					thisCR->constDiags[thisCR->nConstDiags-1].minRow = 0;
					thisCR->constDiags[thisCR->nConstDiags-1].maxRow = thisCR->nrows-1-i+bandwidth;
					DEBUG_LOG(1,"range: 0..%"PRmatIDX,thisCR->nrows-1-i+bandwidth);
				}
			}

			if (thisCR->nConstDiags == 0) 
			{ // no constant diagonals found
				thisCR->val = tmpval;
				thisCR->col = tmpcol;
			} 
			else {
				mat_idx_t d = 0;

				DEBUG_LOG(1,"Adjusting the number of matrix entries, old: %"PRmatNNZ,thisCR->nEnts);
				for (d=0; d<thisCR->nConstDiags; d++) {
					thisCR->nEnts -= thisCR->constDiags[d].len;
				}
				DEBUG_LOG(1,"Adjusting the number of matrix entries, new: %"PRmatNNZ,thisCR->nEnts);

				DEBUG_LOG(2,"Allocate memory for thisCR->col and thisCR->val");
				thisCR->col       = (mat_idx_t*)    allocateMemory( thisCR->nEnts * sizeof(mat_idx_t),  "col" );
				thisCR->val       = (mat_data_t*) allocateMemory( thisCR->nEnts * sizeof(mat_data_t),  "val" );

				//TODO NUMA
				mat_idx_t *newRowOffset = (mat_idx_t *)allocateMemory((thisCR->nrows+1)*sizeof(mat_idx_t),"newRowOffset");

				idx = 0;
				mat_idx_t oidx = 0; // original idx in tmp arrays
				for (i=0; i<thisCR->nrows; ++i) {
					newRowOffset[i] = idx;
					for(j = thisCR->rpt[i] ; j < thisCR->rpt[i+1]; j++) {
						if (ABS(tmpcol[oidx]-i) <= bandwidth) { // in band
							int diagFound = 0;
							for (d=0; d<thisCR->nConstDiags; d++) {
								if (tmpcol[oidx]-i == thisCR->constDiags[d].idx) { // do not store constant diagonal
									oidx++;
									diagFound = 1;
									break;
								} 
							}
							if (!diagFound) {
								thisCR->col[idx] = tmpcol[oidx];
								thisCR->val[idx] = tmpval[oidx];
								idx++;
								oidx++;
							}

						} else {
							thisCR->col[idx] = tmpcol[oidx];
							thisCR->val[idx] = tmpval[oidx];
							idx++;
							oidx++;

						}
					}
				}
				free(thisCR->rpt);
				free(tmpval);
				free(tmpcol);

				newRowOffset[thisCR->nrows] = thisCR->nEnts;
				thisCR->rpt = newRowOffset;
			}
			close(pfile);
		} 
		else {

			DEBUG_LOG(2,"Allocate memory for thisCR->col and thisCR->val");
			thisCR->col       = (mat_idx_t *)    allocateMemory( thisCR->nEnts * sizeof(mat_idx_t),  "col" );
			thisCR->val       = (mat_data_t *) allocateMemory( thisCR->nEnts * sizeof(mat_data_t),  "val" );

			DEBUG_LOG(1,"NUMA-placement for thisCR->val and thisCR->col");
#pragma omp parallel for schedule(runtime)
			for(i = 0 ; i < thisCR->nrows; ++i) {
				for(j = thisCR->rpt[i] ; j < thisCR->rpt[i+1] ; j++) {
					thisCR->val[j] = 0.0;
					thisCR->col[j] = 0;
				}
			}


			DEBUG_LOG(2,"Reading array with column indices");

			// TODO fread => pread
			fread(&thisCR->col[0], sizeof(int), thisCR->nEnts, RESTFILE);
			DEBUG_LOG(2,"Reading array with values");
			if (datatype == DATATYPE_DESIRED)
			{
				fread(&thisCR->val[0], sizeof(mat_data_t), thisCR->nEnts, RESTFILE);
			} 
			else 
			{
				switch (datatype) {
					case GHOST_DATATYPE_S:
						{
							float *tmp = (float *)allocateMemory(
									thisCR->nEnts*sizeof(float), "tmp");
							fread(tmp, sizeof(float), thisCR->nEnts, RESTFILE);
							for (i = 0; i<thisCR->nEnts; i++) thisCR->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_D:
						{
							double *tmp = (double *)allocateMemory(
									thisCR->nEnts*sizeof(double), "tmp");
							fread(tmp, sizeof(double), thisCR->nEnts, RESTFILE);
							for (i = 0; i<thisCR->nEnts; i++) thisCR->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_C:
						{
							_Complex float *tmp = (_Complex float *)allocateMemory(
									thisCR->nEnts*sizeof(_Complex float), "tmp");
							fread(tmp, sizeof(_Complex float), thisCR->nEnts, RESTFILE);
							for (i = 0; i<thisCR->nEnts; i++) thisCR->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_Z:
						{
							_Complex double *tmp = (_Complex double *)allocateMemory(
									thisCR->nEnts*sizeof(_Complex double), "tmp");
							fread(tmp, sizeof(_Complex double), thisCR->nEnts, RESTFILE);
							for (i = 0; i<thisCR->nEnts; i++) thisCR->val[i] = (mat_data_t) tmp[i];
							free(tmp);
							break;
						}
				}
			}
		}
	}
	fclose(RESTFILE);
	
	thisMat->trait = trait;
	thisMat->nrows = thisCR->nrows;
	thisMat->ncols = thisCR->ncols;
	thisMat->nnz = thisCR->nEnts;

	DEBUG_LOG(1,"Matrix read in successfully");

}

static void CRS_kernel (ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	mat_idx_t i, j;
	mat_data_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<thisCR->nrows; i++){
		hlp1 = 0.0;
		for (j=thisCR->rpt[i]; j<thisCR->rpt[i+1]; j++){
			hlp1 = hlp1 + thisCR->val[j] * rhs->val[thisCR->col[j]]; 
		}
		if (options & GHOST_OPTION_AXPY) 
			lhs->val[i] += hlp1;
		else
			lhs->val[i] = hlp1;
	}
}
