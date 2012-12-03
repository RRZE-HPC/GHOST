#define _XOPEN_SOURCE 500

#include "spm_format_crs.h"
#include "matricks.h"
#include "ghost_util.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define CR(mat) ((CR_TYPE *)(mat->data))

const char name[] = "CRS plugin for ghost";
const char version[] = "0.1a";
const char formatID[] = "CRS";

static mat_nnz_t CRS_nnz(ghost_mat_t *mat);
static mat_idx_t CRS_nrows(ghost_mat_t *mat);
static mat_idx_t CRS_ncols(ghost_mat_t *mat);
static void CRS_fromBin(ghost_mat_t *mat, char *matrixPath);
static void CRS_fromMM(ghost_mat_t *mat, char *matrixPath);
static void CRS_printInfo(ghost_mat_t *mat);
static char * CRS_formatName(ghost_mat_t *mat);
static mat_idx_t CRS_rowLen (ghost_mat_t *mat, mat_idx_t i);
static ghost_mdat_t CRS_entry (ghost_mat_t *mat, mat_idx_t i, mat_idx_t j);
static size_t CRS_byteSize (ghost_mat_t *mat);
static void CRS_free(ghost_mat_t * mat);
static void CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#ifdef OPENCL
static void CRS_kernel_CL (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif


ghost_mat_t *init(ghost_mtraits_t *traits)
{
	ghost_mat_t *mat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	mat->traits = traits;

	DEBUG_LOG(1,"Initializing CRS functions");

	mat->fromBin = &CRS_fromBin;
	mat->fromMM = &CRS_fromMM;
	mat->printInfo = &CRS_printInfo;
	mat->formatName = &CRS_formatName;
	mat->rowLen   = &CRS_rowLen;
	mat->entry    = &CRS_entry;
	mat->byteSize = &CRS_byteSize;
	mat->nnz      = &CRS_nnz;
	mat->nrows    = &CRS_nrows;
	mat->ncols    = &CRS_ncols;
	mat->destroy  = &CRS_free;
#ifdef OPENCL
	if (traits->flags & GHOST_SPM_HOST)
		mat->kernel   = &CRS_kernel_plain;
	else 
		mat->kernel   = &CRS_kernel_CL;
#else
	mat->kernel   = &CRS_kernel_plain;
#endif
	return mat;

}

static mat_nnz_t CRS_nnz(ghost_mat_t *mat)
{
	return CR(mat)->nEnts;
}
static mat_idx_t CRS_nrows(ghost_mat_t *mat)
{
	return CR(mat)->nrows;
}
static mat_idx_t CRS_ncols(ghost_mat_t *mat)
{
	return CR(mat)->ncols;
}

static void CRS_printInfo(ghost_mat_t *mat)
{
	UNUSED(mat);
	return;
}

static char * CRS_formatName(ghost_mat_t *mat)
{
	UNUSED(mat);
	return "CRS";
}

static mat_idx_t CRS_rowLen (ghost_mat_t *mat, mat_idx_t i)
{
	return CR(mat)->rpt[i+1] - CR(mat)->rpt[i];
}

static ghost_mdat_t CRS_entry (ghost_mat_t *mat, mat_idx_t i, mat_idx_t j)
{
	mat_idx_t e;
	for (e=CR(mat)->rpt[i]; e<CR(mat)->rpt[i+1]; e++) {
		if (CR(mat)->col[e] == j)
			return CR(mat)->val[e];
	}
	return 0.;
}

static size_t CRS_byteSize (ghost_mat_t *mat)
{
	return (size_t)((CR(mat)->nrows+1)*sizeof(mat_nnz_t) + 
			CR(mat)->nEnts*(sizeof(mat_idx_t)+sizeof(ghost_mdat_t)));
}

static void CRS_fromBin(ghost_mat_t *mat, char *matrixPath)
{

	int rowPtrOnly = 0;
	int detectDiags = 0;
	mat_idx_t i, j;
	int datatype;
	FILE* RESTFILE;

	mat->data = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "CR(mat)" );
	mat->rowPerm = NULL;
	mat->invRowPerm = NULL;

	DEBUG_LOG(1,"Reading binary CRS matrix %s",matrixPath);

	if ((RESTFILE = fopen(matrixPath, "rb"))==NULL){
		ABORT("Could not open binary CRS file %s",matrixPath);
	}

	fread(&datatype, sizeof(int), 1, RESTFILE);
	fread(&CR(mat)->nrows, sizeof(int), 1, RESTFILE);
	fread(&CR(mat)->ncols, sizeof(int), 1, RESTFILE);
	fread(&CR(mat)->nEnts, sizeof(int), 1, RESTFILE);

	DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",CR(mat)->nrows,CR(mat)->ncols,CR(mat)->nEnts);

	if (datatype != DATATYPE_DESIRED) {
		DEBUG_LOG(0,"Warning in %s:%d! The library has been built for %s data but"
				" the file contains %s data. Casting...\n",__FILE__,__LINE__,
				DATATYPE_NAMES[DATATYPE_DESIRED],DATATYPE_NAMES[datatype]);
	}

	DEBUG_LOG(2,"Allocate memory for CR(mat)->rpt");
	CR(mat)->rpt = (mat_idx_t *)    allocateMemory( (CR(mat)->nrows+1)*sizeof(mat_idx_t), "rpt" );

	DEBUG_LOG(1,"NUMA-placement for CR(mat)->rpt");
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < CR(mat)->nrows+1; i++ ) {
		CR(mat)->rpt[i] = 0;
	}

	DEBUG_LOG(2,"Reading array with row-offsets");
	fread(&CR(mat)->rpt[0],        sizeof(int),    CR(mat)->nrows+1, RESTFILE);


	if (!rowPtrOnly) {
		CR(mat)->constDiags = NULL;

		if (detectDiags) {
			mat_idx_t bandwidth = 2;//CR(mat)->ncols/2;
			mat_idx_t nDiags = 2*bandwidth + 1;

			ghost_mdat_t *diagVals = (ghost_mdat_t *)allocateMemory(nDiags*sizeof(ghost_mdat_t),"diagVals");

			char *diagStatus = (char *)allocateMemory(nDiags*sizeof(char),"diagStatus");
			for (i=0; i<nDiags; i++) diagStatus[i] = DIAG_NEW;

			int *diagEnts = (int *)allocateMemory(nDiags*sizeof(int),"diagEnts");
			for (i=0; i<nDiags; i++) diagEnts[i] = 0;

			DEBUG_LOG(1,"Detecting constant subdiagonals within a band of width %"PRmatIDX,bandwidth);
			mat_idx_t *tmpcol = (mat_idx_t *)allocateMemory(CR(mat)->nEnts*sizeof(mat_idx_t),"tmpcol");
			ghost_mdat_t *tmpval = (ghost_mdat_t *)allocateMemory(CR(mat)->nEnts*sizeof(ghost_mdat_t),"tmpval");

			int pfile;
			pfile = open(matrixPath,O_RDONLY);
			int offs = 4*sizeof(int)+(CR(mat)->nrows+1)*sizeof(int);
			int idx = 0;
			for (i=0; i<CR(mat)->nrows; ++i) {
				for(j = CR(mat)->rpt[i] ; j < CR(mat)->rpt[i+1] ; j++) {
					pread(pfile,&tmpcol[idx],sizeof(int),offs+idx*sizeof(int));
					pread(pfile,&tmpval[idx],sizeof(ghost_mdat_t),offs+CR(mat)->nEnts*sizeof(int)+idx*sizeof(ghost_mdat_t));
					if (ABS(tmpcol[idx]-i) <= bandwidth) { // in band

						int didx = tmpcol[idx]-i+bandwidth; // index of diagonal
						if (diagStatus[didx] == DIAG_NEW) { // first time
							diagVals[didx] =  tmpval[idx];
							diagStatus[didx] = (char)1;
							diagEnts[didx] = 1;
							DEBUG_LOG(2,"Diag %d initialized with %f+%fi",didx,REAL(diagVals[didx]),IMAG(diagVals[didx]));
						} else if (diagStatus[didx] == DIAG_OK) { // diag initialized
							if (EQUALS(diagVals[didx],tmpval[idx])) {
								diagEnts[didx]++;
							} else {
								DEBUG_LOG(2,"Diag %d discontinued in row %"PRmatIDX": %f+%fi (was %f%fi)",didx,i,REAL(tmpval[idx]),IMAG(tmpval[idx]),REAL(diagVals[didx]),IMAG(diagVals[didx]));
								diagStatus[didx] = DIAG_INVALID;
							}
						}
					}
					idx++;
				}
			}

			CR(mat)->nConstDiags = 0;

			for (i=0; i<bandwidth+1; i++) { // lower subdiagonals AND diagonal
				if (diagStatus[i] == DIAG_OK && diagEnts[i] == CR(mat)->ncols-bandwidth+i) {
					DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f+%fi",bandwidth-i,REAL(diagVals[i]),IMAG(diagVals[i]));
					CR(mat)->nConstDiags++;
					CR(mat)->constDiags = realloc(CR(mat)->constDiags,sizeof(CONST_DIAG)*CR(mat)->nConstDiags);
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].idx = bandwidth-i;
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].val = diagVals[i];
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].len = diagEnts[i];
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].minRow = i-bandwidth;
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].maxRow = CR(mat)->nrows-1;
					DEBUG_LOG(1,"range: %"PRmatIDX"..%"PRmatIDX,i-bandwidth,CR(mat)->nrows-1);
				}
			}
			for (i=bandwidth+1; i<nDiags ; i++) { // upper subdiagonals
				if (diagStatus[i] == DIAG_OK && diagEnts[i] == CR(mat)->ncols+bandwidth-i) {
					DEBUG_LOG(1,"The %"PRmatIDX"-th subdiagonal is constant with %f+%fi",-bandwidth+i,REAL(diagVals[i]),IMAG(diagVals[i]));
					CR(mat)->nConstDiags++;
					CR(mat)->constDiags = realloc(CR(mat)->constDiags,sizeof(CONST_DIAG)*CR(mat)->nConstDiags);
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].idx = -bandwidth+i;
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].val = diagVals[i];
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].len = diagEnts[i];
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].minRow = 0;
					CR(mat)->constDiags[CR(mat)->nConstDiags-1].maxRow = CR(mat)->nrows-1-i+bandwidth;
					DEBUG_LOG(1,"range: 0..%"PRmatIDX,CR(mat)->nrows-1-i+bandwidth);
				}
			}

			if (CR(mat)->nConstDiags == 0) 
			{ // no constant diagonals found
				CR(mat)->val = tmpval;
				CR(mat)->col = tmpcol;
			} 
			else {
				mat_idx_t d = 0;

				DEBUG_LOG(1,"Adjusting the number of matrix entries, old: %"PRmatNNZ,CR(mat)->nEnts);
				for (d=0; d<CR(mat)->nConstDiags; d++) {
					CR(mat)->nEnts -= CR(mat)->constDiags[d].len;
				}
				DEBUG_LOG(1,"Adjusting the number of matrix entries, new: %"PRmatNNZ,CR(mat)->nEnts);

				DEBUG_LOG(2,"Allocate memory for CR(mat)->col and CR(mat)->val");
				CR(mat)->col       = (mat_idx_t*)    allocateMemory( CR(mat)->nEnts * sizeof(mat_idx_t),  "col" );
				CR(mat)->val       = (ghost_mdat_t*) allocateMemory( CR(mat)->nEnts * sizeof(ghost_mdat_t),  "val" );

				//TODO NUMA
				mat_idx_t *newRowOffset = (mat_idx_t *)allocateMemory((CR(mat)->nrows+1)*sizeof(mat_idx_t),"newRowOffset");

				idx = 0;
				mat_idx_t oidx = 0; // original idx in tmp arrays
				for (i=0; i<CR(mat)->nrows; ++i) {
					newRowOffset[i] = idx;
					for(j = CR(mat)->rpt[i] ; j < CR(mat)->rpt[i+1]; j++) {
						if (ABS(tmpcol[oidx]-i) <= bandwidth) { // in band
							int diagFound = 0;
							for (d=0; d<CR(mat)->nConstDiags; d++) {
								if (tmpcol[oidx]-i == CR(mat)->constDiags[d].idx) { // do not store constant diagonal
									oidx++;
									diagFound = 1;
									break;
								} 
							}
							if (!diagFound) {
								CR(mat)->col[idx] = tmpcol[oidx];
								CR(mat)->val[idx] = tmpval[oidx];
								idx++;
								oidx++;
							}

						} else {
							CR(mat)->col[idx] = tmpcol[oidx];
							CR(mat)->val[idx] = tmpval[oidx];
							idx++;
							oidx++;

						}
					}
				}
				free(CR(mat)->rpt);
				free(tmpval);
				free(tmpcol);

				newRowOffset[CR(mat)->nrows] = CR(mat)->nEnts;
				CR(mat)->rpt = newRowOffset;
			}
			close(pfile);
		} 
		else {

			DEBUG_LOG(2,"Allocate memory for CR(mat)->col and CR(mat)->val");
			CR(mat)->col       = (mat_idx_t *)    allocateMemory( CR(mat)->nEnts * sizeof(mat_idx_t),  "col" );
			CR(mat)->val       = (ghost_mdat_t *) allocateMemory( CR(mat)->nEnts * sizeof(ghost_mdat_t),  "val" );

			DEBUG_LOG(1,"NUMA-placement for CR(mat)->val and CR(mat)->col");
#pragma omp parallel for schedule(runtime)
			for(i = 0 ; i < CR(mat)->nrows; ++i) {
				for(j = CR(mat)->rpt[i] ; j < CR(mat)->rpt[i+1] ; j++) {
					CR(mat)->val[j] = 0.0;
					CR(mat)->col[j] = 0;
				}
			}


			DEBUG_LOG(2,"Reading array with column indices");

			// TODO fread => pread
			fread(&CR(mat)->col[0], sizeof(int), CR(mat)->nEnts, RESTFILE);
			DEBUG_LOG(2,"Reading array with values");
			if (datatype == DATATYPE_DESIRED)
			{
				fread(&CR(mat)->val[0], sizeof(ghost_mdat_t), CR(mat)->nEnts, RESTFILE);
			} 
			else 
			{
				switch (datatype) {
					case GHOST_DATATYPE_S:
						{
							float *tmp = (float *)allocateMemory(
									CR(mat)->nEnts*sizeof(float), "tmp");
							fread(tmp, sizeof(float), CR(mat)->nEnts, RESTFILE);
							for (i = 0; i<CR(mat)->nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_D:
						{
							double *tmp = (double *)allocateMemory(
									CR(mat)->nEnts*sizeof(double), "tmp");
							fread(tmp, sizeof(double), CR(mat)->nEnts, RESTFILE);
							for (i = 0; i<CR(mat)->nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_C:
						{
							_Complex float *tmp = (_Complex float *)allocateMemory(
									CR(mat)->nEnts*sizeof(_Complex float), "tmp");
							fread(tmp, sizeof(_Complex float), CR(mat)->nEnts, RESTFILE);
							for (i = 0; i<CR(mat)->nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmp[i];
							free(tmp);
							break;
						}
					case GHOST_DATATYPE_Z:
						{
							_Complex double *tmp = (_Complex double *)allocateMemory(
									CR(mat)->nEnts*sizeof(_Complex double), "tmp");
							fread(tmp, sizeof(_Complex double), CR(mat)->nEnts, RESTFILE);
							for (i = 0; i<CR(mat)->nEnts; i++) CR(mat)->val[i] = (ghost_mdat_t) tmp[i];
							free(tmp);
							break;
						}
				}
			}
		}
	}
	fclose(RESTFILE);

#ifdef OPENCL
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on OpenCL device");
		CR(mat)->clmat = (CL_CR_TYPE *)allocateMemory(sizeof(CL_CR_TYPE),"CL_CRS");
		CR(mat)->clmat->rpt = CL_allocDeviceMemory((CR(mat)->nrows+1)*sizeof(ghost_cl_mnnz_t));
		CR(mat)->clmat->col = CL_allocDeviceMemory((CR(mat)->nEnts)*sizeof(ghost_cl_midx_t));
		CR(mat)->clmat->val = CL_allocDeviceMemory((CR(mat)->nEnts)*sizeof(ghost_cl_mdat_t));
	
		CR(mat)->clmat->nrows = CR(mat)->nrows;
		CL_copyHostToDevice(CR(mat)->clmat->rpt, CR(mat)->rpt, (CR(mat)->nrows+1)*sizeof(ghost_cl_mnnz_t));
		CL_copyHostToDevice(CR(mat)->clmat->col, CR(mat)->col, CR(mat)->nEnts*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(CR(mat)->clmat->val, CR(mat)->val, CR(mat)->nEnts*sizeof(ghost_cl_mdat_t));

		cl_int err;
		cl_uint numKernels;
		cl_program program = CL_registerProgram("crs_clkernel.cl","");
		CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
		DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
		mat->clkernel = clCreateKernel(program,"CRS_kernel",&err);
		CL_checkerror(err);
		
		CL_safecall(clSetKernelArg(mat->clkernel,3,sizeof(int), &(CR(mat)->clmat->nrows)));
		CL_safecall(clSetKernelArg(mat->clkernel,4,sizeof(cl_mem), &(CR(mat)->clmat->rpt)));
		CL_safecall(clSetKernelArg(mat->clkernel,5,sizeof(cl_mem), &(CR(mat)->clmat->col)));
		CL_safecall(clSetKernelArg(mat->clkernel,6,sizeof(cl_mem), &(CR(mat)->clmat->val)));
	}
#else
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		ABORT("Device matrix cannot be created without OpenCL");
	}
#endif

	DEBUG_LOG(1,"Matrix read in successfully");
}

static void CRS_fromMM(ghost_mat_t *mat, char *matrixPath)
{
	MM_TYPE * mm = readMMFile(matrixPath);
	
	mat_idx_t* nEntsInRow;
	mat_idx_t i, e, pos;

	size_t size_rpt, size_col, size_val, size_nEntsInRow;


	/* allocate memory ######################################################## */
	DEBUG_LOG(1,"Converting MM to CRS matrix");


	size_rpt  = (size_t)( (mm->nrows+1) * sizeof( mat_idx_t ) );
	size_col        = (size_t)( mm->nEnts     * sizeof( mat_idx_t ) );
	size_val        = (size_t)( mm->nEnts     * sizeof( ghost_mdat_t) );
	size_nEntsInRow = (size_t)(  mm->nrows    * sizeof( mat_idx_t) );

	mat->data = (CR_TYPE*) allocateMemory( sizeof( CR_TYPE ), "CR(mat)" );

	CR(mat)->rpt = (mat_idx_t*)     allocateMemory( size_rpt,    "rpt" );
	CR(mat)->col = (mat_idx_t*)     allocateMemory( size_col,          "col" );
	CR(mat)->val = (ghost_mdat_t*)  allocateMemory( size_val,          "val" );
	nEntsInRow = (mat_idx_t*)     allocateMemory( size_nEntsInRow,   "nEntsInRow" );


	CR(mat)->nrows = mm->nrows;
	CR(mat)->ncols = mm->ncols;
	CR(mat)->nEnts = mm->nEnts;
	for( i = 0; i < mm->nrows; i++ ) nEntsInRow[i] = 0;

	qsort( mm->nze, (size_t)(mm->nEnts), sizeof( NZE_TYPE ), compareNZEPos );

	for( e = 0; e < mm->nEnts; e++ ) nEntsInRow[mm->nze[e].row]++;

	pos = 0;
	CR(mat)->rpt[0] = pos;
#pragma omp parallel for schedule(runtime)
	for( i = 0; i < mm->nrows; i++ ) {
		CR(mat)->rpt[i] = 0;
	}

	for( i = 0; i < mm->nrows; i++ ) {
		CR(mat)->rpt[i] = pos;
		pos += nEntsInRow[i];
	}
	CR(mat)->rpt[mm->nrows] = pos;

	for( i = 0; i < mm->nrows; i++ ) nEntsInRow[i] = 0;

#pragma omp parallel for schedule(runtime)
	for(i=0; i<CR(mat)->nrows; ++i) {
		mat_idx_t start = CR(mat)->rpt[i];
		mat_idx_t end = CR(mat)->rpt[i+1];
		mat_idx_t j;
		for(j=start; j<end; j++) {
			CR(mat)->val[j] = 0.0;
			CR(mat)->col[j] = 0;
		}
	}

	for( e = 0; e < mm->nEnts; e++ ) {
		const int row = mm->nze[e].row,
			  col = mm->nze[e].col;
		const ghost_mdat_t val = mm->nze[e].val;
		pos = CR(mat)->rpt[row] + nEntsInRow[row];
		CR(mat)->col[pos] = col;

		CR(mat)->val[pos] = val;

		nEntsInRow[row]++;
	}
	free( nEntsInRow );

	DEBUG_LOG(1,"CR matrix created from MM successfully" );

}

static void CRS_free(ghost_mat_t * mat)
{
#ifdef OPENCL
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		CL_freeDeviceMemory(CR(mat)->clmat->rpt);
		CL_freeDeviceMemory(CR(mat)->clmat->col);
		CL_freeDeviceMemory(CR(mat)->clmat->val);
	}
#endif
	free(CR(mat)->rpt);
	free(CR(mat)->col);
	free(CR(mat)->val);

	free(mat->data);
	free(mat->rowPerm);
	free(mat->invRowPerm);

	free(mat);

}

static void CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	mat_idx_t i, j;
	ghost_mdat_t hlp1;

#pragma omp	parallel for schedule(runtime) private (hlp1, j)
	for (i=0; i<CR(mat)->nrows; i++){
		hlp1 = 0.0;
		for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++){
			hlp1 = hlp1 + CR(mat)->val[j] * rhs->val[CR(mat)->col[j]]; 
		}
		if (options & GHOST_OPTION_AXPY) 
			lhs->val[i] += hlp1;
		else
			lhs->val[i] = hlp1;
	}
}

#ifdef OPENCL
static void CRS_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	CL_safecall(clSetKernelArg(mat->clkernel,0,sizeof(cl_mem), &(lhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(mat->clkernel,1,sizeof(cl_mem), &(rhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(mat->clkernel,2,sizeof(int), &options));

	size_t gSize = (size_t)CR(mat)->clmat->nrows;

	CL_enqueueKernel(mat->clkernel,&gSize,NULL);
}
#endif
