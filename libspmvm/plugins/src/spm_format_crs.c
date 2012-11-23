#define _XOPEN_SOURCE 500

#include "spm_format_crs.h"
//#include "spm_format_crs_clkernel.h"
#include "matricks.h"
#include "ghost_util.h"


#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

const char name[] = "CRS plugin for ghost";
const char version[] = "0.1a";
const char formatID[] = "CRS";

static mat_nnz_t CRS_nnz();
static mat_idx_t CRS_nrows();
static mat_idx_t CRS_ncols();
static void CRS_fromBin(char *matrixPath, mat_trait_t trait);
static void CRS_printInfo();
static char * CRS_formatName();
static mat_idx_t CRS_rowLen (mat_idx_t i);
static mat_data_t CRS_entry (mat_idx_t i, mat_idx_t j);
static size_t CRS_byteSize (void);
static void CRS_kernel_plain (ghost_vec_t *, ghost_vec_t *, int);
static void CRS_kernel_CL (ghost_vec_t *, ghost_vec_t *, int);

//static ghost_mat_t *thisMat;
//static CR_TYPE *thisCR;


const char *CRS_kernelSource = "#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
"\n"
"#if defined(cl_intel_printf)\n"
"#pragma OPENCL EXTENSION cl_intel_printf : enable\n"
"#elif defined(cl_amd_printf)\n"
"#pragma OPENCL EXTENSION cl_amd_printf : enable\n"
"#endif\n"
"\n"
"#ifdef DOUBLE\n"
"#ifdef COMPLEX\n"
"typedef double2 cl_mat_data_t;\n"
"#else\n"
"typedef double cl_mat_data_t;\n"
"#endif\n"
"#endif\n"
"#ifdef SINGLE\n"
"#ifdef COMPLEX\n"
"typedef float2 cl_mat_data_t;\n"
"#else\n"
"typedef float cl_mat_data_t;\n"
"#endif\n"
"#endif\n"
"\n"
"kernel void CRS_kernel (global cl_mat_data_t *lhs, global cl_mat_data_t *rhs, int options, unsigned int nrows, global unsigned int *rpt, global unsigned int *col, global cl_mat_data_t *val) \n"
"{\n"
"unsigned int i = get_global_id(0);\n"
"/*if (i < nrows) {\n"
"	cl_mat_data_t svalue = 0.0;\n"
"	for(unsigned int j=rpt[i]; j<rpt[i+1]; ++j){\n"
"		svalue += val[j] * rhs[col[j]]; \n"
"	}\n"
"	lhs[i] += svalue;\n"
"}*/\n"	
"}\n";	

void init(ghost_mat_t *mat)
{
	if (!mat)
		mat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");

	DEBUG_LOG(1,"Initializing CRS functions");

	//thisMat = mat;
	//thisCR = (CR_TYPE *)(mat->data);
//	printf("init: %p\n",thisCR);

	mat->fromBin = &CRS_fromBin;
	mat->printInfo = &CRS_printInfo;
	mat->formatName = &CRS_formatName;
	mat->rowLen   = &CRS_rowLen;
	mat->entry    = &CRS_entry;
	mat->byteSize = &CRS_byteSize;
	mat->nnz      = &CRS_nnz;
	mat->nrows    = &CRS_nrows;
	mat->ncols    = &CRS_ncols;
}

static mat_nnz_t CRS_nnz()
{
	return thisCR->nEnts;
}
static mat_idx_t CRS_nrows()
{
	return thisCR->nrows;
}
static mat_idx_t CRS_ncols()
{
	return thisCR->ncols;
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
	thisMat->trait = trait;
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

	if (trait.flags & GHOST_SPM_DEVICE) {
		DEBUG_LOG(1,"Creating matrix on OpenCL device");
		thisCR->clmat = (CL_CR_TYPE *)allocateMemory(sizeof(CL_CR_TYPE),"CL_CRS");
		thisCR->clmat->rpt = CL_allocDeviceMemory((thisCR->nrows+1)*sizeof(ghost_cl_mnnz_t));
		thisCR->clmat->col = CL_allocDeviceMemory((thisCR->nEnts)*sizeof(ghost_cl_midx_t));
		thisCR->clmat->val = CL_allocDeviceMemory((thisCR->nEnts)*sizeof(ghost_cl_mdat_t));
	
		thisCR->clmat->nrows = thisCR->nrows;
		CL_copyHostToDevice(thisCR->clmat->rpt, thisCR->rpt, (thisCR->nrows+1)*sizeof(ghost_cl_mnnz_t));
		CL_copyHostToDevice(thisCR->clmat->col, thisCR->col, thisCR->nEnts*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(thisCR->clmat->val, thisCR->val, thisCR->nEnts*sizeof(ghost_cl_mdat_t));

		cl_int err;
		cl_uint numKernels;

		cl_program program = CL_registerProgram(CRS_kernelSource," -DDOUBLE ");

		CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
		DEBUG_LOG(1,"There are %u kernels",numKernels);

		kernel = clCreateKernel(program,"CRS_kernel",&err);
		CL_checkerror(err);
		
		CL_safecall(clSetKernelArg(kernel,3,sizeof(int), &(thisCR->clmat->nrows)));
		CL_safecall(clSetKernelArg(kernel,4,sizeof(cl_mem), &(thisCR->clmat->rpt)));
		CL_safecall(clSetKernelArg(kernel,5,sizeof(cl_mem), &(thisCR->clmat->col)));
		CL_safecall(clSetKernelArg(kernel,6,sizeof(cl_mem), &(thisCR->clmat->val)));

		thisMat->kernel   = &CRS_kernel_CL;

	} else { // TODO in init()
	
		thisMat->kernel   = &CRS_kernel_plain;
	}
	printf("create: %p %p\n",thisCR,thisCR->clmat);


	DEBUG_LOG(1,"Matrix read in successfully");

}

static void CRS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	printf("plain: %p %p\n",thisCR,thisCR->clmat);
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

static void CRS_kernel_CL (ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	printf("cl: %p %p\n",thisCR,thisCR->clmat);
	CL_safecall(clSetKernelArg(kernel,0,sizeof(cl_mem), &(lhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(kernel,1,sizeof(cl_mem), &(rhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(kernel,2,sizeof(int), &options));

	CL_enqueueKernelWithSize(kernel,(size_t)thisCR->clmat->nrows);
	DEBUG_LOG(1,"Finished iteration");

}
