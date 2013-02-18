#include "ellpack.h"
#include "crs.h"
#include "ghost_mat.h"
#include "ghost_util.h"
#ifdef CUDA
#include "private/ellpack_cukernel.h"
#endif

#include <strings.h>
#if defined(SSE) || defined(AVX) || defined(MIC)
#include <immintrin.h>
#endif

#define ELLPACK(mat) ((ELLPACK_TYPE *)(mat->data))

char name[] = "ELLPACK plugin for ghost";
char version[] = "0.1a";
char formatID[] = "ELLPACK";

static ghost_mnnz_t ELLPACK_nnz(ghost_mat_t *mat);
static ghost_midx_t ELLPACK_nrows(ghost_mat_t *mat);
static ghost_midx_t ELLPACK_ncols(ghost_mat_t *mat);
static void ELLPACK_printInfo(ghost_mat_t *mat);
static char * ELLPACK_formatName(ghost_mat_t *mat);
static ghost_midx_t ELLPACK_rowLen (ghost_mat_t *mat, ghost_midx_t i);
//static ghost_mdat_t ELLPACK_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j);
static size_t ELLPACK_byteSize (ghost_mat_t *mat);
static void ELLPACK_upload(ghost_mat_t *mat);
static void ELLPACK_CUupload(ghost_mat_t *mat);
static void ELLPACK_fromCRS(ghost_mat_t *mat, void *crs);
static void ELLPACK_fromBin(ghost_mat_t *mat, char *, ghost_context_t *ctx, int options);
static void ELLPACK_free(ghost_mat_t *mat);
static void ELLPACK_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#ifdef CUDA
static void ELLPACK_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef OPENCL
static void ELLPACK_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif

ghost_mat_t * init(ghost_mtraits_t * traits)
{
	DEBUG_LOG(1,"Setting functions for ELLPACK matrix");
	ghost_mat_t *mat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	mat->traits = traits;

	mat->fromBin = &ELLPACK_fromBin;
	mat->printInfo = &ELLPACK_printInfo;
	mat->formatName = &ELLPACK_formatName;
	mat->rowLen     = &ELLPACK_rowLen;
//	mat->entry      = &ELLPACK_entry;
	mat->byteSize   = &ELLPACK_byteSize;
	mat->kernel     = &ELLPACK_kernel_plain;
	mat->nnz      = &ELLPACK_nnz;
	mat->nrows    = &ELLPACK_nrows;
	mat->ncols    = &ELLPACK_ncols;
	mat->destroy  = &ELLPACK_free;
	mat->fromCRS  = &ELLPACK_fromCRS;
	mat->CLupload = &ELLPACK_upload;
	mat->CUupload = &ELLPACK_CUupload;
#ifdef CUDA
	if (traits->flags & GHOST_SPM_HOST)
		mat->kernel   = &ELLPACK_kernel_plain;
	else
		mat->kernel   = &ELLPACK_kernel_CU;
#elif defined(OPENCL)
	if (traits->flags & GHOST_SPM_HOST)
		mat->kernel   = &ELLPACK_kernel_plain;
	else
		mat->kernel   = &ELLPACK_kernel_CL;
#else
	mat->kernel   = &ELLPACK_kernel_plain;
#endif
		

	return mat;
}

static ghost_mnnz_t ELLPACK_nnz(ghost_mat_t *mat)
{
	return ELLPACK(mat)->nnz;
}
static ghost_midx_t ELLPACK_nrows(ghost_mat_t *mat)
{
	return ELLPACK(mat)->nrows;
}
static ghost_midx_t ELLPACK_ncols(ghost_mat_t *mat)
{
	UNUSED(mat);
	return 0;
}

static void ELLPACK_printInfo(ghost_mat_t *mat)
{
#if defined (OPENCL) || defined (CUDA)
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		ghost_printLine("Work-items per row",NULL,"%u",ELLPACK(mat)->T);
		ghost_printLine("Work-group size",NULL,"%ux%u",ELLPACK_WGXSIZE,ELLPACK(mat)->T);
	}
#else
	UNUSED(mat);
#endif
	return;
}

static char * ELLPACK_formatName(ghost_mat_t *mat)
{
	UNUSED(mat);
	return "ELLPACK";
}

static ghost_midx_t ELLPACK_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
	if (mat->traits->flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];

	return ELLPACK(mat)->rowLen[i];
}
/*
static ghost_mdat_t ELLPACK_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j)
{
	ghost_midx_t e;
	ghost_midx_t eInRow;

	if (mat->traits->flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];
	if (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX)
		j = mat->rowPerm[j];

	for (e=i, eInRow = 0; eInRow<ELLPACK(mat)->rowLen[i]; e+=ELLPACK(mat)->nrowsPadded, eInRow++) {
		if (ELLPACK(mat)->col[e] == j)
			return ELLPACK(mat)->val[e];
	}
	return 0.;
}*/

static size_t ELLPACK_byteSize (ghost_mat_t *mat)
{
	return (size_t)((ELLPACK(mat)->nrowsPadded)*sizeof(ghost_midx_t) + 
			ELLPACK(mat)->nrowsPadded*ELLPACK(mat)->maxRowLen*(sizeof(ghost_midx_t)+sizeof(ghost_dt)));
}

static void ELLPACK_fromBin(ghost_mat_t *mat, char * matrixPath, ghost_context_t *ctx, int options)
{
	ghost_mtraits_t crsTraits = {.format = "CRS",.flags=GHOST_SPM_HOST,NULL};
	ghost_mat_t *crsMat = ghost_initMatrix(&crsTraits);
	crsMat->fromBin(crsMat,matrixPath, ctx, options);

	mat->symmetry = crsMat->symmetry;
	ELLPACK_fromCRS(mat,crsMat->data);
	crsMat->destroy(crsMat);

	mat->CUupload(mat);
}

static void ELLPACK_fromCRS(ghost_mat_t *mat, void *crs)
{
	DEBUG_LOG(1,"Creating ELLPACK matrix");
	CR_TYPE *cr = (CR_TYPE*)crs;
	ghost_midx_t i,j,c;
	unsigned int flags = mat->traits->flags;

	ghost_midx_t *rowPerm = NULL;
	ghost_midx_t *invRowPerm = NULL;

	ghost_sorting_t* rowSort;

	mat->data = (ELLPACK_TYPE *)allocateMemory(sizeof(ELLPACK_TYPE),"ELLPACK(mat)");
	mat->rowPerm = rowPerm;
	mat->invRowPerm = invRowPerm;

	ELLPACK(mat)->maxRowLen = 0;

	if (mat->traits->flags & GHOST_SPM_SORTED) {
		rowPerm = (ghost_midx_t *)allocateMemory(cr->nrows*sizeof(ghost_midx_t),"ELLPACK(mat)->rowPerm");
		invRowPerm = (ghost_midx_t *)allocateMemory(cr->nrows*sizeof(ghost_midx_t),"ELLPACK(mat)->invRowPerm");

		mat->rowPerm = rowPerm;
		mat->invRowPerm = invRowPerm;
		int sortBlock = *(int *)(mat->traits->aux);
		if (sortBlock == 0)
			sortBlock = cr->nrows;

		DEBUG_LOG(1,"Sorting matrix with a sorting block size of %d",sortBlock);

		/* get max number of entries in one row ###########################*/
		rowSort = (ghost_sorting_t*) allocateMemory( cr->nrows * sizeof( ghost_sorting_t ),
				"rowSort" );

		for (c=0; c<cr->nrows/sortBlock; c++)  
		{
			for( i = c*sortBlock; i < (c+1)*sortBlock; i++ ) 
			{
				rowSort[i].row = i;
				rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
			} 

			qsort( rowSort+c*sortBlock, sortBlock, sizeof( ghost_sorting_t  ), compareNZEPerRow );

			ELLPACK(mat)->maxRowLen = MAX(ELLPACK(mat)->maxRowLen,rowSort[0].nEntsInRow);
		}
		for( i = c*sortBlock; i < cr->nrows; i++ ) 
		{ // remainder
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
		}

		/* sort within same rowlength with asceding row number #################### */
		/*i=0;
		  while(i < cr->nrows) {
		  ghost_midx_t start = i;

		  j = rowSort[start].nEntsInRow;
		  while( i<cr->nrows && rowSort[i].nEntsInRow >= j ) 
		  ++i;

		  DEBUG_LOG(1,"sorting over %"PRmatIDX" rows (%"PRmatIDX"): %"PRmatIDX" - %"PRmatIDX,i-start,j, start, i-1);
		  qsort( &rowSort[start], i-start, sizeof(ghost_sorting_t), compareNZEOrgPos );
		  }

		  for(i=1; i < cr->nrows; ++i) {
		  if( rowSort[i].nEntsInRow == rowSort[i-1].nEntsInRow && rowSort[i].row < rowSort[i-1].row)
		  printf("Error in row %"PRmatIDX": descending row number\n",i);
		  }*/
		for(i=0; i < cr->nrows; ++i) {
			/* invRowPerm maps an index in the permuted system to the original index,
			 * rowPerm gets the original index and returns the corresponding permuted position.
			 */
			if( rowSort[i].row >= cr->nrows ) DEBUG_LOG(0,"error: invalid row number %"PRmatIDX" in %"PRmatIDX,rowSort[i].row, i); 

			(invRowPerm)[i] = rowSort[i].row;
			(rowPerm)[rowSort[i].row] = i;
		}
	} else {

		/* get max number of entries in one row ###########################*/
		rowSort = (ghost_sorting_t*) allocateMemory( cr->nrows * sizeof( ghost_sorting_t ),
				"rowSort" );

		for( i = 0; i < cr->nrows; i++ ) {
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = 0;
		} 

		/* count entries per row ################################################## */
		for( i = 0; i < cr->nrows; i++) 
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];

		/* sort rows with desceding number of NZEs ################################ */
		qsort( rowSort, cr->nrows, sizeof( ghost_sorting_t  ), compareNZEPerRow );

		ELLPACK(mat)->maxRowLen = rowSort[0].nEntsInRow;
	}
	
	if (mat->traits->aux == NULL || *(ghost_midx_t *)(mat->traits->aux) == 0)
		ELLPACK(mat)->T = 1;
	else
		ELLPACK(mat)->T = *(ghost_midx_t *)(mat->traits->aux);


	ELLPACK(mat)->maxRowLen = ghost_pad(ELLPACK(mat)->maxRowLen,ELLPACK(mat)->T);

	ELLPACK(mat)->nrows = cr->nrows;
	ELLPACK(mat)->nnz = cr->nEnts;
	ELLPACK(mat)->nrowsPadded = ghost_pad(ELLPACK(mat)->nrows,ELLPACK_PAD);
	DEBUG_LOG(1,"The ELLPACK matrix has %d rows (padded to %d) and a maximum row length of %d",
		ELLPACK(mat)->nrows, ELLPACK(mat)->nrowsPadded,ELLPACK(mat)->maxRowLen);	


	ELLPACK(mat)->nEnts = ELLPACK(mat)->nrowsPadded*ELLPACK(mat)->maxRowLen;
	ELLPACK(mat)->rowLen = (ghost_midx_t *)allocateMemory(ELLPACK(mat)->nrowsPadded*sizeof(ghost_midx_t),"rowLen");
	ELLPACK(mat)->col = (ghost_midx_t *)allocateMemory(ELLPACK(mat)->nEnts*sizeof(ghost_midx_t),"col");
	ELLPACK(mat)->val = (ghost_dt *)allocateMemory(ELLPACK(mat)->nEnts*sizeof(ghost_dt),"val");


#pragma omp parallel for private(j)	
	for( i=0; i < ELLPACK(mat)->nrowsPadded; ++i) {
		for( j=0; j < ELLPACK(mat)->maxRowLen; ++j) {
			ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded] = 0;
			ELLPACK(mat)->val[i+j*ELLPACK(mat)->nrowsPadded] = 0.0;
		}
	}

	for( i=0; i < ELLPACK(mat)->nrowsPadded; ++i) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				ELLPACK(mat)->rowLen[i] = rowSort[i].nEntsInRow;
			else
				ELLPACK(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			ELLPACK(mat)->rowLen[i] = 0;
		}

		for( j=0; j < ELLPACK(mat)->rowLen[i]; ++j) {
			if (flags & GHOST_SPM_SORTED) {
				ELLPACK(mat)->val[i+j*ELLPACK(mat)->nrowsPadded] = cr->val[cr->rpt[invRowPerm[i]]+j];
				if (flags & GHOST_SPM_PERMUTECOLIDX)
					ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded] = rowPerm[cr->col[cr->rpt[invRowPerm[i]]+j]];
				else 
					ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded] = cr->col[cr->rpt[invRowPerm[i]]+j];
			} else {
				ELLPACK(mat)->val[i+j*ELLPACK(mat)->nrowsPadded] = cr->val[cr->rpt[i]+j];
				ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded] = cr->col[cr->rpt[i]+j];
			}
		}
		ELLPACK(mat)->rowLen[i] = ghost_pad(ELLPACK(mat)->rowLen[i],ELLPACK(mat)->T); // has to be done after copying!!
	}

	free( rowSort );

	DEBUG_LOG(1,"Successfully created ELLPACK");

}

static void ELLPACK_upload(ghost_mat_t *mat)
{
	DEBUG_LOG(1,"Uploading ELLPACK matrix to device");

#ifdef OPENCL
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on OpenCL device");
		ELLPACK(mat)->clmat = (CL_ELLPACK_TYPE *)allocateMemory(sizeof(CL_ELLPACK_TYPE),"CL_ELLPACK");
		ELLPACK(mat)->clmat->rowLen = CL_allocDeviceMemory((ELLPACK(mat)->nrows)*sizeof(ghost_cl_midx_t));
		ELLPACK(mat)->clmat->col = CL_allocDeviceMemory((ELLPACK(mat)->nEnts)*sizeof(ghost_cl_midx_t));
		ELLPACK(mat)->clmat->val = CL_allocDeviceMemory((ELLPACK(mat)->nEnts)*sizeof(ghost_cl_mdat_t));

		ELLPACK(mat)->clmat->nrows       = ELLPACK(mat)->nrows;
		ELLPACK(mat)->clmat->nrowsPadded = ELLPACK(mat)->nrowsPadded;
		ELLPACK(mat)->clmat->maxRowLen	 = ELLPACK(mat)->maxRowLen;

		CL_copyHostToDevice(ELLPACK(mat)->clmat->rowLen, ELLPACK(mat)->rowLen, ELLPACK(mat)->nrows*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(ELLPACK(mat)->clmat->col,    ELLPACK(mat)->col,    ELLPACK(mat)->nEnts*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(ELLPACK(mat)->clmat->val,    ELLPACK(mat)->val,    ELLPACK(mat)->nEnts*sizeof(ghost_cl_mdat_t));

		cl_int err;
			char opt[32];
			snprintf(opt,32,"-DT=%"PRmatIDX,ELLPACK(mat)->T);
			cl_program program = CL_registerProgram("ellpack_clkernel.cl",opt);

		if (ELLPACK(mat)->T == 1) {
			mat->clkernel = clCreateKernel(program,"ELLPACK_kernel",&err);
		} else {
			mat->clkernel = clCreateKernel(program,"ELLPACKT_kernel",&err);
		}
		CL_checkerror(err);
		
		CL_safecall(clSetKernelArg(mat->clkernel,3,sizeof(ghost_cl_midx_t), &(ELLPACK(mat)->clmat->nrows)));
		CL_safecall(clSetKernelArg(mat->clkernel,4,sizeof(ghost_cl_midx_t), &(ELLPACK(mat)->clmat->nrowsPadded)));
		CL_safecall(clSetKernelArg(mat->clkernel,5,sizeof(cl_mem), &(ELLPACK(mat)->clmat->rowLen)));
		CL_safecall(clSetKernelArg(mat->clkernel,6,sizeof(cl_mem), &(ELLPACK(mat)->clmat->col)));
		CL_safecall(clSetKernelArg(mat->clkernel,7,sizeof(cl_mem), &(ELLPACK(mat)->clmat->val)));
	}
#else
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		ABORT("Device matrix cannot be created without OpenCL");
	}
#endif
}

static void ELLPACK_CUupload(ghost_mat_t *mat)
{
	DEBUG_LOG(1,"Uploading ELLPACK matrix to CUDA device");

#ifdef CUDA
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on CUDA device");
		ELLPACK(mat)->cumat = (CU_ELLPACK_TYPE *)allocateMemory(sizeof(CU_ELLPACK_TYPE),"CU_ELLPACK");
		ELLPACK(mat)->cumat->rowLen = CU_allocDeviceMemory((ELLPACK(mat)->nrows)*sizeof(ghost_midx_t));
		ELLPACK(mat)->cumat->col = CU_allocDeviceMemory((ELLPACK(mat)->nEnts)*sizeof(ghost_midx_t));
		ELLPACK(mat)->cumat->val = CU_allocDeviceMemory((ELLPACK(mat)->nEnts)*sizeof(ghost_dt));

		ELLPACK(mat)->cumat->nrows       = ELLPACK(mat)->nrows;
		ELLPACK(mat)->cumat->nrowsPadded = ELLPACK(mat)->nrowsPadded;
		ELLPACK(mat)->cumat->maxRowLen	 = ELLPACK(mat)->maxRowLen;

		CU_copyHostToDevice(ELLPACK(mat)->cumat->rowLen, ELLPACK(mat)->rowLen, ELLPACK(mat)->nrows*sizeof(ghost_midx_t));
		CU_copyHostToDevice(ELLPACK(mat)->cumat->col,    ELLPACK(mat)->col,    ELLPACK(mat)->nEnts*sizeof(ghost_midx_t));
		CU_copyHostToDevice(ELLPACK(mat)->cumat->val,    ELLPACK(mat)->val,    ELLPACK(mat)->nEnts*sizeof(ghost_dt));
		DEBUG_LOG(1,"ELLPACK matrix successfully created on CUDA device");
	}

#else
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		ABORT("Device matrix cannot be created without CUDA");
	}
#endif
}

static void ELLPACK_free(ghost_mat_t *mat)
{
	free(ELLPACK(mat)->rowLen);
	free(ELLPACK(mat)->val);
	free(ELLPACK(mat)->col);
	
	free(mat->data);
	free(mat->rowPerm);
	free(mat->invRowPerm);

	free(mat);


}

static void ELLPACK_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
   double *rhsv = (double *)rhs->val;	
   double *lhsv = (double *)lhs->val;	
	ghost_midx_t j,i;
	double tmp; 

#pragma omp parallel for schedule(runtime) private(j,tmp)
	for( i=0; i < ELLPACK(mat)->nrows; ++i) {
		tmp = 0;
		for( j=0; j < ELLPACK(mat)->maxRowLen; ++j) {
			tmp += (double)ELLPACK(mat)->val[i+j*ELLPACK(mat)->nrowsPadded] * 
				rhsv[ELLPACK(mat)->col[i+j*ELLPACK(mat)->nrowsPadded]];
		}
		if (options & GHOST_SPMVM_AXPY)
			lhsv[i] += tmp;
		else
			lhsv[i] = tmp;
	}

}

#ifdef CUDA
static void ELLPACK_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	DEBUG_LOG(1,"Calling ELLPACK CUDA kernel");
	
	ELLPACK_kernel_wrap(lhs->CU_val, rhs->CU_val, options, ELLPACK(mat)->cumat->nrows, ELLPACK(mat)->cumat->nrowsPadded, ELLPACK(mat)->cumat->rowLen, ELLPACK(mat)->cumat->col, ELLPACK(mat)->cumat->val);

}
#endif

#ifdef OPENCL
static void ELLPACK_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	DEBUG_LOG(1,"Calling ELLPACK OpenCL kernel");

	CL_safecall(clSetKernelArg(mat->clkernel,0,sizeof(cl_mem), &(lhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(mat->clkernel,1,sizeof(cl_mem), &(rhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(mat->clkernel,2,sizeof(int), &options));


	char kernelName[256] = "";
	CL_safecall(clGetKernelInfo(mat->clkernel,CL_KERNEL_FUNCTION_NAME,256,kernelName,NULL));
	if (!strcasecmp(kernelName,"ELLPACKT_kernel"))
		CL_safecall(clSetKernelArg(mat->clkernel,8,sizeof(ghost_cl_mdat_t)*ELLPACK_WGXSIZE*ELLPACK(mat)->T, NULL));


	size_t lSize[] = {ELLPACK_WGXSIZE,ELLPACK(mat)->T};
	size_t gSize[] = {(size_t)ELLPACK(mat)->clmat->nrowsPadded,ELLPACK(mat)->T};
	CL_enqueueKernel(mat->clkernel,2,gSize,lSize);
}
#endif
