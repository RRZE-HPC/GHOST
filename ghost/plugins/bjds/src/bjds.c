#include "bjds.h"
#include "crs.h"
#include "ghost_mat.h"
#include "ghost_util.h"

#include <libgen.h>

#ifdef CUDA
#include "private/bjds_cukernel.h"
#endif

#if defined(SSE) || defined(AVX) || defined(MIC)
#include <immintrin.h>
#endif
#if defined(VSX)
#include <altivec.h>
#endif

#define BJDS(mat) ((BJDS_TYPE *)(mat->data))

#define vecdt float
#define prefix s
#include "bjds_kernel.def"
#undef vecdt
#undef prefix

#define vecdt double
#define prefix d
#include "bjds_kernel.def"
#undef vecdt
#undef prefix

#define vecdt complex double
#define prefix z
#include "bjds_kernel.def"
#undef vecdt
#undef prefix

#define vecdt complex float
#define prefix c
#include "bjds_kernel.def"
#undef vecdt
#undef prefix

char name[] = "BJDS plugin for ghost";
char version[] = "0.1a";
char formatID[] = "BJDS";

static ghost_mnnz_t BJDS_nnz(ghost_mat_t *mat);
static ghost_midx_t BJDS_nrows(ghost_mat_t *mat);
static ghost_midx_t BJDS_ncols(ghost_mat_t *mat);
static void BJDS_printInfo(ghost_mat_t *mat);
static char * BJDS_formatName(ghost_mat_t *mat);
static ghost_midx_t BJDS_rowLen (ghost_mat_t *mat, ghost_midx_t i);
//static ghost_dt BJDS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j);
static size_t BJDS_byteSize (ghost_mat_t *mat);
static void BJDS_fromCRS(ghost_mat_t *mat, void *crs);
static void BJDS_upload(ghost_mat_t* mat); 
static void BJDS_CUupload(ghost_mat_t *mat);
static void BJDS_fromBin(ghost_mat_t *mat, ghost_context_t *, char *);
static void BJDS_free(ghost_mat_t *mat);
static void BJDS_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#ifdef SSE_INTR
static void BJDS_kernel_SSE (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef AVX_INTR
static void BJDS_kernel_AVX (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef MIC_INTR
static void BJDS_kernel_MIC (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
static void BJDS_kernel_MIC_16 (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef OPENCL
static void BJDS_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef CUDA
static void BJDS_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef VSX_INTR
static void BJDS_kernel_VSX (ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options);
#endif

//static ghost_mat_t *thisMat;
//static BJDS_TYPE *BJDS(mat);

ghost_mat_t * init(ghost_mtraits_t * traits)
{
	ghost_mat_t *mat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	mat->traits = traits;
	DEBUG_LOG(1,"Setting functions for TBJDS matrix");

	mat->CLupload = &BJDS_upload;
	mat->CUupload = &BJDS_CUupload;
	mat->fromFile = &BJDS_fromBin;
	mat->printInfo = &BJDS_printInfo;
	mat->formatName = &BJDS_formatName;
	mat->rowLen     = &BJDS_rowLen;
//	mat->entry      = &BJDS_entry;
	mat->byteSize   = &BJDS_byteSize;
	mat->kernel     = &BJDS_kernel_plain;
	mat->fromCRS    = &BJDS_fromCRS;
#ifdef SSE_INTR
	mat->kernel   = &BJDS_kernel_SSE;
#endif
#ifdef AVX_INTR
	mat->kernel   = &BJDS_kernel_AVX;
#endif
#ifdef VSX_INTR
	mat->kernel = &BJDS_kernel_VSX;
#endif
#ifdef MIC_INTR
	mat->kernel   = &BJDS_kernel_MIC_16;
	UNUSED(&BJDS_kernel_MIC);
#endif
#ifdef OPENCL
	if (!(traits->flags & GHOST_SPM_HOST))
		mat->kernel   = &BJDS_kernel_CL;
#endif
#ifdef CUDA
	if (!(traits->flags & GHOST_SPM_HOST))
		mat->kernel   = &BJDS_kernel_CU;
#endif
	mat->nnz      = &BJDS_nnz;
	mat->nrows    = &BJDS_nrows;
	mat->ncols    = &BJDS_ncols;
	mat->destroy  = &BJDS_free;

	mat->localPart = NULL;
	mat->remotePart = NULL;

	return mat;
}

static ghost_mnnz_t BJDS_nnz(ghost_mat_t *mat)
{
	if (mat->data == NULL)
		return -1;
	return BJDS(mat)->nnz;
}
static ghost_midx_t BJDS_nrows(ghost_mat_t *mat)
{
	if (mat->data == NULL)
		return -1;
	return BJDS(mat)->nrows;
}
static ghost_midx_t BJDS_ncols(ghost_mat_t *mat)
{
	UNUSED(mat);
	return 0;
}

static void BJDS_printInfo(ghost_mat_t *mat)
{
	ghost_printLine("Vector block size",NULL,"%d",BJDS_LEN);
	ghost_printLine("Nu",NULL,"%f",BJDS(mat)->nu);
	ghost_printLine("Mu",NULL,"%f",BJDS(mat)->mu);
	ghost_printLine("Beta",NULL,"%f",BJDS(mat)->beta);
	if (mat->traits->flags & GHOST_SPM_SORTED) {
		ghost_printLine("Sorted",NULL,"yes");
		ghost_printLine("Sort block size",NULL,"%u",*(unsigned int *)(mat->traits->aux));
		ghost_printLine("Permuted columns",NULL,"%s",mat->traits->flags&GHOST_SPM_PERMUTECOLIDX?"yes":"no");
	} else {
		ghost_printLine("Sorted",NULL,"no");
	}
}

static char * BJDS_formatName(ghost_mat_t *mat)
{
	UNUSED(mat);
	return "BJDS";
}

static ghost_midx_t BJDS_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
	if (mat->traits->flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];

	return BJDS(mat)->rowLen[i];
}

/*static ghost_dt BJDS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j)
{
	ghost_midx_t e;

	if (mat->traits->flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];
	if (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX)
		j = mat->rowPerm[j];

	for (e=BJDS(mat)->chunkStart[i/BJDS_LEN]+i%BJDS_LEN; 
			e<BJDS(mat)->chunkStart[i/BJDS_LEN+1]; 
			e+=BJDS_LEN) {
		if (BJDS(mat)->col[e] == j)
			return BJDS(mat)->val[e];
	}
	return 0.;
}*/

static size_t BJDS_byteSize (ghost_mat_t *mat)
{
	return (size_t)((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_mnnz_t) + 
			BJDS(mat)->nEnts*(sizeof(ghost_midx_t)+sizeof(ghost_dt)));
}

static void BJDS_fromBin(ghost_mat_t *mat, ghost_context_t *ctx, char *matrixPath)
{
	DEBUG_LOG(1,"Creating BJDS matrix from binary file");
	ghost_mtraits_t crsTraits = {.format = "CRS",.flags=GHOST_SPM_HOST,NULL};
	ghost_mat_t *crsMat = ghost_initMatrix(&crsTraits);
	crsMat->fromFile(crsMat,ctx,matrixPath);
	mat->context = ctx;
	mat->name = basename(matrixPath);

#ifdef MPI
	
	DEBUG_LOG(1,"Converting local and remote part to the desired data format");	
	mat->localPart = ghost_initMatrix(&mat->traits[0]); // TODO trats[1]
	mat->localPart->symmetry = mat->symmetry;
	mat->localPart->fromCRS(mat->localPart,crsMat->localPart->data);

	mat->remotePart = ghost_initMatrix(&mat->traits[0]); // TODO traits[2]
	mat->remotePart->fromCRS(mat->remotePart,crsMat->remotePart->data);


#ifdef OPENCL
		if (!(mat->localPart->traits->flags & GHOST_SPM_HOST))
			mat->localPart->CLupload(mat->localPart);
		if (!(mat->remotePart->traits->flags & GHOST_SPM_HOST))
			mat->remotePart->CLupload(mat->remotePart);
#endif
#ifdef CUDA
		if (!(mat->localPart->traits->flags & GHOST_SPM_HOST))
			mat->localPart->CUupload(mat->localPart);
		if (!(mat->remotePart->traits->flags & GHOST_SPM_HOST))
			mat->remotePart->CUupload(mat->remotePart);
#endif
#endif

	mat->symmetry = crsMat->symmetry;
	mat->fromCRS(mat,crsMat->data);
	crsMat->destroy(crsMat);
	
#ifdef OPENCL
		if (!(context->fullMatrix->traits->flags & GHOST_SPM_HOST))
			mat->CLupload(mat);
#endif
#ifdef CUDA
		if (!(mat->traits->flags & GHOST_SPM_HOST))
			mat->CUupload(mat);
#endif

	DEBUG_LOG(1,"BJDS matrix successfully created");
}

static void BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{
	DEBUG_LOG(1,"Creating BJDS matrix");
	CR_TYPE *cr = (CR_TYPE*)crs;
	ghost_midx_t i,j,c;
	unsigned int flags = mat->traits->flags;

	ghost_midx_t *rowPerm = NULL;
	ghost_midx_t *invRowPerm = NULL;

	ghost_sorting_t* rowSort = NULL;


	mat->data = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"BJDS(mat)");
	mat->data = BJDS(mat);
	mat->rowPerm = rowPerm;
	mat->invRowPerm = invRowPerm;
	if (mat->traits->flags & GHOST_SPM_SORTED) {
		rowPerm = (ghost_midx_t *)allocateMemory(cr->nrows*sizeof(ghost_midx_t),"BJDS(mat)->rowPerm");
		invRowPerm = (ghost_midx_t *)allocateMemory(cr->nrows*sizeof(ghost_midx_t),"BJDS(mat)->invRowPerm");

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
		}
		for( i = c*sortBlock; i < cr->nrows; i++ ) 
		{ // remainder
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
		}
		qsort( rowSort+c*sortBlock, cr->nrows-c*sortBlock, sizeof( ghost_sorting_t  ), compareNZEPerRow );

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
	}




	BJDS(mat)->nrows = cr->nrows;
	BJDS(mat)->nnz = cr->nEnts;
	BJDS(mat)->nEnts = 0;
	BJDS(mat)->nrowsPadded = ghost_pad(BJDS(mat)->nrows,BJDS_LEN);

	ghost_midx_t nChunks = BJDS(mat)->nrowsPadded/BJDS_LEN;
	BJDS(mat)->chunkStart = (ghost_mnnz_t *)allocateMemory((nChunks+1)*sizeof(ghost_mnnz_t),"BJDS(mat)->chunkStart");
	BJDS(mat)->chunkMin = (ghost_midx_t *)allocateMemory((nChunks)*sizeof(ghost_midx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->chunkLen = (ghost_midx_t *)allocateMemory((nChunks)*sizeof(ghost_midx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->rowLen = (ghost_midx_t *)allocateMemory((BJDS(mat)->nrowsPadded)*sizeof(ghost_midx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->chunkStart[0] = 0;

	ghost_midx_t chunkMin = cr->ncols;
	ghost_midx_t chunkLen = 0;
	ghost_midx_t chunkEnts = 0;
	ghost_mnnz_t nnz = 0;
	double chunkAvg = 0.;
	ghost_midx_t curChunk = 1;
	BJDS(mat)->nu = 0.;
	BJDS(mat)->mu = 0.;
	BJDS(mat)->beta = 0.;

	for (i=0; i<BJDS(mat)->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				BJDS(mat)->rowLen[i] = rowSort[i].nEntsInRow;
			else
				BJDS(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			BJDS(mat)->rowLen[i] = 0;
		}
		nnz += BJDS(mat)->rowLen[i];


		chunkMin = BJDS(mat)->rowLen[i]<chunkMin?BJDS(mat)->rowLen[i]:chunkMin;
		chunkLen = BJDS(mat)->rowLen[i]>chunkLen?BJDS(mat)->rowLen[i]:chunkLen;
		chunkAvg += BJDS(mat)->rowLen[i];
		chunkEnts += BJDS(mat)->rowLen[i];

		if ((i+1)%BJDS_LEN == 0) {
			chunkAvg /= (double)BJDS_LEN;

			BJDS(mat)->nEnts += BJDS_LEN*chunkLen;
			BJDS(mat)->chunkStart[curChunk] = BJDS(mat)->nEnts;
			BJDS(mat)->chunkMin[curChunk-1] = chunkMin;
			BJDS(mat)->chunkLen[curChunk-1] = chunkLen;

			BJDS(mat)->nu += (double)chunkMin/chunkLen;
			BJDS(mat)->mu += (double)chunkAvg*1.0/(double)chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			chunkAvg = 0;
			curChunk++;
			chunkEnts = 0;
		}
	}
	BJDS(mat)->nu /= (double)nChunks;
	BJDS(mat)->mu /= (double)nChunks;
	BJDS(mat)->beta = nnz*1.0/(double)BJDS(mat)->nEnts;

	BJDS(mat)->val = (ghost_dt *)allocateMemory(sizeof(ghost_dt)*BJDS(mat)->nEnts,"BJDS(mat)->val");
	BJDS(mat)->col = (ghost_midx_t *)allocateMemory(sizeof(ghost_midx_t)*BJDS(mat)->nEnts,"BJDS(mat)->col");

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<BJDS(mat)->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])/BJDS_LEN; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0.;
				BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0;
			}
		}
	}



	for (c=0; c<nChunks; c++) {

		for (j=0; j<BJDS(mat)->chunkLen[c]; j++) {

			for (i=0; i<BJDS_LEN; i++) {
				ghost_midx_t row = c*BJDS_LEN+i;

				if (j<BJDS(mat)->rowLen[row]) {
					if (flags & GHOST_SPM_SORTED) {
						BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->val[cr->rpt[row]+j];
						BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0.0;
					BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0;
				}
			//	printf("%f ",BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i]);


			}
		}
	}



	DEBUG_LOG(1,"Successfully created BJDS");



}

static void BJDS_upload(ghost_mat_t* mat) 
{
	DEBUG_LOG(1,"Uploading BJDS matrix to device");
#ifdef OPENCL
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on OpenCL device");
		BJDS(mat)->clmat = (CL_BJDS_TYPE *)allocateMemory(sizeof(CL_BJDS_TYPE),"CL_CRS");
		BJDS(mat)->clmat->rowLen = CL_allocDeviceMemory((BJDS(mat)->nrows)*sizeof(ghost_cl_midx_t));
		BJDS(mat)->clmat->col = CL_allocDeviceMemory((BJDS(mat)->nEnts)*sizeof(ghost_cl_midx_t));
		BJDS(mat)->clmat->val = CL_allocDeviceMemory((BJDS(mat)->nEnts)*sizeof(ghost_cl_dt));
		BJDS(mat)->clmat->chunkStart = CL_allocDeviceMemory((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_cl_mnnz_t));
		BJDS(mat)->clmat->chunkLen = CL_allocDeviceMemory((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_cl_midx_t));
	
		BJDS(mat)->clmat->nrows = BJDS(mat)->nrows;
		BJDS(mat)->clmat->nrowsPadded = BJDS(mat)->nrowsPadded;
		CL_copyHostToDevice(BJDS(mat)->clmat->rowLen, BJDS(mat)->rowLen, BJDS(mat)->nrows*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(BJDS(mat)->clmat->col, BJDS(mat)->col, BJDS(mat)->nEnts*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(BJDS(mat)->clmat->val, BJDS(mat)->val, BJDS(mat)->nEnts*sizeof(ghost_cl_dt));
		CL_copyHostToDevice(BJDS(mat)->clmat->chunkStart, BJDS(mat)->chunkStart, (BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_cl_mnnz_t));
		CL_copyHostToDevice(BJDS(mat)->clmat->chunkLen, BJDS(mat)->chunkLen, (BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_cl_midx_t));
		char options[32];
		snprintf(options,32,"-DBJDS_LEN=%d",BJDS_LEN);

		cl_int err;
		cl_uint numKernels;
		cl_program program = CL_registerProgram("bjds_clkernel.cl",options);
		CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
		DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
		mat->clkernel = clCreateKernel(program,"BJDS_kernel",&err);
		
	//	printf("### %lu\n",CL_getLocalSize(mat->clkernel));
		CL_checkerror(err);
		
		CL_safecall(clSetKernelArg(mat->clkernel,3,sizeof(ghost_cl_midx_t), &(BJDS(mat)->clmat->nrows)));
		CL_safecall(clSetKernelArg(mat->clkernel,4,sizeof(ghost_cl_midx_t), &(BJDS(mat)->clmat->nrowsPadded)));
		CL_safecall(clSetKernelArg(mat->clkernel,5,sizeof(cl_mem), &(BJDS(mat)->clmat->rowLen)));
		CL_safecall(clSetKernelArg(mat->clkernel,6,sizeof(cl_mem), &(BJDS(mat)->clmat->col)));
		CL_safecall(clSetKernelArg(mat->clkernel,7,sizeof(cl_mem), &(BJDS(mat)->clmat->val)));
		CL_safecall(clSetKernelArg(mat->clkernel,8,sizeof(cl_mem), &(BJDS(mat)->clmat->chunkStart)));
		CL_safecall(clSetKernelArg(mat->clkernel,9,sizeof(cl_mem), &(BJDS(mat)->clmat->chunkLen)));
	}
#else
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		ABORT("Device matrix cannot be created without OpenCL");
	}
#endif
}

static void BJDS_CUupload(ghost_mat_t* mat) 
{
	DEBUG_LOG(1,"Uploading BJDS matrix to CUDA device");
#ifdef CUDA
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on CUDA device");
		BJDS(mat)->cumat = (CU_BJDS_TYPE *)allocateMemory(sizeof(CU_BJDS_TYPE),"CU_CRS");
		BJDS(mat)->cumat->rowLen = CU_allocDeviceMemory((BJDS(mat)->nrows)*sizeof(ghost_midx_t));
		BJDS(mat)->cumat->col = CU_allocDeviceMemory((BJDS(mat)->nEnts)*sizeof(ghost_midx_t));
		BJDS(mat)->cumat->val = CU_allocDeviceMemory((BJDS(mat)->nEnts)*sizeof(ghost_dt));
		BJDS(mat)->cumat->chunkStart = CU_allocDeviceMemory((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_mnnz_t));
		BJDS(mat)->cumat->chunkLen = CU_allocDeviceMemory((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_midx_t));
	
		BJDS(mat)->cumat->nrows = BJDS(mat)->nrows;
		BJDS(mat)->cumat->nrowsPadded = BJDS(mat)->nrowsPadded;
		CU_copyHostToDevice(BJDS(mat)->cumat->rowLen, BJDS(mat)->rowLen, BJDS(mat)->nrows*sizeof(ghost_midx_t));
		CU_copyHostToDevice(BJDS(mat)->cumat->col, BJDS(mat)->col, BJDS(mat)->nEnts*sizeof(ghost_midx_t));
		CU_copyHostToDevice(BJDS(mat)->cumat->val, BJDS(mat)->val, BJDS(mat)->nEnts*sizeof(ghost_dt));
		CU_copyHostToDevice(BJDS(mat)->cumat->chunkStart, BJDS(mat)->chunkStart, (BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_mnnz_t));
		CU_copyHostToDevice(BJDS(mat)->cumat->chunkLen, BJDS(mat)->chunkLen, (BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_midx_t));
	}
#else
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		ABORT("Device matrix cannot be created without CUDA");
	}
#endif
}


static void BJDS_free(ghost_mat_t *mat)
{
	free(BJDS(mat)->val);
	free(BJDS(mat)->col);
	free(BJDS(mat)->chunkStart);
	free(BJDS(mat)->chunkMin);
	free(BJDS(mat)->chunkLen);
	free(BJDS(mat)->rowLen);
	
	free(mat->data);
	free(mat->rowPerm);
	free(mat->invRowPerm);

	free(mat);

}



static void BJDS_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{

	DEBUG_LOG(2,"lhs vector has %s data and %d sub-vectors",ghost_datatypeName(lhs->traits->datatype),lhs->traits->nvecs);

	if (lhs->traits->nvecs == 1) {
		if (lhs->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
			if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
				c_BJDS_kernel_plain(mat,lhs,rhs,options);
			else
				s_BJDS_kernel_plain(mat,lhs,rhs,options);
		} else {
			if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
				z_BJDS_kernel_plain(mat,lhs,rhs,options);
			else
				d_BJDS_kernel_plain(mat,lhs,rhs,options);
		}
	} else {
		ABORT("There is no multivec variant for BJDS");
	}
}

#ifdef SSE_INTR
static void BJDS_kernel_SSE (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m128d tmp;
	__m128d val;
	__m128d rhs;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>1; c++) 
	{ // loop over chunks
		tmp = _mm_setzero_pd(); // tmp = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>1; j++) 
		{ // loop inside chunk
			val    = _mm_load_pd(&BJDS(mat)->val[offs]);                      // load values
			rhs    = _mm_loadl_pd(rhs,&invec->val[(BJDS(mat)->col[offs++])]); // load first 128 bits of RHS
			rhs    = _mm_loadh_pd(rhs,&invec->val[(BJDS(mat)->col[offs++])]);
			tmp    = _mm_add_pd(tmp,_mm_mul_pd(val,rhs));           // accumulate
		}
		if (options & GHOST_SPMVM_AXPY) {
			_mm_store_pd(&lhs->val[c*BJDS_LEN],_mm_add_pd(tmp,_mm_load_pd(&lhs->val[c*BJDS_LEN])));
		} else {
			_mm_stream_pd(&lhs->val[c*BJDS_LEN],tmp);
		}
	}


}
#endif

#ifdef AVX_INTR
static void BJDS_kernel_AVX(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,offs,rhs,rhstmp)
	for (c=0; c<BJDS(mat)->nrowsPadded>>2; c++) 
	{ // loop over chunks
		tmp = _mm256_setzero_pd(); // tmp = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>2; j++) 
		{ // loop inside chunk

			val    = _mm256_load_pd(&BJDS(mat)->val[offs]);                      // load values
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(BJDS(mat)->col[offs++])]); // load first 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(BJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,0);                  // insert to RHS
			rhstmp = _mm_loadl_pd(rhstmp,&invec->val[(BJDS(mat)->col[offs++])]); // load second 128 bits of RHS
			rhstmp = _mm_loadh_pd(rhstmp,&invec->val[(BJDS(mat)->col[offs++])]);
			rhs    = _mm256_insertf128_pd(rhs,rhstmp,1);                  // insert to RHS
			tmp    = _mm256_add_pd(tmp,_mm256_mul_pd(val,rhs));           // accumulate
		}
		if (spmvmOptions & GHOST_SPMVM_AXPY) {
			_mm256_store_pd(&res->val[c*BJDS_LEN],_mm256_add_pd(tmp,_mm256_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm256_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}
#endif

#ifdef MIC_INTR
static void BJDS_kernel_MIC(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m512d tmp;
	__m512d val;
	__m512d rhs;
	__m512i idx;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,idx,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>3; c++) 
	{ // loop over chunks
		tmp = _mm512_setzero_pd(); // tmp = 0
		//		int offset = BJDS(mat)->chunkStart[c];
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>3; j+=2) 
		{ // loop inside chunk
			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_load_epi32(&BJDS(mat)->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp = _mm512_add_pd(tmp,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & GHOST_SPMVM_AXPY) {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],_mm512_add_pd(tmp,_mm512_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}

static void BJDS_kernel_MIC_16(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m512d tmp1;
	__m512d tmp2;
	__m512d val;
	__m512d rhs;
	__m512i idx;
	UNUSED(invec);

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,idx,val,rhs,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>4; c++) 
	{ // loop over chunks
		tmp1 = _mm512_setzero_pd(); // tmp1 = 0
		tmp2 = _mm512_setzero_pd(); // tmp2 = 0
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>4; j++) 
		{ // loop inside chunk
			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_load_epi32(&BJDS(mat)->col[offs]);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			//rhs = _mm512_set1_pd(invec->val[j]);
			tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			//rhs = _mm512_set1_pd(invec->val[j]);
			tmp2 = _mm512_add_pd(tmp2,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & GHOST_SPMVM_AXPY) {
		//	_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],_mm512_add_pd(tmp1,_mm512_load_pd(&res->val[c*BJDS_LEN])));
		//	_mm512_storenrngo_pd(&res->val[c*BJDS_LEN+8],_mm512_add_pd(tmp2,_mm512_load_pd(&res->val[c*BJDS_LEN+8])));
			_mm512_store_pd(&res->val[c*BJDS_LEN],_mm512_add_pd(tmp1,_mm512_load_pd(&res->val[c*BJDS_LEN])));
			_mm512_store_pd(&res->val[c*BJDS_LEN+8],_mm512_add_pd(tmp2,_mm512_load_pd(&res->val[c*BJDS_LEN+8])));
		} else {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],tmp1);
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN+8],tmp2);
		}
	}
}
#endif

#ifdef CUDA
static void BJDS_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	DEBUG_LOG(1,"Calling BJDS CUDA kernel");
	DEBUG_LOG(2,"lhs vector has %s data",ghost_datatypeName(lhs->traits->datatype));

	if (lhs->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
		if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
			c_BJDS_kernel_wrap(mat, lhs, rhs, options);
		else
			s_BJDS_kernel_wrap(mat, lhs, rhs, options);
	} else {
		if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
			z_BJDS_kernel_wrap(mat, lhs, rhs, options);
		else
			d_BJDS_kernel_wrap(mat, lhs, rhs, options);
	}
	

}
#endif

#ifdef OPENCL
static void BJDS_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	CL_safecall(clSetKernelArg(mat->clkernel,0,sizeof(cl_mem), &(lhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(mat->clkernel,1,sizeof(cl_mem), &(rhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(mat->clkernel,2,sizeof(int), &options));

	size_t gSize = (size_t)BJDS(mat)->clmat->nrowsPadded;
	size_t lSize = BJDS_LEN;

	CL_enqueueKernel(mat->clkernel,1,&gSize,&lSize);
}
#endif

#ifdef VSX_INTR
static void BJDS_kernel_VSX (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	vector double tmp;
	vector double val;
	vector double rhs;


#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<BJDS(mat)->nrowsPadded>>1; c++) 
	{ // loop over chunks
		tmp = vec_splats(0.);
		offs = BJDS(mat)->chunkStart[c];

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])>>1; j++) 
		{ // loop inside chunk
			val = vec_xld2(offs*sizeof(ghost_dt),BJDS(mat)->val);                      // load values
			rhs = vec_insert(invec->val[BJDS(mat)->col[offs++]],rhs,0);
			rhs = vec_insert(invec->val[BJDS(mat)->col[offs++]],rhs,1);
			tmp = vec_madd(val,rhs,tmp);
		}
		if (options & GHOST_SPMVM_AXPY) {
			vec_xstd2(vec_add(tmp,vec_xld2(c*BJDS_LEN*sizeof(ghost_dt),lhs->val)),c*BJDS_LEN*sizeof(ghost_dt),lhs->val);
		} else {
			vec_xstd2(tmp,c*BJDS_LEN*sizeof(ghost_dt),lhs->val);
		}
	}
}
#endif
