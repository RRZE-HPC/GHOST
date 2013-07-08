#include "sell.h"
#include "crs.h"
#include "ghost_mat.h"
#include "ghost_util.h"

#include <libgen.h>
#include <string.h>

#ifdef CUDA
//#include "private/sell_cukernel.h"
#endif

#if defined(SSE) || defined(AVX) || defined(MIC)
#include <immintrin.h>
#endif
#if defined(VSX)
#include <altivec.h>
#endif

void (*SELL_kernels_SSE[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
	{NULL,&dd_SELL_kernel_SSE,NULL,NULL},
	{NULL,NULL,NULL,NULL},
	{NULL,NULL,NULL,NULL}};

void (*SELL_kernels_AVX[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
	{NULL,&dd_SELL_kernel_AVX,NULL,NULL},
	{NULL,NULL,NULL,NULL},
	{NULL,NULL,NULL,NULL}};

void (*SELL_kernels_AVX_32[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
	{NULL,&dd_SELL_kernel_AVX_32,NULL,NULL},
	{NULL,NULL,NULL,NULL},
	{NULL,NULL,NULL,NULL}};

void (*SELL_kernels_MIC_16[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
	{NULL,&dd_SELL_kernel_MIC_16,NULL,NULL},
	{NULL,NULL,NULL,NULL},
	{NULL,NULL,NULL,NULL}};

void (*SELL_kernels_MIC_32[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
	{NULL,&dd_SELL_kernel_MIC_32,NULL,NULL},
	{NULL,NULL,NULL,NULL},
	{NULL,NULL,NULL,NULL}};

void (*SELL_kernels_plain[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{&ss_SELL_kernel_plain,&sd_SELL_kernel_plain,&sc_SELL_kernel_plain,&sz_SELL_kernel_plain},
	{&ds_SELL_kernel_plain,&dd_SELL_kernel_plain,&dc_SELL_kernel_plain,&dz_SELL_kernel_plain},
	{&cs_SELL_kernel_plain,&cd_SELL_kernel_plain,&cc_SELL_kernel_plain,&cz_SELL_kernel_plain},
	{&zs_SELL_kernel_plain,&zd_SELL_kernel_plain,&zc_SELL_kernel_plain,&zz_SELL_kernel_plain}};

#ifdef CUDA
void (*SELL_kernels_CU[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{&ss_SELL_kernel_CU,&sd_SELL_kernel_CU,&sc_SELL_kernel_CU,&sz_SELL_kernel_CU},
	{&ds_SELL_kernel_CU,&dd_SELL_kernel_CU,&dc_SELL_kernel_CU,&dz_SELL_kernel_CU},
	{&cs_SELL_kernel_CU,&cd_SELL_kernel_CU,&cc_SELL_kernel_CU,&cz_SELL_kernel_CU},
	{&zs_SELL_kernel_CU,&zd_SELL_kernel_CU,&zc_SELL_kernel_CU,&zz_SELL_kernel_CU}};
#endif

void (*SELL_fromCRS_funcs[4]) (ghost_mat_t *, void *) = 
{&s_SELL_fromCRS, &d_SELL_fromCRS, &c_SELL_fromCRS, &z_SELL_fromCRS}; 

//char name[] = "SELL plugin for ghost";
//char version[] = "0.1a";
//char formatID[] = "SELL";

static ghost_mnnz_t SELL_nnz(ghost_mat_t *mat);
static ghost_midx_t SELL_nrows(ghost_mat_t *mat);
static ghost_midx_t SELL_ncols(ghost_mat_t *mat);
static void SELL_printInfo(ghost_mat_t *mat);
static char * SELL_formatName(ghost_mat_t *mat);
static ghost_midx_t SELL_rowLen (ghost_mat_t *mat, ghost_midx_t i);
//static ghost_dt SELL_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j);
static size_t SELL_byteSize (ghost_mat_t *mat);
static void SELL_fromCRS(ghost_mat_t *mat, void *crs);
static void SELL_upload(ghost_mat_t* mat); 
static void SELL_CUupload(ghost_mat_t *mat);
static void SELL_fromBin(ghost_mat_t *mat, ghost_context_t *, char *);
static void SELL_free(ghost_mat_t *mat);
static void SELL_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
//#ifdef SSE_INTR
//static void SELL_kernel_SSE (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
//#endif
//#ifdef AVX_INTR
//static void SELL_kernel_AVX (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
//#endif
//#ifdef MIC_INTR
//static void SELL_kernel_MIC (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
//static void SELL_kernel_MIC_16 (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
//#endif
#ifdef OPENCL
static void SELL_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef CUDA
static void SELL_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options);
#endif

//static ghost_mat_t *thisMat;
//static SELL_TYPE *SELL(mat);

ghost_mat_t * ghost_SELL_init(ghost_mtraits_t * traits)
{
	ghost_mat_t *mat = (ghost_mat_t *)ghost_malloc(sizeof(ghost_mat_t));
	mat->traits = traits;
	DEBUG_LOG(1,"Setting functions for SELL matrix");

	mat->CLupload = &SELL_upload;
	mat->CUupload = &SELL_CUupload;
	mat->fromFile = &SELL_fromBin;
	mat->printInfo = &SELL_printInfo;
	mat->formatName = &SELL_formatName;
	mat->rowLen     = &SELL_rowLen;
	//	mat->entry      = &SELL_entry;
	mat->byteSize   = &SELL_byteSize;
	mat->kernel     = &SELL_kernel_plain;
	mat->fromCRS    = &SELL_fromCRS;
	//#ifdef SSE_INTR
	//	mat->kernel   = &SELL_kernel_SSE;
	//#endif
	//#ifdef AVX_INTR
	//	mat->kernel   = &SELL_kernel_AVX;
	//#endif
#ifdef VSX_INTR
	mat->kernel = &SELL_kernel_VSX;
#endif
	//#ifdef MIC_INTR
	//	mat->kernel   = &SELL_kernel_MIC_16;
	//	UNUSED(&SELL_kernel_MIC);
	//#endif
#ifdef OPENCL
	if (!(traits->flags & GHOST_SPM_HOST))
		mat->kernel   = &SELL_kernel_CL;
#endif
#ifdef CUDA
	if (!(traits->flags & GHOST_SPM_HOST))
		mat->kernel   = &SELL_kernel_CU;
#endif
	mat->nnz      = &SELL_nnz;
	mat->nrows    = &SELL_nrows;
	mat->ncols    = &SELL_ncols;
	mat->destroy  = &SELL_free;

	mat->localPart = NULL;
	mat->remotePart = NULL;


	/*#ifdef MIC
	  SELL(mat)->chunkHeight = 16;
#elif defined (AVX)
SELL(mat)->chunkHeight = 4;
#elif defined (SSE)
SELL(mat)->chunkHeight = 2;
#elif defined (OPENCL) || defined (CUDA)
SELL(mat)->chunkHeight = 256;
#elif defined (VSX)
SELL(mat)->chunkHeight = 2;
#else
SELL(mat)->chunkHeight = 4;
#endif*/

	return mat;
}

static ghost_mnnz_t SELL_nnz(ghost_mat_t *mat)
{
	if (mat->data == NULL)
		return -1;
	return SELL(mat)->nnz;
}
static ghost_midx_t SELL_nrows(ghost_mat_t *mat)
{
	if (mat->data == NULL)
		return -1;
	return SELL(mat)->nrows;
}
static ghost_midx_t SELL_ncols(ghost_mat_t *mat)
{
	UNUSED(mat);
	return 0;
}

static void SELL_printInfo(ghost_mat_t *mat)
{
	ghost_printLine("Chunk height",NULL,"%d",SELL(mat)->chunkHeight);
	ghost_printLine("Vectorization friendliness (beta)",NULL,"%f",SELL(mat)->beta);
	if (mat->traits->flags & GHOST_SPM_SORTED) {
		ghost_printLine("Sorted",NULL,"yes");
		ghost_printLine("Scope",NULL,"%u",*(unsigned int *)(mat->traits->aux));
		ghost_printLine("Permuted columns",NULL,"%s",mat->traits->flags&GHOST_SPM_PERMUTECOLIDX?"yes":"no");
	} else {
		ghost_printLine("Sorted",NULL,"no");
	}
}

static char * SELL_formatName(ghost_mat_t *mat)
{
	UNUSED(mat);
	return "SELL";
}

static ghost_midx_t SELL_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
	if (mat->traits->flags & GHOST_SPM_SORTED)
		i = mat->rowPerm[i];

	return SELL(mat)->rowLen[i];
}

/*static ghost_dt SELL_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j)
  {
  ghost_midx_t e;

  if (mat->traits->flags & GHOST_SPM_SORTED)
  i = mat->rowPerm[i];
  if (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX)
  j = mat->rowPerm[j];

  for (e=SELL(mat)->chunkStart[i/SELL_LEN]+i%SELL_LEN; 
  e<SELL(mat)->chunkStart[i/SELL_LEN+1]; 
  e+=SELL_LEN) {
  if (SELL(mat)->col[e] == j)
  return SELL(mat)->val[e];
  }
  return 0.;
  }*/

static size_t SELL_byteSize (ghost_mat_t *mat)
{
	return (size_t)((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_mnnz_t) + 
			SELL(mat)->nEnts*(sizeof(ghost_midx_t)+ghost_sizeofDataType(mat->traits->datatype)));
}

static void SELL_fromBin(ghost_mat_t *mat, ghost_context_t *ctx, char *matrixPath)
{
	DEBUG_LOG(1,"Creating SELL matrix from binary file");
	ghost_mtraits_t crsTraits = {.format = GHOST_SPM_FORMAT_CRS,.flags=GHOST_SPM_HOST,.datatype=mat->traits->datatype};
	ghost_mat_t *crsMat = ghost_initMatrix(&crsTraits);
	crsMat->fromFile(crsMat,ctx,matrixPath);
	mat->context = ctx;
	mat->name = basename(matrixPath);
	

#ifdef GHOST_MPI

	DEBUG_LOG(1,"Converting local and remote part to the desired data format");	
	mat->localPart = ghost_initMatrix(&mat->traits[0]); // TODO trats[1]
	mat->localPart->symmetry = crsMat->symmetry;
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
	if ((CR(crsMat)->nrows != CR(crsMat)->ncols) && (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX)) { // TODO not here???	
		WARNING_LOG("Preventing column re-ordering as the matrix is not square!");
		mat->traits->flags &= ~GHOST_SPM_PERMUTECOLIDX;
	}
	mat->fromCRS(mat,crsMat->data);
	crsMat->destroy(crsMat);

#ifdef OPENCL
	if (!(mat->traits->flags & GHOST_SPM_HOST))
		mat->CLupload(mat);
#endif
#ifdef CUDA
	if (!(mat->traits->flags & GHOST_SPM_HOST))
		mat->CUupload(mat);
#endif

	DEBUG_LOG(1,"SELL matrix successfully created");
}

static void SELL_fromCRS(ghost_mat_t *mat, void *crs)
{
	/*	DEBUG_LOG(1,"Creating SELL matrix");
		CR_TYPE *cr = (CR_TYPE*)crs;
		ghost_midx_t i,j,c;
		unsigned int flags = mat->traits->flags;

		ghost_midx_t *rowPerm = NULL;
		ghost_midx_t *invRowPerm = NULL;

		ghost_sorting_t* rowSort = NULL;


		mat->data = (SELL_TYPE *)allocateMemory(sizeof(SELL_TYPE),"SELL(mat)");
		mat->data = SELL(mat);
		mat->rowPerm = rowPerm;
		mat->invRowPerm = invRowPerm;
		if (mat->traits->flags & GHOST_SPM_SORTED) {
		rowPerm = (ghost_midx_t *)allocateMemory(cr->nrows*sizeof(ghost_midx_t),"SELL(mat)->rowPerm");
		invRowPerm = (ghost_midx_t *)ghost_malloc(cr->nrows*sizeof(ghost_midx_t));

		mat->rowPerm = rowPerm;
		mat->invRowPerm = invRowPerm;
		int sortBlock = *(int *)(mat->traits->aux);
		if (sortBlock == 0)
		sortBlock = cr->nrows;

		DEBUG_LOG(1,"Sorting matrix with a sorting block size of %d",sortBlock);

		rowSort = (ghost_sorting_t*)(cr->nrows * sizeof(ghost_sorting_t));

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

		for(i=0; i < cr->nrows; ++i) {
		if( rowSort[i].row >= cr->nrows ) DEBUG_LOG(0,"error: invalid row number %"PRmatIDX" in %"PRmatIDX,rowSort[i].row, i); 

		(invRowPerm)[i] = rowSort[i].row;
		(rowPerm)[rowSort[i].row] = i;
		}
		}




		SELL(mat)->nrows = cr->nrows;
		SELL(mat)->nnz = cr->nEnts;
		SELL(mat)->nEnts = 0;
		SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,SELL_LEN);

		ghost_midx_t nChunks = SELL(mat)->nrowsPadded/SELL_LEN;
		SELL(mat)->chunkStart = (ghost_mnnz_t *)ghost_malloc((nChunks+1)*sizeof(ghost_mnnz_t));
		SELL(mat)->chunkMin = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
		SELL(mat)->chunkLen = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
		SELL(mat)->rowLen = (ghost_midx_t *)ghost_malloc((SELL(mat)->nrowsPadded)*sizeof(ghost_midx_t));
		SELL(mat)->chunkStart[0] = 0;

		ghost_midx_t chunkMin = cr->ncols;
		ghost_midx_t chunkLen = 0;
	ghost_midx_t chunkEnts = 0;
	ghost_mnnz_t nnz = 0;
	double chunkAvg = 0.;
	ghost_midx_t curChunk = 1;
	SELL(mat)->nu = 0.;
	SELL(mat)->mu = 0.;
	SELL(mat)->beta = 0.;

	for (i=0; i<SELL(mat)->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				SELL(mat)->rowLen[i] = rowSort[i].nEntsInRow;
			else
				SELL(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			SELL(mat)->rowLen[i] = 0;
		}
		nnz += SELL(mat)->rowLen[i];


		chunkMin = SELL(mat)->rowLen[i]<chunkMin?SELL(mat)->rowLen[i]:chunkMin;
		chunkLen = SELL(mat)->rowLen[i]>chunkLen?SELL(mat)->rowLen[i]:chunkLen;
		chunkAvg += SELL(mat)->rowLen[i];
		chunkEnts += SELL(mat)->rowLen[i];

		if ((i+1)%SELL_LEN == 0) {
			chunkAvg /= (double)SELL_LEN;

			SELL(mat)->nEnts += SELL_LEN*chunkLen;
			SELL(mat)->chunkStart[curChunk] = SELL(mat)->nEnts;
			SELL(mat)->chunkMin[curChunk-1] = chunkMin;
			SELL(mat)->chunkLen[curChunk-1] = chunkLen;

			SELL(mat)->nu += (double)chunkMin/chunkLen;
			SELL(mat)->mu += (double)chunkAvg*1.0/(double)chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			chunkAvg = 0;
			curChunk++;
			chunkEnts = 0;
		}
	}
	SELL(mat)->nu /= (double)nChunks;
	SELL(mat)->mu /= (double)nChunks;
	SELL(mat)->beta = nnz*1.0/(double)SELL(mat)->nEnts;

	//SELL(mat)->val = (ghost_dt *)allocateMemory(sizeof(ghost_dt)*SELL(mat)->nEnts,"SELL(mat)->val");
	SELL(mat)->val = ghost_malloc(ghost_sizeofDataType(mat->traits->datatype)*SELL(mat)->nEnts);
	SELL(mat)->col = (ghost_midx_t *)ghost_malloc(sizeof(ghost_midx_t)*SELL(mat)->nEnts);

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<SELL(mat)->nrowsPadded/SELL_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])/SELL_LEN; j++)
		{
			for (i=0; i<SELL_LEN; i++)
			{
				SELL(mat)->val[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = 0.;
				SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = 0;
			}
		}
	}



	for (c=0; c<nChunks; c++) {

		for (j=0; j<SELL(mat)->chunkLen[c]; j++) {

			for (i=0; i<SELL_LEN; i++) {
				ghost_midx_t row = c*SELL_LEN+i;

				if (j<SELL(mat)->rowLen[row]) {
					if (flags & GHOST_SPM_SORTED) {
						SELL(mat)->val[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = cr->val[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						SELL(mat)->val[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = cr->val[cr->rpt[row]+j];
						SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					SELL(mat)->val[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = 0.0;
					SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL_LEN+i] = 0;
				}
			}
		}
	}
	DEBUG_LOG(1,"Successfully created SELL");*/

		SELL_fromCRS_funcs[ghost_dataTypeIdx(mat->traits->datatype)](mat,crs);

}

static void SELL_upload(ghost_mat_t* mat) 
{
	DEBUG_LOG(1,"Uploading SELL matrix to device");
#ifdef OPENCL
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on OpenCL device");
		SELL(mat)->clmat = (CL_SELL_TYPE *)ghost_malloc(sizeof(CL_SELL_TYPE));
		SELL(mat)->clmat->rowLen = CL_allocDeviceMemory((SELL(mat)->nrows)*sizeof(ghost_cl_midx_t));
		SELL(mat)->clmat->col = CL_allocDeviceMemory((SELL(mat)->nEnts)*sizeof(ghost_cl_midx_t));
		SELL(mat)->clmat->val = CL_allocDeviceMemory((SELL(mat)->nEnts)*ghost_sizeofDataType(mat->traits->datatype));
		SELL(mat)->clmat->chunkStart = CL_allocDeviceMemory((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_cl_mnnz_t));
		SELL(mat)->clmat->chunkLen = CL_allocDeviceMemory((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_cl_midx_t));

		SELL(mat)->clmat->nrows = SELL(mat)->nrows;
		SELL(mat)->clmat->nrowsPadded = SELL(mat)->nrowsPadded;
		CL_copyHostToDevice(SELL(mat)->clmat->rowLen, SELL(mat)->rowLen, SELL(mat)->nrows*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(SELL(mat)->clmat->col, SELL(mat)->col, SELL(mat)->nEnts*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(SELL(mat)->clmat->val, SELL(mat)->val, SELL(mat)->nEnts*ghost_sizeofDataType(mat->traits->datatype));
		CL_copyHostToDevice(SELL(mat)->clmat->chunkStart, SELL(mat)->chunkStart, (SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_cl_mnnz_t));
		CL_copyHostToDevice(SELL(mat)->clmat->chunkLen, SELL(mat)->chunkLen, (SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_cl_midx_t));

		int nDigits = (int)log10(SELL(mat)->chunkHeight)+1;
		char options[128];
		char sellLenStr[32];
		snprintf(sellLenStr,32,"-DSELL_LEN=%d",SELL(mat)->chunkHeight);
		int sellLenStrlen = 11+nDigits;
		strncpy(options,sellLenStr,sellLenStrlen);


		cl_int err;
		cl_uint numKernels;

		if (mat->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
			if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
				strncpy(options+sellLenStrlen," -DGHOST_MAT_C",14);
			} else {
				strncpy(options+sellLenStrlen," -DGHOST_MAT_Z",14);
			}
		} else {
			if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
				strncpy(options+sellLenStrlen," -DGHOST_MAT_S",14);
			} else {
				strncpy(options+sellLenStrlen," -DGHOST_MAT_D",14);
			}

		}
		strncpy(options+sellLenStrlen+14," -DGHOST_VEC_S\0",15);
		cl_program program = CL_registerProgram("sell_clkernel.cl",options);
		CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
		DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
		mat->clkernel[0] = clCreateKernel(program,"SELL_kernel",&err);
		CL_checkerror(err);

		strncpy(options+sellLenStrlen+14," -DGHOST_VEC_D\0",15);
		program = CL_registerProgram("sell_clkernel.cl",options);
		CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
		DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
		mat->clkernel[1] = clCreateKernel(program,"SELL_kernel",&err);
		CL_checkerror(err);

		strncpy(options+sellLenStrlen+14," -DGHOST_VEC_C\0",15);
		program = CL_registerProgram("sell_clkernel.cl",options);
		CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
		DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
		mat->clkernel[2] = clCreateKernel(program,"SELL_kernel",&err);
		CL_checkerror(err);

		strncpy(options+sellLenStrlen+14," -DGHOST_VEC_Z\0",15);
		program = CL_registerProgram("sell_clkernel.cl",options);
		CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
		DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
		mat->clkernel[3] = clCreateKernel(program,"SELL_kernel",&err);
		CL_checkerror(err);

		int i;
		for (i=0; i<4; i++) {
			CL_safecall(clSetKernelArg(mat->clkernel[i],3,sizeof(ghost_cl_midx_t), &(SELL(mat)->clmat->nrows)));
			CL_safecall(clSetKernelArg(mat->clkernel[i],4,sizeof(ghost_cl_midx_t), &(SELL(mat)->clmat->nrowsPadded)));
			CL_safecall(clSetKernelArg(mat->clkernel[i],5,sizeof(cl_mem), &(SELL(mat)->clmat->rowLen)));
			CL_safecall(clSetKernelArg(mat->clkernel[i],6,sizeof(cl_mem), &(SELL(mat)->clmat->col)));
			CL_safecall(clSetKernelArg(mat->clkernel[i],7,sizeof(cl_mem), &(SELL(mat)->clmat->val)));
			CL_safecall(clSetKernelArg(mat->clkernel[i],8,sizeof(cl_mem), &(SELL(mat)->clmat->chunkStart)));
			CL_safecall(clSetKernelArg(mat->clkernel[i],9,sizeof(cl_mem), &(SELL(mat)->clmat->chunkLen)));
		}
		//	printf("### %lu\n",CL_getLocalSize(mat->clkernel));
		CL_checkerror(err);

	}
#else
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		ABORT("Device matrix cannot be created without OpenCL");
	}
#endif
}

static void SELL_CUupload(ghost_mat_t* mat) 
{
	DEBUG_LOG(1,"Uploading SELL matrix to CUDA device");
#ifdef CUDA
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on CUDA device");
		SELL(mat)->cumat = (CU_SELL_TYPE *)ghost_malloc(sizeof(CU_SELL_TYPE));
		SELL(mat)->cumat->rowLen = CU_allocDeviceMemory((SELL(mat)->nrows)*sizeof(ghost_midx_t));
		SELL(mat)->cumat->col = CU_allocDeviceMemory((SELL(mat)->nEnts)*sizeof(ghost_midx_t));
		SELL(mat)->cumat->val = CU_allocDeviceMemory((SELL(mat)->nEnts)*ghost_sizeofDataType(mat->traits->datatype));
		SELL(mat)->cumat->chunkStart = CU_allocDeviceMemory((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_mnnz_t));
		SELL(mat)->cumat->chunkLen = CU_allocDeviceMemory((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_midx_t));

		SELL(mat)->cumat->nrows = SELL(mat)->nrows;
		SELL(mat)->cumat->nrowsPadded = SELL(mat)->nrowsPadded;
		CU_copyHostToDevice(SELL(mat)->cumat->rowLen, SELL(mat)->rowLen, SELL(mat)->nrows*sizeof(ghost_midx_t));
		CU_copyHostToDevice(SELL(mat)->cumat->col, SELL(mat)->col, SELL(mat)->nEnts*sizeof(ghost_midx_t));
		CU_copyHostToDevice(SELL(mat)->cumat->val, SELL(mat)->val, SELL(mat)->nEnts*ghost_sizeofDataType(mat->traits->datatype));
		CU_copyHostToDevice(SELL(mat)->cumat->chunkStart, SELL(mat)->chunkStart, (SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_mnnz_t));
		CU_copyHostToDevice(SELL(mat)->cumat->chunkLen, SELL(mat)->chunkLen, (SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_midx_t));
	}
#else
	if (mat->traits->flags & GHOST_SPM_DEVICE) {
		ABORT("Device matrix cannot be created without CUDA");
	}
#endif
}


static void SELL_free(ghost_mat_t *mat)
{
	free(SELL(mat)->val);
	free(SELL(mat)->col);
	free(SELL(mat)->chunkStart);
	free(SELL(mat)->chunkMin);
	free(SELL(mat)->chunkLen);
	free(SELL(mat)->rowLen);

	free(mat->data);
	free(mat->rowPerm);
	free(mat->invRowPerm);

	free(mat);

}

static void SELL_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	DEBUG_LOG(2,"lhs vector has %s data and %d sub-vectors",ghost_datatypeName(lhs->traits->datatype),lhs->traits->nvecs);

	void (*kernel) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int) = NULL;

#ifdef SSE_INTR
	kernel = SELL_kernels_SSE
		[ghost_dataTypeIdx(mat->traits->datatype)]
		[ghost_dataTypeIdx(lhs->traits->datatype)];
#elif defined(AVX_INTR)
	if (SELL(mat)->chunkHeight == 4) {
	kernel = SELL_kernels_AVX
		[ghost_dataTypeIdx(mat->traits->datatype)]
		[ghost_dataTypeIdx(lhs->traits->datatype)];
	} else if (SELL(mat)->chunkHeight == 32) {
	kernel = SELL_kernels_AVX_32
		[ghost_dataTypeIdx(mat->traits->datatype)]
		[ghost_dataTypeIdx(lhs->traits->datatype)];
	}

#elif defined(MIC_INTR)
#ifndef LONGIDX
	if (SELL(mat)->chunkHeight == 16) {
//	printf("$$$$$$$$$$1\n");
		kernel = SELL_kernels_MIC_16
			[ghost_dataTypeIdx(mat->traits->datatype)]
			[ghost_dataTypeIdx(lhs->traits->datatype)];
	} else if (SELL(mat)->chunkHeight == 32) {
	printf("$$$$$$$$$$2\n");
		kernel = SELL_kernels_MIC_32
			[ghost_dataTypeIdx(mat->traits->datatype)]
			[ghost_dataTypeIdx(lhs->traits->datatype)];
	}

#endif
#else
	kernel = SELL_kernels_plain
		[ghost_dataTypeIdx(mat->traits->datatype)]
		[ghost_dataTypeIdx(lhs->traits->datatype)];
#endif

	if (kernel == NULL) {
		WARNING_LOG("Selected kernel cannot be found. Falling back to plain C version!");
		kernel = SELL_kernels_plain
			[ghost_dataTypeIdx(mat->traits->datatype)]
			[ghost_dataTypeIdx(lhs->traits->datatype)];
	}

	kernel(mat,lhs,rhs,options);
}


#ifdef CUDA
static void SELL_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	DEBUG_LOG(1,"Calling SELL CUDA kernel");
	DEBUG_LOG(2,"lhs vector has %s data",ghost_datatypeName(lhs->traits->datatype));

	/*if (lhs->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
	  if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
	  c_SELL_kernel_wrap(mat, lhs, rhs, options);
	  else
	  s_SELL_kernel_wrap(mat, lhs, rhs, options);
	  } else {
	  if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
	  z_SELL_kernel_wrap(mat, lhs, rhs, options);
	  else
	  d_SELL_kernel_wrap(mat, lhs, rhs, options);
	  }*/
	SELL_kernels_CU
		[ghost_dataTypeIdx(mat->traits->datatype)]
		[ghost_dataTypeIdx(lhs->traits->datatype)](mat,lhs,rhs,options);


}
#endif

#ifdef OPENCL
static void SELL_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	cl_kernel kernel = mat->clkernel[ghost_dataTypeIdx(rhs->traits->datatype)];
	CL_safecall(clSetKernelArg(kernel,0,sizeof(cl_mem), &(lhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(kernel,1,sizeof(cl_mem), &(rhs->CL_val_gpu)));
	CL_safecall(clSetKernelArg(kernel,2,sizeof(int), &options));
	if (mat->traits->shift != NULL) {
		CL_safecall(clSetKernelArg(kernel,10,ghost_sizeofDataType(mat->traits->datatype), mat->traits->shift));
	} else {
		if (options & GHOST_SPMVM_APPLY_SHIFT)
			ABORT("A shift should be applied but the pointer is NULL!");
		complex double foo = 0.+I*0.; // should never be needed
		CL_safecall(clSetKernelArg(kernel,10,ghost_sizeofDataType(mat->traits->datatype), &foo))
	}

	size_t gSize = (size_t)SELL(mat)->clmat->nrowsPadded;
	size_t lSize = SELL(mat)->chunkHeight;

	CL_enqueueKernel(kernel,1,&gSize,&lSize);
}
#endif

#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	vector double tmp;
	vector double val;
	vector double rhs;


#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
	for (c=0; c<SELL(mat)->nrowsPadded>>1; c++) 
	{ // loop over chunks
		tmp = vec_splats(0.);
		offs = SELL(mat)->chunkStart[c];

		for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>1; j++) 
		{ // loop inside chunk
			val = vec_xld2(offs*sizeof(ghost_dt),SELL(mat)->val);                      // load values
			rhs = vec_insert(invec->val[SELL(mat)->col[offs++]],rhs,0);
			rhs = vec_insert(invec->val[SELL(mat)->col[offs++]],rhs,1);
			tmp = vec_madd(val,rhs,tmp);
		}
		if (options & GHOST_SPMVM_AXPY) {
			vec_xstd2(vec_add(tmp,vec_xld2(c*SELL(mat)->chunkHeight*sizeof(ghost_dt),lhs->val)),c*SELL(mat)->chunkHeight*sizeof(ghost_dt),lhs->val);
		} else {
			vec_xstd2(tmp,c*SELL(mat)->chunkHeight*sizeof(ghost_dt),lhs->val);
		}
	}
}
#endif
