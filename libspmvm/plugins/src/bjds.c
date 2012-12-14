#include "spm_format_bjds.h"
#include "matricks.h"
#include "ghost_util.h"
#include "kernel.h"

#include <immintrin.h>

#define BJDS(mat) ((BJDS_TYPE *)(mat->data))

char name[] = "BJDS plugin for ghost";
char version[] = "0.1a";
char formatID[] = "BJDS";

static ghost_mnnz_t BJDS_nnz(ghost_mat_t *mat);
static ghost_midx_t BJDS_nrows(ghost_mat_t *mat);
static ghost_midx_t BJDS_ncols(ghost_mat_t *mat);
static void BJDS_printInfo(ghost_mat_t *mat);
static char * BJDS_formatName(ghost_mat_t *mat);
static ghost_midx_t BJDS_rowLen (ghost_mat_t *mat, ghost_midx_t i);
static ghost_mdat_t BJDS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j);
static size_t BJDS_byteSize (ghost_mat_t *mat);
static void BJDS_fromCRS(ghost_mat_t *mat, CR_TYPE *cr);
static void BJDS_fromBin(ghost_mat_t *mat, char *);
static void BJDS_free(ghost_mat_t *mat);
static void BJDS_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#ifdef SSE
static void BJDS_kernel_SSE (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef AVX
static void BJDS_kernel_AVX (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef MIC
static void BJDS_kernel_MIC (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
static void BJDS_kernel_MIC_16 (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif
#ifdef OPENCL
static void BJDS_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif

//static ghost_mat_t *thisMat;
//static BJDS_TYPE *BJDS(mat);

ghost_mat_t * init(ghost_mtraits_t * traits)
{
	ghost_mat_t *mat = (ghost_mat_t *)allocateMemory(sizeof(ghost_mat_t),"matrix");
	mat->traits = traits;
	DEBUG_LOG(1,"Setting functions for TBJDS matrix");

	mat->fromBin = &BJDS_fromBin;
	mat->printInfo = &BJDS_printInfo;
	mat->formatName = &BJDS_formatName;
	mat->rowLen     = &BJDS_rowLen;
	mat->entry      = &BJDS_entry;
	mat->byteSize   = &BJDS_byteSize;
	mat->kernel     = &BJDS_kernel_plain;
#ifdef SSE
	mat->kernel   = &BJDS_kernel_SSE;
#endif
#ifdef AVX
	mat->kernel   = &BJDS_kernel_AVX;
#endif
#ifdef MIC
	mat->kernel   = &BJDS_kernel_MIC_16;
	UNUSED(&BJDS_kernel_MIC);
#endif
#ifdef OPENCL
	if (!(traits->flags & GHOST_SPM_HOST))
		mat->kernel   = &BJDS_kernel_CL;
#endif
	mat->nnz      = &BJDS_nnz;
	mat->nrows    = &BJDS_nrows;
	mat->ncols    = &BJDS_ncols;
	mat->destroy  = &BJDS_free;

	return mat;
}

static ghost_mnnz_t BJDS_nnz(ghost_mat_t *mat)
{
	return BJDS(mat)->nnz;
}
static ghost_midx_t BJDS_nrows(ghost_mat_t *mat)
{
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
	ghost_printLine("Row length oscillation nu",NULL,"%f",BJDS(mat)->nu);
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

static ghost_mdat_t BJDS_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j)
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
}

static size_t BJDS_byteSize (ghost_mat_t *mat)
{
	return (size_t)((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_mnnz_t) + 
			BJDS(mat)->nEnts*(sizeof(ghost_midx_t)+sizeof(ghost_mdat_t)));
}

static void BJDS_fromBin(ghost_mat_t *mat, char *matrixPath)
{
	ghost_mtraits_t crsTraits = {.format = "CRS",.flags=GHOST_SPM_HOST,NULL};
	ghost_mat_t *crsMat = ghost_initMatrix(&crsTraits);
	crsMat->fromBin(crsMat,matrixPath);
	
	BJDS_fromCRS(mat,crsMat->data);
}

static void BJDS_fromCRS(ghost_mat_t *mat, CR_TYPE *cr)
{
	DEBUG_LOG(1,"Creating BJDS matrix");
	ghost_midx_t i,j,c;
	unsigned int flags = mat->traits->flags;

	ghost_midx_t *rowPerm = NULL;
	ghost_midx_t *invRowPerm = NULL;

	JD_SORT_TYPE* rowSort;

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
		/*i=0;
		while(i < cr->nrows) {
			ghost_midx_t start = i;

			j = rowSort[start].nEntsInRow;
			while( i<cr->nrows && rowSort[i].nEntsInRow >= j ) 
				++i;

			DEBUG_LOG(1,"sorting over %"PRmatIDX" rows (%"PRmatIDX"): %"PRmatIDX" - %"PRmatIDX,i-start,j, start, i-1);
			qsort( &rowSort[start], i-start, sizeof(JD_SORT_TYPE), compareNZEOrgPos );
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
	BJDS(mat)->nrowsPadded = pad(BJDS(mat)->nrows,BJDS_LEN);

	ghost_midx_t nChunks = BJDS(mat)->nrowsPadded/BJDS_LEN;
	BJDS(mat)->chunkStart = (ghost_mnnz_t *)allocateMemory((nChunks+1)*sizeof(ghost_mnnz_t),"BJDS(mat)->chunkStart");
	BJDS(mat)->chunkMin = (ghost_midx_t *)allocateMemory((nChunks)*sizeof(ghost_midx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->chunkLen = (ghost_midx_t *)allocateMemory((nChunks)*sizeof(ghost_midx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->rowLen = (ghost_midx_t *)allocateMemory((BJDS(mat)->nrowsPadded)*sizeof(ghost_midx_t),"BJDS(mat)->chunkMin");
	BJDS(mat)->chunkStart[0] = 0;

	ghost_midx_t chunkMin = cr->ncols;
	ghost_midx_t chunkLen = 0;
	ghost_midx_t curChunk = 1;
	BJDS(mat)->nu = 0.;

	for (i=0; i<BJDS(mat)->nrowsPadded; i++) {
		if (i<cr->nrows) {
			if (flags & GHOST_SPM_SORTED)
				BJDS(mat)->rowLen[i] = rowSort[i].nEntsInRow;
			else
				BJDS(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
		} else {
			BJDS(mat)->rowLen[i] = 0;
		}


		chunkMin = BJDS(mat)->rowLen[i]<chunkMin?BJDS(mat)->rowLen[i]:chunkMin;
		chunkLen = BJDS(mat)->rowLen[i]>chunkLen?BJDS(mat)->rowLen[i]:chunkLen;

		if ((i+1)%BJDS_LEN == 0) {
			BJDS(mat)->nEnts += BJDS_LEN*chunkLen;
			BJDS(mat)->chunkStart[curChunk] = BJDS(mat)->nEnts;
			BJDS(mat)->chunkMin[curChunk-1] = chunkMin;
			BJDS(mat)->chunkLen[curChunk-1] = chunkLen;

			BJDS(mat)->nu += (double)chunkMin/chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			curChunk++;
		}
	}
	BJDS(mat)->nu /= (double)nChunks;

	BJDS(mat)->val = (ghost_mdat_t *)allocateMemory(sizeof(ghost_mdat_t)*BJDS(mat)->nEnts,"BJDS(mat)->val");
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
#ifdef OPENCL
	if (!(mat->traits->flags & GHOST_SPM_HOST)) {
		DEBUG_LOG(1,"Creating matrix on OpenCL device");
		BJDS(mat)->clmat = (CL_BJDS_TYPE *)allocateMemory(sizeof(CL_BJDS_TYPE),"CL_CRS");
		BJDS(mat)->clmat->rowLen = CL_allocDeviceMemory((BJDS(mat)->nrows)*sizeof(ghost_cl_midx_t));
		BJDS(mat)->clmat->col = CL_allocDeviceMemory((BJDS(mat)->nEnts)*sizeof(ghost_cl_midx_t));
		BJDS(mat)->clmat->val = CL_allocDeviceMemory((BJDS(mat)->nEnts)*sizeof(ghost_cl_mdat_t));
		BJDS(mat)->clmat->chunkStart = CL_allocDeviceMemory((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_cl_mnnz_t));
		BJDS(mat)->clmat->chunkLen = CL_allocDeviceMemory((BJDS(mat)->nrowsPadded/BJDS_LEN)*sizeof(ghost_cl_midx_t));
	
		BJDS(mat)->clmat->nrows = BJDS(mat)->nrows;
		BJDS(mat)->clmat->nrowsPadded = BJDS(mat)->nrowsPadded;
		CL_copyHostToDevice(BJDS(mat)->clmat->rowLen, BJDS(mat)->rowLen, BJDS(mat)->nrows*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(BJDS(mat)->clmat->col, BJDS(mat)->col, BJDS(mat)->nEnts*sizeof(ghost_cl_midx_t));
		CL_copyHostToDevice(BJDS(mat)->clmat->val, BJDS(mat)->val, BJDS(mat)->nEnts*sizeof(ghost_cl_mdat_t));
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


	DEBUG_LOG(1,"Successfully created BJDS");



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
	//	sse_kernel_0_intr(lhs, BJDS(mat), rhs, options);	
	ghost_midx_t c,j,i;
	ghost_vdat_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for (c=0; c<BJDS(mat)->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])/BJDS_LEN; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += BJDS(mat)->val[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] * rhs->val[BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i]];
			}

		}
		for (i=0; i<BJDS_LEN; i++)
		{
			if (options & GHOST_OPTION_AXPY)
				lhs->val[c*BJDS_LEN+i] += tmp[i];
			else
				lhs->val[c*BJDS_LEN+i] = tmp[i];

		}
	}
}

#ifdef SSE
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
		if (options & GHOST_OPTION_AXPY) {
			_mm_store_pd(&lhs->val[c*BJDS_LEN],_mm_add_pd(tmp,_mm_load_pd(&lhs->val[c*BJDS_LEN])));
		} else {
			_mm_stream_pd(&lhs->val[c*BJDS_LEN],tmp);
		}
	}


}
#endif

#ifdef AVX
static void BJDS_kernel_AVX(ghost_mat_t *mat, ghost_vec_t* res, ghost_vec_t* invec, int spmvmOptions)
{
	ghost_midx_t c,j;
	ghost_mnnz_t offs;
	__m256d tmp;
	__m256d val;
	__m256d rhs;
	__m128d rhstmp;

#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs,rhstmp)
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
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm256_store_pd(&res->val[c*BJDS_LEN],_mm256_add_pd(tmp,_mm256_load_pd(&res->val[c*BJDS_LEN])));
		} else {
			_mm256_stream_pd(&res->val[c*BJDS_LEN],tmp);
		}
	}
}
#endif

#ifdef MIC
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
		if (spmvmOptions & GHOST_OPTION_AXPY) {
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

#pragma omp parallel for schedule(runtime) private(j,tmp1,tmp2,val,rhs,idx,offs)
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
			tmp1 = _mm512_add_pd(tmp1,_mm512_mul_pd(val,rhs));

			offs += 8;

			val = _mm512_load_pd(&BJDS(mat)->val[offs]);
			idx = _mm512_permute4f128_epi32(idx,_MM_PERM_BADC);
			rhs = _mm512_i32logather_pd(idx,invec->val,8);
			tmp2 = _mm512_add_pd(tmp2,_mm512_mul_pd(val,rhs));

			offs += 8;
		}
		if (spmvmOptions & GHOST_OPTION_AXPY) {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],_mm512_add_pd(tmp1,_mm512_load_pd(&res->val[c*BJDS_LEN])));
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN+8],_mm512_add_pd(tmp2,_mm512_load_pd(&res->val[c*BJDS_LEN+8])));
		} else {
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN],tmp1);
			_mm512_storenrngo_pd(&res->val[c*BJDS_LEN+8],tmp2);
		}
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
