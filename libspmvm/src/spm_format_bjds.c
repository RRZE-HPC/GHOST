#include "spm_format_bjds.h"
#include "matricks.h"
#include "spmvm_util.h"
#include "kernel.h"

static void BJDS_printInfo();
static char * BJDS_formatName();
static mat_idx_t BJDS_rowLen (mat_idx_t i);
static mat_data_t BJDS_entry (mat_idx_t i, mat_idx_t j);
static size_t BJDS_byteSize (void);
static void BJDS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options);

static ghost_mat_t *thisMat;
static BJDS_TYPE *thisBJDS;

void BJDS_init(ghost_mat_t *mat)
{
	thisMat = mat;
	thisBJDS = (BJDS_TYPE *)(mat->data);

	mat->printInfo = &BJDS_printInfo;
	mat->formatName = &BJDS_formatName;
	mat->rowLen     = &BJDS_rowLen;
	mat->entry      = &BJDS_entry;
	mat->byteSize   = &BJDS_byteSize;
	mat->kernel     = &BJDS_kernel_plain;
}

static void BJDS_printInfo()
{
	SpMVM_printLine("Vector block size",NULL,"%d",BJDS_LEN);
}
					
static char * BJDS_formatName()
{
	return "BJDS";
}

static mat_idx_t BJDS_rowLen (mat_idx_t i)
{
	if (thisMat->trait.flags & GHOST_SPM_SORTED)
		i = thisMat->rowPerm[i];
	
	return thisBJDS->rowLen[i];
}

static mat_data_t BJDS_entry (mat_idx_t i, mat_idx_t j)
{
	mat_idx_t e;

	if (thisMat->trait.flags & GHOST_SPM_SORTED)
		i = thisMat->rowPerm[i];
	if (thisMat->trait.flags & GHOST_SPM_PERMUTECOLIDX)
		j = thisMat->rowPerm[j];
	
	for (e=thisBJDS->chunkStart[i/BJDS_LEN]+i%BJDS_LEN; 
			e<thisBJDS->chunkStart[i/BJDS_LEN+1]; 
			e+=BJDS_LEN) {
		if (thisBJDS->col[e] == j)
			return thisBJDS->val[e];
	}
	return 0.;
}

static size_t BJDS_byteSize (void)
{
	return (size_t)((thisBJDS->nrowsPadded/BJDS_LEN)*sizeof(mat_nnz_t) + 
			thisBJDS->nEnts*(sizeof(mat_idx_t)+sizeof(mat_data_t)));
}

static void BJDS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
//	sse_kernel_0_intr(lhs, thisBJDS, rhs, options);	
	mat_idx_t c,j,i;
	mat_data_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for (c=0; c<thisBJDS->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<(thisBJDS->chunkStart[c+1]-thisBJDS->chunkStart[c])/BJDS_LEN; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += thisBJDS->val[thisBJDS->chunkStart[c]+j*BJDS_LEN+i] * rhs->val[thisBJDS->col[thisBJDS->chunkStart[c]+j*BJDS_LEN+i]];
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
