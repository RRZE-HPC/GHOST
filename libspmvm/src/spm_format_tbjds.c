#include "spm_format_tbjds.h"
#include "matricks.h"
#include "spmvm_util.h"
#include "kernel.h"

static char * TBJDS_formatName();
static mat_idx_t TBJDS_rowLen (mat_idx_t i);
static mat_data_t TBJDS_entry (mat_idx_t i, mat_idx_t j);
static size_t TBJDS_byteSize (void);
static void TBJDS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options);

static ghost_mat_t *thisMat;
static BJDS_TYPE *thisTBJDS;

void TBJDS_init(ghost_mat_t *mat)
{
	DEBUG_LOG(1,"Setting functions for TBJDS matrix");
	thisMat = mat;
	thisTBJDS = (BJDS_TYPE *)(mat->data);

	mat->formatName = &TBJDS_formatName;
	mat->rowLen   = &TBJDS_rowLen;
	mat->entry    = &TBJDS_entry;
	mat->byteSize = &TBJDS_byteSize;
	mat->kernel   = &TBJDS_kernel_plain;
}

static char * TBJDS_formatName()
{
	return "TBJDS";
}

static mat_idx_t TBJDS_rowLen (mat_idx_t i)
{
	if (thisMat->trait.flags & GHOST_SPM_SORTED)
		i = thisMat->rowPerm[i];
	
	return thisTBJDS->rowLen[i];
}

static mat_data_t TBJDS_entry (mat_idx_t i, mat_idx_t j)
{
	mat_idx_t e;
	if (thisMat->trait.flags & GHOST_SPM_SORTED)
		i = thisMat->rowPerm[i];
	if (thisMat->trait.flags & GHOST_SPM_PERMUTECOLIDX)
		j = thisMat->rowPerm[j];


	for (e=thisTBJDS->chunkStart[i/BJDS_LEN]+i%BJDS_LEN; 
			e<thisTBJDS->chunkStart[i/BJDS_LEN+1]; 
			e+=BJDS_LEN) {
		if (thisTBJDS->col[e] == j)
			return thisTBJDS->val[e];
	}
	return 0.;
}

static size_t TBJDS_byteSize (void)
{
	return (size_t)((thisTBJDS->nrowsPadded/BJDS_LEN)*sizeof(mat_nnz_t) + 
			thisTBJDS->nEnts*(sizeof(mat_idx_t)+sizeof(mat_data_t)));
}

static void TBJDS_kernel_plain (ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
	DEBUG_LOG(2,"Calling plain TBJDS kernel");
	mat_idx_t c,j,i;
	mat_nnz_t offs;
	mat_data_t tmp[BJDS_LEN]; 

#pragma omp parallel for schedule(runtime) private(j,tmp,i,offs)
	for (c=0; c<thisTBJDS->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		offs = thisTBJDS->chunkStart[c];
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = 0;
		}

		for (j=0; j<thisTBJDS->chunkMin[c]; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += thisTBJDS->val[offs] * rhs->val[thisTBJDS->col[offs++]];
			}

		}
		for (i=0; i<BJDS_LEN; i++)
		{
			for (j=thisTBJDS->chunkMin[c]; j<thisTBJDS->rowLen[c*BJDS_LEN+i]; j++)
			{
				tmp[i] += thisTBJDS->val[offs] * rhs->val[thisTBJDS->col[offs++]];
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
