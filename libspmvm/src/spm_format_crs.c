#include "spm_format_crs.h"
#include "matricks.h"
#include "spmvm_util.h"

static char * CRS_formatName();
static mat_idx_t CRS_rowLen (mat_idx_t i);
static mat_data_t CRS_entry (mat_idx_t i, mat_idx_t j);
static size_t CRS_byteSize (void);
static void CRS_kernel (ghost_vec_t *, ghost_vec_t *, int);

//static ghost_mat_t *this;
static CR_TYPE *thisCR;

void CRS_init(ghost_mat_t *mat)
{
//	this = mat;
	thisCR = (CR_TYPE *)(mat->data);

	mat->formatName = &CRS_formatName;
	mat->rowLen   = &CRS_rowLen;
	mat->entry    = &CRS_entry;
	mat->byteSize = &CRS_byteSize;
	mat->kernel   = &CRS_kernel;
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
