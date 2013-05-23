#include <ghost_util.h>
#include <ghost.h>
#include <ghost_mat.h>
#include <bjds.h>
#include <stdio.h>
#include <crs.h>
#include "ghost_complex.h"

#define CHOOSE_KERNEL(kernel,dt1,dt2,ch,mat,lhs,rhs,options) \
	switch(ch) { \
		case 1: \
				return BJDS_kernel_plain_tmpl< dt1, dt2, 1 >(mat,lhs,rhs,options); \
		break; \
		case 2: \
				return BJDS_kernel_plain_tmpl< dt1, dt2, 2 >(mat,lhs,rhs,options); \
		break; \
		case 4: \
				return BJDS_kernel_plain_tmpl< dt1, dt2, 4 >(mat,lhs,rhs,options); \
		break; \
		case 8: \
				return BJDS_kernel_plain_tmpl< dt1, dt2, 8 >(mat,lhs,rhs,options); \
		break; \
		case 16: \
				 return BJDS_kernel_plain_tmpl< dt1, dt2, 16 >(mat,lhs,rhs,options); \
		break; \
		case 32: \
				 return BJDS_kernel_plain_tmpl< dt1, dt2, 32 >(mat,lhs,rhs,options); \
		break; \
		case 64: \
				 return BJDS_kernel_plain_tmpl< dt1, dt2, 64 >(mat,lhs,rhs,options); \
		break; \
		case 256: \
				  return BJDS_kernel_plain_tmpl< dt1, dt2, 256 >(mat,lhs,rhs,options); \
		break; \
		default: \
				 return BJDS_kernel_plain_ELLPACK_tmpl< dt1, dt2 >(mat,lhs,rhs,options); \
		break; \
	}

using namespace std;

template<typename m_t, typename v_t, int chunkHeight> void BJDS_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
	DEBUG_LOG(2,"In plain BJDS kernel");
	v_t *rhsd = (v_t *)(rhs->val);
	v_t *lhsd = (v_t *)(lhs->val);
	ghost_midx_t i,j,c;
	v_t tmp[chunkHeight];
	BJDS_TYPE *bjds = (BJDS_TYPE *)(mat->data);


#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for (c=0; c<bjds->nrowsPadded/chunkHeight; c++) 
	{ // loop over chunks
		for (i=0; i<chunkHeight; i++)
		{
			tmp[i] = (v_t)0;
		}

		for (j=0; j<(bjds->chunkStart[c+1]-bjds->chunkStart[c])/chunkHeight; j++) 
		{ // loop inside chunk
			for (i=0; i<chunkHeight; i++)
			{
				tmp[i] += (v_t)(((m_t*)(bjds->val))[bjds->chunkStart[c]+j*chunkHeight+i]) * 
					rhsd[bjds->col[bjds->chunkStart[c]+j*chunkHeight+i]];
			}
		}
		for (i=0; i<chunkHeight; i++)
		{
			if (c*chunkHeight+i < bjds->nrows) {
				if (options & GHOST_SPMVM_AXPY)
					lhsd[c*chunkHeight+i] += tmp[i];
				else
					lhsd[c*chunkHeight+i] = tmp[i];
			}

		}
	}
}

template<typename m_t, typename v_t> void BJDS_kernel_plain_ELLPACK_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
	DEBUG_LOG(2,"In plain ELLPACK (BJDS) kernel");
	v_t *rhsd = (v_t *)(rhs->val);
	v_t *lhsd = (v_t *)(lhs->val);
	ghost_midx_t i,j;
	v_t tmp;
	BJDS_TYPE *bjds = (BJDS_TYPE *)(mat->data);
	m_t *bjdsv = (m_t*)(bjds->val);


#pragma omp parallel for schedule(runtime) private(j,tmp)
	for (i=0; i<bjds->nrows; i++) 
	{
		tmp = (v_t)0;

		for (j=0; j<bjds->rowLen[i]; j++) 
		{ 
			tmp += (v_t)bjdsv[bjds->nrowsPadded*j+i] * 
				rhsd[bjds->col[bjds->nrowsPadded*j+i]];
		}
		if (options & GHOST_SPMVM_AXPY)
			lhsd[i] += tmp;
		else
			lhsd[i] = tmp;

	}
}

template <typename m_t> void BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{
	DEBUG_LOG(1,"Creating BJDS matrix");
	CR_TYPE *cr = (CR_TYPE*)crs;
	ghost_midx_t i,j,c;
	unsigned int flags = mat->traits->flags;

	ghost_midx_t *rowPerm = NULL;
	ghost_midx_t *invRowPerm = NULL;

	ghost_sorting_t* rowSort = NULL;


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
			//	if( rowSort[i].row >= cr->nrows ) DEBUG_LOG(0,"error: invalid row number %"PRmatIDX" in %"PRmatIDX,rowSort[i].row, i); 

			(invRowPerm)[i] = rowSort[i].row;
			(rowPerm)[rowSort[i].row] = i;
		}
	}


	BJDS(mat)->nrows = cr->nrows;
	BJDS(mat)->nnz = cr->nEnts;
	BJDS(mat)->nEnts = 0;
	BJDS(mat)->nrowsPadded = ghost_pad(BJDS(mat)->nrows,BJDS(mat)->chunkHeight);
	BJDS(mat)->chunkHeight = BJDS(mat)->nrowsPadded;

	ghost_midx_t nChunks = BJDS(mat)->nrowsPadded/BJDS(mat)->chunkHeight;
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

		if ((i+1)%BJDS(mat)->chunkHeight == 0) {
			chunkAvg /= (double)BJDS(mat)->chunkHeight;

			BJDS(mat)->nEnts += BJDS(mat)->chunkHeight*chunkLen;
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

	//BJDS(mat)->val = (ghost_dt *)allocateMemory(sizeof(ghost_dt)*BJDS(mat)->nEnts,"BJDS(mat)->val");
	BJDS(mat)->val = (char *)allocateMemory(ghost_sizeofDataType(mat->traits->datatype)*BJDS(mat)->nEnts,"BJDS(mat)->val");
	BJDS(mat)->col = (ghost_midx_t *)allocateMemory(sizeof(ghost_midx_t)*BJDS(mat)->nEnts,"BJDS(mat)->col");

	if (BJDS(mat)->chunkHeight < BJDS(mat)->nrowsPadded) 
	{ // BJDS NUMA initialization

#pragma omp parallel for schedule(runtime) private(j,i)
		for (c=0; c<BJDS(mat)->nrowsPadded/BJDS(mat)->chunkHeight; c++) 
		{ // loop over chunks

			DEBUG_LOG(2,"Doing BJDS NUMA first-touch initialization");
			for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])/BJDS(mat)->chunkHeight; j++)
			{
				for (i=0; i<BJDS(mat)->chunkHeight; i++)
				{
					((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = (m_t)0.;
					BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = 0;
				}
			}
		}
	} else 
	{ // ELLPACK NUMA
		DEBUG_LOG(2,"Doing ELLPACK NUMA first-touch initialization");

#pragma omp parallel for schedule(runtime) private(j)
		for (i=0; i<BJDS(mat)->nrows; i++) { 
			for (j=0; j<BJDS(mat)->chunkLen[0]; j++) {
				((m_t *)(BJDS(mat)->val))[BJDS(mat)->nrowsPadded*j+i] = (m_t)0.;
				BJDS(mat)->col[BJDS(mat)->nrowsPadded*j+i] = 0;
			}
		}
	}





	for (c=0; c<nChunks; c++) {

		for (j=0; j<BJDS(mat)->chunkLen[c]; j++) {

			for (i=0; i<BJDS(mat)->chunkHeight; i++) {
				ghost_midx_t row = c*BJDS(mat)->chunkHeight+i;

				if (j<BJDS(mat)->rowLen[row]) {
					if (flags & GHOST_SPM_SORTED) {
						((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = ((m_t *)(cr->val))[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = ((m_t *)(cr->val))[cr->rpt[row]+j];
						BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = (m_t)0.0;
					BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i] = 0;
				}
				//printf("%f ",((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS(mat)->chunkHeight+i]);
			}
		}
	}
	//	printf("\n");
	DEBUG_LOG(1,"Successfully created BJDS");
}

extern "C" void dd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,double,double,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void ds_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,double,float,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void dc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,double,ghost_complex<float>,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void dz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,double,ghost_complex<double>,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void sd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,float,double,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void ss_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,float,float,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void sc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,float,ghost_complex<float>,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void sz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,float,ghost_complex<double>,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void cd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,ghost_complex<float>,double,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void cs_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,ghost_complex<float>,float,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void cc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,ghost_complex<float>,ghost_complex<float>,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void cz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,ghost_complex<float>,ghost_complex<double>,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void zd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,ghost_complex<double>,double,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void zs_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,ghost_complex<double>,float,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void zc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,ghost_complex<double>,ghost_complex<float>,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void zz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(BJDS_kernel_plain_tmpl,ghost_complex<double>,ghost_complex<double>,BJDS(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void d_BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{ return BJDS_fromCRS< double >(mat,crs); }

extern "C" void s_BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{ return BJDS_fromCRS< float >(mat,crs); }

extern "C" void z_BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{ return BJDS_fromCRS< ghost_complex<double> >(mat,crs); }

extern "C" void c_BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{ return BJDS_fromCRS< ghost_complex<float> >(mat,crs); }
