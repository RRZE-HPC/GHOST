#include <ghost.h>
#include <ghost_util.h>
#include <ghost_mat.h>
#include <bjds.h>
#include <stdio.h>
#include <crs.h>
#include "ghost_complex.h"

using namespace std;

template<typename m_t, typename v_t> void BJDS_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
	v_t *rhsd = (v_t *)(rhs->val);
	v_t *lhsd = (v_t *)(lhs->val);
	ghost_midx_t i,j,c;
	v_t tmp[BJDS_LEN];
	BJDS_TYPE *bjds = (BJDS_TYPE *)(mat->data);

#pragma omp parallel for schedule(runtime) private(j,tmp,i)
	for (c=0; c<bjds->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks
		for (i=0; i<BJDS_LEN; i++)
		{
			tmp[i] = (v_t)0;
		}

		for (j=0; j<(bjds->chunkStart[c+1]-bjds->chunkStart[c])/BJDS_LEN; j++) 
		{ // loop inside chunk
			for (i=0; i<BJDS_LEN; i++)
			{
				tmp[i] += (v_t)(bjds->val[(bjds->chunkStart[c]+j*BJDS_LEN+i)*sizeof(m_t)]) * 
					rhsd[bjds->col[bjds->chunkStart[c]+j*BJDS_LEN+i]];
			}
		}
		for (i=0; i<BJDS_LEN; i++)
		{
			if (c*BJDS_LEN+i < bjds->nrows) {
				if (options & GHOST_SPMVM_AXPY)
					lhsd[c*BJDS_LEN+i] += tmp[i];
				else
					lhsd[c*BJDS_LEN+i] = tmp[i];
			}

		}
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


	mat->data = (BJDS_TYPE *)allocateMemory(sizeof(BJDS_TYPE),"BJDS(mat)");
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

	//BJDS(mat)->val = (ghost_dt *)allocateMemory(sizeof(ghost_dt)*BJDS(mat)->nEnts,"BJDS(mat)->val");
	BJDS(mat)->val = (char *)allocateMemory(ghost_sizeofDataType(mat->traits->datatype)*BJDS(mat)->nEnts,"BJDS(mat)->val");
	BJDS(mat)->col = (ghost_midx_t *)allocateMemory(sizeof(ghost_midx_t)*BJDS(mat)->nEnts,"BJDS(mat)->col");

#pragma omp parallel for schedule(runtime) private(j,i)
	for (c=0; c<BJDS(mat)->nrowsPadded/BJDS_LEN; c++) 
	{ // loop over chunks

		for (j=0; j<(BJDS(mat)->chunkStart[c+1]-BJDS(mat)->chunkStart[c])/BJDS_LEN; j++)
		{
			for (i=0; i<BJDS_LEN; i++)
			{
				((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = (m_t)0.;
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
						((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = ((m_t *)(cr->val))[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = ((m_t *)(cr->val))[cr->rpt[row]+j];
						BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					((m_t *)(BJDS(mat)->val))[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = (m_t)0.0;
					BJDS(mat)->col[BJDS(mat)->chunkStart[c]+j*BJDS_LEN+i] = 0;
				}
			}
		}
	}
	DEBUG_LOG(1,"Successfully created BJDS");
}


extern "C" void dd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< double,double >(mat,lhs,rhs,options); }

extern "C" void ds_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< double,float >(mat,lhs,rhs,options); }

extern "C" void dc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< double,ghost_complex< float > >(mat,lhs,rhs,options); }

extern "C" void dz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< double,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" void sd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< float,double >(mat,lhs,rhs,options); }

extern "C" void ss_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< float,float >(mat,lhs,rhs,options); }

extern "C" void sc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< float,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" void sz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< float,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" void cd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< ghost_complex<float>,double >(mat,lhs,rhs,options); }

extern "C" void cs_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< ghost_complex<float>,float >(mat,lhs,rhs,options); }

extern "C" void cc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< ghost_complex<float>,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" void cz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< ghost_complex<float>,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" void zd_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< ghost_complex<double>,double >(mat,lhs,rhs,options); }

extern "C" void zs_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< ghost_complex<double>,float >(mat,lhs,rhs,options); }

extern "C" void zc_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< ghost_complex<double>,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" void zz_BJDS_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ return BJDS_kernel_plain_tmpl< ghost_complex<double>,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" void d_BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{ return BJDS_fromCRS< double >(mat,crs); }

extern "C" void s_BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{ return BJDS_fromCRS< float >(mat,crs); }

extern "C" void z_BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{ return BJDS_fromCRS< ghost_complex<double> >(mat,crs); }

extern "C" void c_BJDS_fromCRS(ghost_mat_t *mat, void *crs)
{ return BJDS_fromCRS< ghost_complex<float> >(mat,crs); }
