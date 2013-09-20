#include <ghost.h>
#include <ghost_util.h>
#include <ghost_mat.h>
#include <sell.h>
#include <stdio.h>
#include <crs.h>
#include "ghost_complex.h"

#define CHOOSE_KERNEL(kernel,dt1,dt2,ch,mat,lhs,rhs,options) \
	switch(ch) { \
		case 1: \
				return SELL_kernel_plain_tmpl< dt1, dt2, 1 >(mat,lhs,rhs,options); \
		break; \
		case 2: \
				return SELL_kernel_plain_tmpl< dt1, dt2, 2 >(mat,lhs,rhs,options); \
		break; \
		case 4: \
				return SELL_kernel_plain_tmpl< dt1, dt2, 4 >(mat,lhs,rhs,options); \
		break; \
		case 8: \
				return SELL_kernel_plain_tmpl< dt1, dt2, 8 >(mat,lhs,rhs,options); \
		break; \
		case 16: \
				 return SELL_kernel_plain_tmpl< dt1, dt2, 16 >(mat,lhs,rhs,options); \
		break; \
		case 32: \
				 return SELL_kernel_plain_tmpl< dt1, dt2, 32 >(mat,lhs,rhs,options); \
		break; \
		case 64: \
				 return SELL_kernel_plain_tmpl< dt1, dt2, 64 >(mat,lhs,rhs,options); \
		break; \
		case 256: \
				  return SELL_kernel_plain_tmpl< dt1, dt2, 256 >(mat,lhs,rhs,options); \
		break; \
		default: \
				 return SELL_kernel_plain_ELLPACK_tmpl< dt1, dt2 >(mat,lhs,rhs,options); \
		break; \
	}

using namespace std;

template<typename m_t, typename v_t, int chunkHeight> void SELL_kernel_plain_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
	DEBUG_LOG(2,"In plain SELL kernel");
	v_t *rhsd = (v_t *)(rhs->val);
	v_t *lhsd = (v_t *)(lhs->val);
	ghost_midx_t i,j,c;
	v_t tmp[chunkHeight];
	SELL_TYPE *sell = (SELL_TYPE *)(mat->data);


	if (options & GHOST_SPMVM_APPLY_SHIFT) {
		m_t shift = *((m_t *)(mat->traits->shift));
		if (options & GHOST_SPMVM_APPLY_SCALE) {
			m_t scale = *((m_t *)(mat->traits->scale));
#pragma omp parallel for schedule(runtime) private(j,tmp,i)
			for (c=0; c<sell->nrowsPadded/chunkHeight; c++) 
			{ // loop over chunks
				for (i=0; i<chunkHeight; i++) {
					tmp[i] = (v_t)0;
				}

				for (j=0; j<(sell->chunkStart[c+1]-sell->chunkStart[c])/chunkHeight; j++) 
				{ // loop inside chunk
					for (i=0; i<chunkHeight; i++) {
						tmp[i] += (v_t)(scale*((((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]) + shift)) * 
							rhsd[sell->col[sell->chunkStart[c]+j*chunkHeight+i]];
					}
				}
				for (i=0; i<chunkHeight; i++) {
					if (c*chunkHeight+i < sell->nrows) {
						if (options & GHOST_SPMVM_AXPY)
							lhsd[c*chunkHeight+i] += tmp[i];
						else
							lhsd[c*chunkHeight+i] = tmp[i];
					}

				}
			}
		} else {
#pragma omp parallel for schedule(runtime) private(j,tmp,i)
			for (c=0; c<sell->nrowsPadded/chunkHeight; c++) 
			{ // loop over chunks
				for (i=0; i<chunkHeight; i++) {
					tmp[i] = (v_t)0;
				}

				for (j=0; j<(sell->chunkStart[c+1]-sell->chunkStart[c])/chunkHeight; j++) 
				{ // loop inside chunk
					for (i=0; i<chunkHeight; i++) {
						tmp[i] += (v_t)((((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]) + shift) * 
							rhsd[sell->col[sell->chunkStart[c]+j*chunkHeight+i]];
					}
				}
				for (i=0; i<chunkHeight; i++) {
					if (c*chunkHeight+i < sell->nrows) {
						if (options & GHOST_SPMVM_AXPY)
							lhsd[c*chunkHeight+i] += tmp[i];
						else
							lhsd[c*chunkHeight+i] = tmp[i];
					}

				}
			}
		}
	} else {
		if (options & GHOST_SPMVM_APPLY_SCALE) {
			m_t scale = *((m_t *)(mat->traits->scale));
#pragma omp parallel for schedule(runtime) private(j,tmp,i)
			for (c=0; c<sell->nrowsPadded/chunkHeight; c++) 
			{ // loop over chunks
				for (i=0; i<chunkHeight; i++) {
					tmp[i] = (v_t)0;
				}

				for (j=0; j<(sell->chunkStart[c+1]-sell->chunkStart[c])/chunkHeight; j++) 
				{ // loop inside chunk
					for (i=0; i<chunkHeight; i++) {
						tmp[i] += (v_t)(scale*((((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]))) * 
							rhsd[sell->col[sell->chunkStart[c]+j*chunkHeight+i]];
					}
				}
				for (i=0; i<chunkHeight; i++) {
					if (c*chunkHeight+i < sell->nrows) {
						if (options & GHOST_SPMVM_AXPY)
							lhsd[c*chunkHeight+i] += tmp[i];
						else
							lhsd[c*chunkHeight+i] = tmp[i];
					}

				}
			}
		} else {
#pragma omp parallel for schedule(runtime) private(j,tmp,i)
			for (c=0; c<sell->nrowsPadded/chunkHeight; c++) 
			{ // loop over chunks
				for (i=0; i<chunkHeight; i++) {
					tmp[i] = (v_t)0;
				}

				for (j=0; j<(sell->chunkStart[c+1]-sell->chunkStart[c])/chunkHeight; j++) 
				{ // loop inside chunk
					for (i=0; i<chunkHeight; i++) {
						tmp[i] += (v_t)(((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]) * 
							rhsd[sell->col[sell->chunkStart[c]+j*chunkHeight+i]];
					}
				}
				for (i=0; i<chunkHeight; i++) {
					if (c*chunkHeight+i < sell->nrows) {
						if (options & GHOST_SPMVM_AXPY)
							lhsd[c*chunkHeight+i] += tmp[i];
						else
							lhsd[c*chunkHeight+i] = tmp[i];
					}

				}
			}
		}

	}
}

template<typename m_t, typename v_t> void SELL_kernel_plain_ELLPACK_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
	DEBUG_LOG(2,"In plain ELLPACK (SELL) kernel");
	v_t *rhsd = (v_t *)(rhs->val);
	v_t *lhsd = (v_t *)(lhs->val);
	ghost_midx_t i,j;
	v_t tmp;
	SELL_TYPE *sell = (SELL_TYPE *)(mat->data);
	m_t *sellv = (m_t*)(sell->val);
	v_t shift, scale;
	if (options & GHOST_SPMVM_APPLY_SHIFT)
		shift = *((v_t *)(mat->traits->shift));
	if (options & GHOST_SPMVM_APPLY_SCALE)
		scale = *((v_t *)(mat->traits->scale));


#pragma omp parallel for schedule(runtime) private(j,tmp)
	for (i=0; i<sell->nrows; i++) 
	{
		tmp = (v_t)0;

		for (j=0; j<sell->rowLen[i]; j++) 
		{ 
			tmp += (v_t)sellv[sell->nrowsPadded*j+i] * 
				rhsd[sell->col[sell->nrowsPadded*j+i]];
		}
		if (options & GHOST_SPMVM_APPLY_SHIFT) {
			if (options & GHOST_SPMVM_APPLY_SCALE) {
				if (options & GHOST_SPMVM_AXPY) {
					lhsd[i] += scale*(tmp+shift*rhsd[i]);
				} else {
					lhsd[i] = scale*(tmp+shift*rhsd[i]);
				}
			} else {
				if (options & GHOST_SPMVM_AXPY) {
					lhsd[i] += (tmp+shift*rhsd[i]);
				} else {
					lhsd[i] = (tmp+shift*rhsd[i]);
				}
			}
		} else {
			if (options & GHOST_SPMVM_APPLY_SCALE) {
				if (options & GHOST_SPMVM_AXPY) {
					lhsd[i] += scale*(tmp);
				} else {
					lhsd[i] = scale*(tmp);
				}
			} else {
				if (options & GHOST_SPMVM_AXPY) {
					lhsd[i] += (tmp);
				} else {
					lhsd[i] = (tmp);
				}
			}

		}
	}

}

static int ghost_selectSellChunkHeight(int datatype) {
	int ch = 1;

	if (datatype & GHOST_BINCRS_DT_FLOAT)
		ch *= 2;

	if (datatype & GHOST_BINCRS_DT_REAL)
		ch *= 2;

#ifdef AVX
	ch *= 2;
#endif

#ifdef MIC
	ch *= 4;
#ifndef LONGIDX
	ch *= 2;
#endif
#endif

#if defined (OPENCL) || defined (CUDA)
	ch = 256;
#endif

	return ch;
}


template <typename m_t> void SELL_fromCRS(ghost_mat_t *mat, void *crs)
{
	DEBUG_LOG(1,"Creating SELL matrix");
	CR_TYPE *cr = (CR_TYPE*)crs;
	ghost_midx_t i,j,c;
	unsigned int flags = mat->traits->flags;

	ghost_midx_t *rowPerm = NULL;
	ghost_midx_t *invRowPerm = NULL;

	ghost_sorting_t* rowSort = NULL;
	mat->data = (SELL_TYPE *)ghost_malloc(sizeof(SELL_TYPE));

	SELL(mat)->nrows = cr->nrows;
	SELL(mat)->nnz = cr->nEnts;
	SELL(mat)->nEnts = 0;

	if (mat->traits->aux == NULL) {
		SELL(mat)->scope = 1;
		SELL(mat)->chunkHeight = ghost_selectSellChunkHeight(mat->traits->datatype);
		SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,SELL(mat)->chunkHeight);
	} else {
		SELL(mat)->scope = *(int *)(mat->traits->aux);
		if (SELL(mat)->scope == GHOST_SELL_SORT_GLOBALLY) {
			SELL(mat)->scope = cr->nrows;
		}

		if (mat->traits->nAux == 1 || ((int *)(mat->traits->aux))[1] == GHOST_SELL_CHUNKHEIGHT_AUTO) {
			SELL(mat)->chunkHeight = ghost_selectSellChunkHeight(mat->traits->datatype);
			SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,SELL(mat)->chunkHeight);
		} else {
			if (((int *)(mat->traits->aux))[1] == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
				SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,GHOST_PAD_MAX); // TODO padding anpassen an architektur
				SELL(mat)->chunkHeight = SELL(mat)->nrowsPadded;
			} else {
				SELL(mat)->chunkHeight = ((int *)(mat->traits->aux))[1];
				SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,SELL(mat)->chunkHeight);
			}
		}
	}
	mat->context->rowPerm = rowPerm;
	mat->context->invRowPerm = invRowPerm;
	if (mat->traits->flags & GHOST_SPM_SORTED) {
		rowPerm = (ghost_midx_t *)ghost_malloc(cr->nrows*sizeof(ghost_midx_t));
		invRowPerm = (ghost_midx_t *)ghost_malloc(cr->nrows*sizeof(ghost_midx_t));

		mat->context->rowPerm = rowPerm;
		mat->context->invRowPerm = invRowPerm;

		DEBUG_LOG(1,"Sorting matrix rows");

		/* get max number of entries in one row ###########################*/
		rowSort = (ghost_sorting_t*)ghost_malloc(cr->nrows * sizeof(ghost_sorting_t));

		for (c=0; c<cr->nrows/SELL(mat)->scope; c++)  
		{
			for( i = c*SELL(mat)->scope; i < (c+1)*SELL(mat)->scope; i++ ) 
			{
				rowSort[i].row = i;
				rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
			} 

			qsort( rowSort+c*SELL(mat)->scope, SELL(mat)->scope, sizeof( ghost_sorting_t  ), compareNZEPerRow );
		}
		for( i = c*SELL(mat)->scope; i < cr->nrows; i++ ) 
		{ // remainder
			rowSort[i].row = i;
			rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
		}
		qsort( rowSort+c*SELL(mat)->scope, cr->nrows-c*SELL(mat)->scope, sizeof( ghost_sorting_t  ), compareNZEPerRow );

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



	// aux[0] = SELL(mat)->scope
	// aux[1] = chunk height

	//	SELL(mat)->chunkHeight = SELL(mat)->nrowsPadded;

	ghost_midx_t nChunks = SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight;
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
	SELL(mat)->beta = 0.;

	// TODO CHECK FOR OVERFLOW

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

		if ((i+1)%SELL(mat)->chunkHeight == 0) {
			chunkAvg /= (double)SELL(mat)->chunkHeight;

			SELL(mat)->nEnts += SELL(mat)->chunkHeight*chunkLen;
			SELL(mat)->chunkStart[curChunk] = SELL(mat)->nEnts;
			SELL(mat)->chunkMin[curChunk-1] = chunkMin;
			SELL(mat)->chunkLen[curChunk-1] = chunkLen;

			chunkMin = cr->ncols;
			chunkLen = 0;
			chunkAvg = 0;
			curChunk++;
			chunkEnts = 0;
		}
	}
	SELL(mat)->beta = nnz*1.0/(double)SELL(mat)->nEnts;

	SELL(mat)->val = (char *)ghost_malloc_align(ghost_sizeofDataType(mat->traits->datatype)*(size_t)SELL(mat)->nEnts,GHOST_DATA_ALIGNMENT);
	SELL(mat)->col = (ghost_midx_t *)ghost_malloc_align(sizeof(ghost_midx_t)*(size_t)SELL(mat)->nEnts,GHOST_DATA_ALIGNMENT);

			DEBUG_LOG(2,"Doing SELL NUMA first-touch initialization");
	if (SELL(mat)->chunkHeight < SELL(mat)->nrowsPadded) 
	{ // SELL NUMA initialization

#pragma omp parallel for schedule(runtime) private(j,i)
		for (c=0; c<SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight; c++) 
		{ // loop over chunks

			for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])/SELL(mat)->chunkHeight; j++)
			{
				for (i=0; i<SELL(mat)->chunkHeight; i++)
				{
					((m_t *)(SELL(mat)->val))[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = (m_t)0.;
					SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = 0;
				}
			}
		}
	} else 
	{ // ELLPACK NUMA
		DEBUG_LOG(2,"Doing ELLPACK NUMA first-touch initialization");

#pragma omp parallel for schedule(runtime) private(j)
		for (i=0; i<SELL(mat)->nrowsPadded; i++) { 
			for (j=0; j<SELL(mat)->chunkLen[0]; j++) {
			//	printf("%p %p\n",&(((m_t *)(SELL(mat)->val))[SELL(mat)->nrowsPadded*j+i]),&(SELL(mat)->col[SELL(mat)->nrowsPadded*j+i]));
				((m_t *)(SELL(mat)->val))[SELL(mat)->nrowsPadded*j+i] = (m_t)0.;
				SELL(mat)->col[SELL(mat)->nrowsPadded*j+i] = 0;
			}
		}
	}


	DEBUG_LOG(2,"Copying CRS to SELL");
	for (c=0; c<nChunks; c++) {

		for (j=0; j<SELL(mat)->chunkLen[c]; j++) {

			for (i=0; i<SELL(mat)->chunkHeight; i++) {
				ghost_midx_t row = c*SELL(mat)->chunkHeight+i;

				if (j<SELL(mat)->rowLen[row]) {
					if (flags & GHOST_SPM_SORTED) {
						((m_t *)(SELL(mat)->val))[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = ((m_t *)(cr->val))[cr->rpt[(invRowPerm)[row]]+j];
						if (flags & GHOST_SPM_PERMUTECOLIDX)
							SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = (rowPerm)[cr->col[cr->rpt[(invRowPerm)[row]]+j]];
						else
							SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = cr->col[cr->rpt[(invRowPerm)[row]]+j];
					} else {
						((m_t *)(SELL(mat)->val))[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = ((m_t *)(cr->val))[cr->rpt[row]+j];
						SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = cr->col[cr->rpt[row]+j];
					}

				} else {
					((m_t *)(SELL(mat)->val))[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = (m_t)0.0;
					SELL(mat)->col[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i] = 0;
				}
				//printf("%f ",((m_t *)(SELL(mat)->val))[SELL(mat)->chunkStart[c]+j*SELL(mat)->chunkHeight+i]);
			}
		}
	}
	//	printf("\n");
	DEBUG_LOG(1,"Successfully created SELL");
}

extern "C" void dd_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,double,double,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void ds_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,double,float,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void dc_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,double,ghost_complex<float>,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void dz_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,double,ghost_complex<double>,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void sd_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,float,double,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void ss_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,float,float,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void sc_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,float,ghost_complex<float>,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void sz_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,float,ghost_complex<double>,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void cd_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,ghost_complex<float>,double,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void cs_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,ghost_complex<float>,float,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void cc_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,ghost_complex<float>,ghost_complex<float>,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void cz_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,ghost_complex<float>,ghost_complex<double>,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void zd_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,ghost_complex<double>,double,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void zs_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,ghost_complex<double>,float,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void zc_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,ghost_complex<double>,ghost_complex<float>,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void zz_SELL_kernel_plain(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{ CHOOSE_KERNEL(SELL_kernel_plain_tmpl,ghost_complex<double>,ghost_complex<double>,SELL(mat)->chunkHeight,mat,lhs,rhs,options); }

extern "C" void d_SELL_fromCRS(ghost_mat_t *mat, void *crs)
{ return SELL_fromCRS< double >(mat,crs); }

extern "C" void s_SELL_fromCRS(ghost_mat_t *mat, void *crs)
{ return SELL_fromCRS< float >(mat,crs); }

extern "C" void z_SELL_fromCRS(ghost_mat_t *mat, void *crs)
{ return SELL_fromCRS< ghost_complex<double> >(mat,crs); }

extern "C" void c_SELL_fromCRS(ghost_mat_t *mat, void *crs)
{ return SELL_fromCRS< ghost_complex<float> >(mat,crs); }
