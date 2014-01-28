#include "ghost/config.h"
#include "ghost/types.h"

#if GHOST_HAVE_MPI
#include <mpi.h> //mpi.h has to be included before stdio.h
#endif
#include <stdio.h>

#include "ghost/complex.h"
#include "ghost/util.h"
#include "ghost/crs.h"
#include "ghost/sell.h"
#include "ghost/vec.h"
#include "ghost/math.h"
#include "ghost/constants.h"
#include <iostream>
#include <cstdlib>
#include <map>
#include <iostream>

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
    SELL_TYPE *sell = (SELL_TYPE *)(mat->data);
    v_t *rhsv;
    v_t *lhsv;
    v_t *local_dot_product, *partsums;
    ghost_midx_t i,j,c;
    ghost_vidx_t v;
    int nthreads = 1;
    v_t tmp[chunkHeight];
    
    v_t shift, scale, beta;
    if (options & GHOST_SPMVM_APPLY_SHIFT)
        shift = *((v_t *)(mat->traits->shift));
    if (options & GHOST_SPMVM_APPLY_SCALE)
        scale = *((v_t *)(mat->traits->scale));
    if (options & GHOST_SPMVM_AXPBY)
        beta = *((v_t *)(mat->traits->beta));
    if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
        local_dot_product = ((v_t *)(lhs->traits->localdot));

#pragma omp parallel
        nthreads = ghost_ompGetNumThreads();

        partsums = (v_t *)ghost_malloc(16*lhs->traits->nvecs*nthreads*sizeof(v_t));

        for (i=0; i<16*lhs->traits->nvecs*nthreads; i++) {
            partsums[i] = 0.;
        }
    }

#pragma omp parallel private(c,j,tmp,i,v)
    {
        int tid = ghost_ompGetThreadNum();

#pragma omp for schedule(runtime) 
        for (c=0; c<mat->nrowsPadded/chunkHeight; c++) 
        { // loop over chunks
            for (v=0; v<MIN(lhs->traits->nvecs,rhs->traits->nvecs); v++)
            {
                rhsv = (v_t *)rhs->val[v];
                lhsv = (v_t *)lhs->val[v];

                for (i=0; i<chunkHeight; i++) {
                    tmp[i] = (v_t)0;
                }

                for (j=0; j<(sell->chunkStart[c+1]-sell->chunkStart[c])/chunkHeight; j++) 
                { // loop inside chunk
                    for (i=0; i<chunkHeight; i++) {
                   // INFO_LOG("%d: %f * %f",i,(v_t)(((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]), rhsv[sell->col[sell->chunkStart[c]+j*chunkHeight+i]]);
                        tmp[i] += (v_t)(((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]) * 
                            rhsv[sell->col[sell->chunkStart[c]+j*chunkHeight+i]];
                    }
                }
                for (i=0; i<chunkHeight; i++) {
                    if (c*chunkHeight+i < mat->nrows) {
                        if (options & GHOST_SPMVM_APPLY_SHIFT) {
                            tmp[i] = tmp[i]-shift*rhsv[c*chunkHeight+i];
                        }
                        if (options & GHOST_SPMVM_APPLY_SCALE) {
                            tmp[i] = tmp[i]*scale;
                        }
                        if (options & GHOST_SPMVM_AXPY) {
                            lhsv[c*chunkHeight+i] += tmp[i];
                        } else if (options & GHOST_SPMVM_AXPBY) {
                            lhsv[c*chunkHeight+i] = beta*lhsv[c*chunkHeight+i] + tmp[i];
                        } else {
                            lhsv[c*chunkHeight+i] = tmp[i];
                        }
                    
                        if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
                            partsums[(v+tid*lhs->traits->nvecs)*16 + 0] += conjugate(&lhsv[c*chunkHeight+i])*lhsv[c*chunkHeight+i];
                            partsums[(v+tid*lhs->traits->nvecs)*16 + 1] += conjugate(&lhsv[c*chunkHeight+i])*rhsv[c*chunkHeight+i];
                            partsums[(v+tid*lhs->traits->nvecs)*16 + 2] += conjugate(&rhsv[c*chunkHeight+i])*rhsv[c*chunkHeight+i];
                        }
                   }

                }

            }
        }
    }
    if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
        for (v=0; v<MIN(lhs->traits->nvecs,rhs->traits->nvecs); v++) {
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                       ] += partsums[(v+i*lhs->traits->nvecs)*16 + 0];
                local_dot_product[v +   lhs->traits->nvecs] += partsums[(v+i*lhs->traits->nvecs)*16 + 1];
                local_dot_product[v + 2*lhs->traits->nvecs] += partsums[(v+i*lhs->traits->nvecs)*16 + 2];
            }
        }
        free(partsums);
    }
}


template<typename m_t, typename v_t> void SELL_kernel_plain_ELLPACK_tmpl(ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options)
{
    DEBUG_LOG(2,"In plain ELLPACK (SELL) kernel");
    v_t *rhsv;
    v_t *lhsv;
    v_t *local_dot_product, *partsums;
    int nthreads = 1;
    ghost_midx_t i,j;
    ghost_vidx_t v;
    v_t tmp;
    SELL_TYPE *sell = (SELL_TYPE *)(mat->data);
    m_t *sellv = (m_t*)(sell->val);
    v_t shift, scale, beta;
    if (options & GHOST_SPMVM_APPLY_SHIFT)
        shift = *((v_t *)(mat->traits->shift));
    if (options & GHOST_SPMVM_APPLY_SCALE)
        scale = *((v_t *)(mat->traits->scale));
    if (options & GHOST_SPMVM_AXPBY)
        beta = *((v_t *)(mat->traits->beta));
    if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
        local_dot_product = ((v_t *)(lhs->traits->localdot));

#pragma omp parallel
        nthreads = ghost_ompGetNumThreads();

        partsums = (v_t *)ghost_malloc(3*lhs->traits->nvecs*nthreads*sizeof(v_t));

        for (i=0; i<3*lhs->traits->nvecs*nthreads; i++) {
            partsums[i] = 0.;
        }
    }


#pragma omp parallel private(i,j,tmp,v)
    {
        int tid = ghost_ompGetThreadNum();
#pragma omp for schedule(runtime)
        for (i=0; i<mat->nrows; i++) 
        {
            for (v=0; v<MIN(lhs->traits->nvecs,rhs->traits->nvecs); v++)
            {
                rhsv = (v_t *)rhs->val[v];
                lhsv = (v_t *)lhs->val[v];
                tmp = (v_t)0;

                for (j=0; j<sell->rowLen[i]; j++) 
                {
    //                INFO_LOG("%d: %f * %f",i,(v_t)sellv[mat->nrowsPadded*j+i], rhsv[sell->col[mat->nrowsPadded*j+i]]);
                    tmp += (v_t)sellv[mat->nrowsPadded*j+i] * rhsv[sell->col[mat->nrowsPadded*j+i]];
                }
                
                if (options & GHOST_SPMVM_APPLY_SHIFT) {
                    tmp = tmp-shift*rhsv[i];
                }
                if (options & GHOST_SPMVM_APPLY_SCALE) {
                    tmp = tmp*scale;
                }
                if (options & GHOST_SPMVM_AXPY) {
                    lhsv[i] += tmp;
                } else if (options & GHOST_SPMVM_AXPBY) {
                    lhsv[i] = beta*lhsv[i] + tmp;
                } else {
                    lhsv[i] = tmp;
                }
                if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
                    partsums[(v+tid*lhs->traits->nvecs)*16 + 0] += conjugate(&lhsv[i])*lhsv[i];
                    partsums[(v+tid*lhs->traits->nvecs)*16 + 1] += conjugate(&lhsv[i])*rhsv[i];
                    partsums[(v+tid*lhs->traits->nvecs)*16 + 2] += conjugate(&rhsv[i])*rhsv[i];
                }
            }
        }
    }
    if (options & GHOST_SPMVM_COMPUTE_LOCAL_DOTPRODUCT) {
        for (v=0; v<MIN(lhs->traits->nvecs,rhs->traits->nvecs); v++) {
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                       ] += partsums[(v+i*lhs->traits->nvecs)*16 + 0];
                local_dot_product[v +   lhs->traits->nvecs] += partsums[(v+i*lhs->traits->nvecs)*16 + 1];
                local_dot_product[v + 2*lhs->traits->nvecs] += partsums[(v+i*lhs->traits->nvecs)*16 + 2];
            }
        }
        free(partsums);
    }

}

static int compareNZEPerRow( const void* a, const void* b ) 
{
    /* comparison function for ghost_sorting_t; 
     * sorts rows with higher number of non-zero elements first */

    return  ((ghost_sorting_t*)b)->nEntsInRow - ((ghost_sorting_t*)a)->nEntsInRow;
}

template <typename m_t> void SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crsmat)
{
    DEBUG_LOG(1,"Creating SELL matrix");
    CR_TYPE *cr = (CR_TYPE*)(crsmat->data);
    ghost_midx_t i,j,c;
    unsigned int flags = mat->traits->flags;

    ghost_midx_t *rowPerm = NULL;
    ghost_midx_t *invRowPerm = NULL;

    ghost_sorting_t* rowSort = NULL;
    //mat->data = (SELL_TYPE *)ghost_malloc(sizeof(SELL_TYPE));
    mat->nnz = crsmat->nnz;

   /* mat->nrows = cr->nrows;
    mat->nEnts = 0;

    if (mat->traits->aux == NULL) {
        SELL(mat)->scope = 1;
        SELL(mat)->T = 1;
        SELL(mat)->chunkHeight = ghost_selectSellChunkHeight(mat->traits->datatype);
        mat->nrowsPadded = ghost_pad(mat->nrows,SELL(mat)->chunkHeight);
    } else {
        SELL(mat)->scope = *(int *)(mat->traits->aux);
        if (SELL(mat)->scope == GHOST_SELL_SORT_GLOBALLY) {
            SELL(mat)->scope = cr->nrows;
        }

        if (mat->traits->nAux == 1 || ((int *)(mat->traits->aux))[1] == GHOST_SELL_CHUNKHEIGHT_AUTO) {
            SELL(mat)->chunkHeight = ghost_selectSellChunkHeight(mat->traits->datatype);
            mat->nrowsPadded = ghost_pad(mat->nrows,SELL(mat)->chunkHeight);
        } else {
            if (((int *)(mat->traits->aux))[1] == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
                mat->nrowsPadded = ghost_pad(mat->nrows,GHOST_PAD_MAX); // TODO padding anpassen an architektur
                SELL(mat)->chunkHeight = mat->nrowsPadded;
            } else {
                SELL(mat)->chunkHeight = ((int *)(mat->traits->aux))[1];
                mat->nrowsPadded = ghost_pad(mat->nrows,SELL(mat)->chunkHeight);
            }
        }
        SELL(mat)->T = ((int *)(mat->traits->aux))[2];
    }*/
    mat->context->rowPerm = rowPerm;
    mat->context->invRowPerm = invRowPerm;
    if (mat->traits->flags & GHOST_SPM_SORTED) {
        rowPerm = (ghost_midx_t *)ghost_malloc(crsmat->nrows*sizeof(ghost_midx_t));
        invRowPerm = (ghost_midx_t *)ghost_malloc(crsmat->nrows*sizeof(ghost_midx_t));

        mat->context->rowPerm = rowPerm;
        mat->context->invRowPerm = invRowPerm;

        DEBUG_LOG(1,"Sorting matrix rows");

        /* get max number of entries in one row ###########################*/
        rowSort = (ghost_sorting_t*)ghost_malloc(crsmat->nrows * sizeof(ghost_sorting_t));

        for (c=0; c<crsmat->nrows/SELL(mat)->scope; c++)  
        {
            for( i = c*SELL(mat)->scope; i < (c+1)*SELL(mat)->scope; i++ ) 
            {
                rowSort[i].row = i;
                rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
            } 

            qsort( rowSort+c*SELL(mat)->scope, SELL(mat)->scope, sizeof( ghost_sorting_t  ), compareNZEPerRow );
        }
        for( i = c*SELL(mat)->scope; i < crsmat->nrows; i++ ) 
        { // remainder
            rowSort[i].row = i;
            rowSort[i].nEntsInRow = cr->rpt[i+1] - cr->rpt[i];
        }
        qsort( rowSort+c*SELL(mat)->scope, crsmat->nrows-c*SELL(mat)->scope, sizeof( ghost_sorting_t  ), compareNZEPerRow );

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
        for(i=0; i < crsmat->nrows; ++i) {
            /* invRowPerm maps an index in the permuted system to the original index,
             * rowPerm gets the original index and returns the corresponding permuted position.
             */
            //    if( rowSort[i].row >= cr->nrows ) DEBUG_LOG(0,"error: invalid row number %"PRmatIDX" in %"PRmatIDX,rowSort[i].row, i); 

            (invRowPerm)[i] = rowSort[i].row;
            (rowPerm)[rowSort[i].row] = i;
        }
    }



    // aux[0] = SELL(mat)->scope
    // aux[1] = chunk height

    //    SELL(mat)->chunkHeight = mat->nrowsPadded;

    ghost_midx_t nChunks = mat->nrowsPadded/SELL(mat)->chunkHeight;
    SELL(mat)->chunkStart = (ghost_mnnz_t *)ghost_malloc((nChunks+1)*sizeof(ghost_mnnz_t));
    SELL(mat)->chunkMin = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->chunkLen = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->chunkLenPadded = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->rowLen = (ghost_midx_t *)ghost_malloc((mat->nrowsPadded)*sizeof(ghost_midx_t));
    SELL(mat)->rowLenPadded = (ghost_midx_t *)ghost_malloc((mat->nrowsPadded)*sizeof(ghost_midx_t));
    SELL(mat)->chunkStart[0] = 0;
    SELL(mat)->maxRowLen = 0;

    ghost_midx_t chunkMin = crsmat->ncols;
    ghost_midx_t chunkLen = 0;
    ghost_midx_t chunkLenPadded = 0;
    ghost_midx_t chunkEnts = 0;
    ghost_mnnz_t nnz = 0;
    double chunkAvg = 0.;
    ghost_midx_t curChunk = 1;
    SELL(mat)->beta = 0.;

    // TODO CHECK FOR OVERFLOW

    std::map<int,int> rowlengths;

    for (i=0; i<mat->nrowsPadded; i++) {
        if (i<crsmat->nrows) {
            if (flags & GHOST_SPM_SORTED)
                SELL(mat)->rowLen[i] = rowSort[i].nEntsInRow;
            else
                SELL(mat)->rowLen[i] = cr->rpt[i+1]-cr->rpt[i];
        } else {
            SELL(mat)->rowLen[i] = 0;
        }
        
        nnz += SELL(mat)->rowLen[i];
        
        rowlengths[SELL(mat)->rowLen[i]]++;
        
        SELL(mat)->rowLenPadded[i] = ghost_pad(SELL(mat)->rowLen[i],SELL(mat)->T);

        chunkMin = SELL(mat)->rowLen[i]<chunkMin?SELL(mat)->rowLenPadded[i]:chunkMin;
        chunkLen = SELL(mat)->rowLen[i]>chunkLen?SELL(mat)->rowLen[i]:chunkLen;
        chunkLenPadded = SELL(mat)->rowLenPadded[i]>chunkLenPadded?SELL(mat)->rowLenPadded[i]:chunkLenPadded;
        chunkAvg += SELL(mat)->rowLenPadded[i];
        chunkEnts += SELL(mat)->rowLenPadded[i];

        if ((i+1)%SELL(mat)->chunkHeight == 0) {
            chunkAvg /= (double)SELL(mat)->chunkHeight;

            mat->nEnts += SELL(mat)->chunkHeight*chunkLenPadded;
            SELL(mat)->chunkStart[curChunk] = mat->nEnts;
            SELL(mat)->chunkMin[curChunk-1] = chunkMin;
            SELL(mat)->chunkLen[curChunk-1] = chunkLen;
            SELL(mat)->chunkLenPadded[curChunk-1] = chunkLenPadded;

            chunkMin = crsmat->ncols;
            chunkLen = 0;
            chunkLenPadded = 0;
            chunkAvg = 0;
            curChunk++;
            chunkEnts = 0;
        }

        SELL(mat)->maxRowLen = MAX(SELL(mat)->maxRowLen,SELL(mat)->rowLenPadded[i]);
    }
    SELL(mat)->beta = nnz*1.0/(double)mat->nEnts;

    double avgRowlen = nnz*1.0/(double)mat->nrows;

    rowlengths.erase(0); // erase padded rows
    SELL(mat)->variance = 0.;
    SELL(mat)->deviation = 0.;
    for (std::map<int,int>::const_iterator it = rowlengths.begin(); it != rowlengths.end(); it++) {
        SELL(mat)->variance += (it->first-avgRowlen)*(it->first-avgRowlen)*it->second;
    }
    SELL(mat)->variance /= mat->nrows;
    SELL(mat)->deviation = sqrt(SELL(mat)->variance);
    SELL(mat)->cv = SELL(mat)->deviation*1./(nnz*1.0/(double)mat->nrows);

    if (rowlengths.size() > 0) {
        SELL(mat)->nMaxRows = rowlengths.rbegin()->second;
    }

    SELL(mat)->val = (char *)ghost_malloc_align(ghost_sizeofDataType(mat->traits->datatype)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT);
    SELL(mat)->col = (ghost_midx_t *)ghost_malloc_align(sizeof(ghost_midx_t)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT);

    DEBUG_LOG(2,"Doing SELL NUMA first-touch initialization");
    if (SELL(mat)->chunkHeight < mat->nrowsPadded) 
    { // SELL NUMA initialization

#pragma omp parallel for schedule(runtime) private(j,i)
        for (c=0; c<nChunks; c++) 
        { // loop over chunks
            for (j=0; j<SELL(mat)->chunkLenPadded[c]; j++)
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

#pragma omp parallel for schedule(runtime) private(j,i)
        for (i=0; i<mat->nrowsPadded; i++) 
        { 
            for (j=0; j<SELL(mat)->chunkLenPadded[0]; j++) 
            {
                    ((m_t *)(SELL(mat)->val))[mat->nrowsPadded*j+i] = (m_t)0.;
                    SELL(mat)->col[mat->nrowsPadded*j+i] = 0;
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
                        if (invRowPerm == NULL) {
                            ABORT("The matris is sorted but the permutation vector is NULL");
                        }
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
            }
        }
    }
    DEBUG_LOG(1,"Successfully created SELL");
}

template <typename m_t> static const char * SELL_stringify(ghost_mat_t *mat, int dense)
{
    ghost_midx_t chunk,i,j,row=0;
    m_t *val = (m_t *)SELL(mat)->val;

    stringstream buffer;

    buffer << "---" << endl;
    for (chunk = 0; chunk < mat->nrowsPadded/SELL(mat)->chunkHeight; chunk++) {
        for (i=0; i<SELL(mat)->chunkHeight && row<mat->nrows; i++, row++) {
            for (j=0; j<(dense?mat->ncols:SELL(mat)->chunkLen[chunk]); j++) {
                ghost_mnnz_t idx = SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i;
                if (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX) {
                    if (SELL(mat)->col[idx] < mat->nrows) {
                        buffer << val[idx] << " (o " << mat->context->invRowPerm[SELL(mat)->col[idx]] << "|p " << SELL(mat)->col[idx] << ")" << "\t";
                    } else {
                        buffer << val[idx] << " (p " << SELL(mat)->col[idx] << "|p " << SELL(mat)->col[idx] << ")" << "\t";
                    }

                } else {
                    buffer << val[idx] << " (" << SELL(mat)->col[idx] << ")" << "\t";
                }

            }
            buffer << endl;
        }
        buffer << "---" << endl;
    }

    return buffer.str().c_str();
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

extern "C" void d_SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs)
{ return SELL_fromCRS< double >(mat,crs); }

extern "C" void s_SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs)
{ return SELL_fromCRS< float >(mat,crs); }

extern "C" void z_SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs)
{ return SELL_fromCRS< ghost_complex<double> >(mat,crs); }

extern "C" void c_SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs)
{ return SELL_fromCRS< ghost_complex<float> >(mat,crs); }

extern "C" const char * d_SELL_stringify(ghost_mat_t *mat, int dense)
{ return SELL_stringify< double >(mat, dense); }

extern "C" const char * s_SELL_stringify(ghost_mat_t *mat, int dense)
{ return SELL_stringify< float >(mat, dense); }

extern "C" const char * z_SELL_stringify(ghost_mat_t *mat, int dense)
{ return SELL_stringify< ghost_complex<double> >(mat, dense); }

extern "C" const char * c_SELL_stringify(ghost_mat_t *mat, int dense)
{ return SELL_stringify< ghost_complex<float> >(mat, dense); }
