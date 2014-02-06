#include "ghost/sell.h"
#include "ghost/core.h"
#include "ghost/crs.h"
#include "ghost/util.h"
#include "ghost/affinity.h"
#include "ghost/mat.h"
#include "ghost/constants.h"
#include "ghost/context.h"
#include "ghost/io.h"
#include "ghost/log.h"

#include <libgen.h>
#include <string.h>
#include <stdlib.h>

#if GHOST_HAVE_OPENMP
#include <omp.h>
#endif

#ifdef GHOST_HAVE_CUDA
//#include "private/sell_cukernel.h"
#endif

#if defined(SSE) || defined(AVX) || defined(MIC)
#include <immintrin.h>
#endif
#if defined(VSX)
#include <altivec.h>
#endif

ghost_error_t (*SELL_kernels_SSE[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_SSE,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

ghost_error_t (*SELL_kernels_AVX[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_AVX,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

ghost_error_t (*SELL_kernels_AVX_32[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_AVX_32_rich,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

ghost_error_t (*SELL_kernels_MIC_16[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_MIC_16,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

ghost_error_t (*SELL_kernels_MIC_32[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_MIC_32,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

ghost_error_t (*SELL_kernels_plain[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t) = 
{{&ss_SELL_kernel_plain,&sd_SELL_kernel_plain,&sc_SELL_kernel_plain,&sz_SELL_kernel_plain},
    {&ds_SELL_kernel_plain,&dd_SELL_kernel_plain,&dc_SELL_kernel_plain,&dz_SELL_kernel_plain},
    {&cs_SELL_kernel_plain,&cd_SELL_kernel_plain,&cc_SELL_kernel_plain,&cz_SELL_kernel_plain},
    {&zs_SELL_kernel_plain,&zd_SELL_kernel_plain,&zc_SELL_kernel_plain,&zz_SELL_kernel_plain}};

#ifdef GHOST_HAVE_CUDA
ghost_error_t (*SELL_kernels_CU[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t) = 
{{&ss_SELL_kernel_CU,&sd_SELL_kernel_CU,&sc_SELL_kernel_CU,&sz_SELL_kernel_CU},
    {&ds_SELL_kernel_CU,&dd_SELL_kernel_CU,&dc_SELL_kernel_CU,&dz_SELL_kernel_CU},
    {&cs_SELL_kernel_CU,&cd_SELL_kernel_CU,&cc_SELL_kernel_CU,&cz_SELL_kernel_CU},
    {&zs_SELL_kernel_CU,&zd_SELL_kernel_CU,&zc_SELL_kernel_CU,&zz_SELL_kernel_CU}};
#endif

void (*SELL_fromCRS_funcs[4]) (ghost_mat_t *, ghost_mat_t *) = 
{&s_SELL_fromCRS, &d_SELL_fromCRS, &c_SELL_fromCRS, &z_SELL_fromCRS}; 

const char * (*SELL_stringify_funcs[4]) (ghost_mat_t *, int) = 
{&s_SELL_stringify, &d_SELL_stringify, &c_SELL_stringify, &z_SELL_stringify}; 

static void SELL_printInfo(ghost_mat_t *mat);
static const char * SELL_formatName(ghost_mat_t *mat);
static ghost_midx_t SELL_rowLen (ghost_mat_t *mat, ghost_midx_t i);
static size_t SELL_byteSize (ghost_mat_t *mat);
static void SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs);
static const char * SELL_stringify(ghost_mat_t *mat, int dense);
static ghost_error_t SELL_split(ghost_mat_t *mat);
static ghost_error_t SELL_permute(ghost_mat_t *, ghost_midx_t *, ghost_midx_t *);
static ghost_error_t SELL_upload(ghost_mat_t *mat);
static ghost_error_t SELL_fromBin(ghost_mat_t *mat, char *);
static ghost_error_t SELL_fromRowFunc(ghost_mat_t *mat, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, ghost_spmFromRowFunc_flags_t flags);
static void SELL_free(ghost_mat_t *mat);
static ghost_error_t SELL_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
static int ghost_selectSellChunkHeight(int datatype);
#ifdef GHOST_HAVE_CUDA
static void SELL_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options);
#endif

ghost_error_t ghost_SELL_init(ghost_context_t *ctx, ghost_mtraits_t * traits, ghost_mat_t **mat)
{
    *mat = (ghost_mat_t *)ghost_malloc(sizeof(ghost_mat_t));
    (*mat)->data = (SELL_TYPE *)ghost_malloc(sizeof(SELL_TYPE));
    (*mat)->context = ctx;
    (*mat)->traits = traits;
    DEBUG_LOG(1,"Setting functions for SELL matrix");
    if (!((*mat)->traits->flags & (GHOST_SPM_HOST | GHOST_SPM_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting matrix placement");
        ghost_type_t ghost_type;
        GHOST_CALL_RETURN(ghost_getType(&ghost_type));
        if (ghost_type == GHOST_TYPE_CUDAMGMT) {
            (*mat)->traits->flags |= GHOST_SPM_DEVICE;
        } else {
            (*mat)->traits->flags |= GHOST_SPM_HOST;
        }
    }
    //TODO is it reasonable that a matrix has HOST&DEVICE?

    (*mat)->upload = &SELL_upload;
    (*mat)->fromFile = &SELL_fromBin;
    (*mat)->fromRowFunc = &SELL_fromRowFunc;
    (*mat)->printInfo = &SELL_printInfo;
    (*mat)->formatName = &SELL_formatName;
    (*mat)->rowLen     = &SELL_rowLen;
    (*mat)->byteSize   = &SELL_byteSize;
    (*mat)->spmv     = &SELL_kernel_plain;
    (*mat)->fromCRS    = &SELL_fromCRS;
    (*mat)->stringify    = &SELL_stringify;
    (*mat)->split = &SELL_split;
    (*mat)->permute = &SELL_permute;
#ifdef VSX_INTR
    (*mat)->kernel = &SELL_kernel_VSX;
#endif
#if GHOST_HAVE_CUDA
    if (ghost_type == GHOST_TYPE_CUDAMGMT) {
        (*mat)->spmv   = &SELL_kernel_CU;
    }
#endif
    (*mat)->destroy  = &SELL_free;

    (*mat)->localPart = NULL;
    (*mat)->remotePart = NULL;
    (*mat)->name = NULL;

    int me;
    GHOST_CALL_RETURN(ghost_getRank((*mat)->context->mpicomm,&me));

    (*mat)->nrows = (*mat)->context->lnrows[me];
    (*mat)->ncols = (*mat)->context->gncols;

    if ((*mat)->traits->aux == NULL) {
        SELL((*mat))->scope = 1;
        SELL((*mat))->T = 1;
        SELL((*mat))->chunkHeight = ghost_selectSellChunkHeight((*mat)->traits->datatype);
        (*mat)->nrowsPadded = ghost_pad((*mat)->nrows,SELL((*mat))->chunkHeight);
    } else {
        SELL((*mat))->scope = *(int *)((*mat)->traits->aux);
        if (SELL((*mat))->scope == GHOST_SELL_SORT_GLOBALLY) {
            SELL((*mat))->scope = (*mat)->context->lnrows[me];
        }

        if ((*mat)->traits->nAux == 1 || ((int *)((*mat)->traits->aux))[1] == GHOST_SELL_CHUNKHEIGHT_AUTO) {
            SELL((*mat))->chunkHeight = ghost_selectSellChunkHeight((*mat)->traits->datatype);
            (*mat)->nrowsPadded = ghost_pad((*mat)->nrows,SELL((*mat))->chunkHeight);
        } else {
            if (((int *)((*mat)->traits->aux))[1] == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
                (*mat)->nrowsPadded = ghost_pad((*mat)->nrows,GHOST_PAD_MAX); // TODO padding anpassen an architektur
                SELL((*mat))->chunkHeight = (*mat)->nrowsPadded;
            } else {
                SELL((*mat))->chunkHeight = ((int *)((*mat)->traits->aux))[1];
                (*mat)->nrowsPadded = ghost_pad((*mat)->nrows,SELL((*mat))->chunkHeight);
            }
        }
        SELL((*mat))->T = ((int *)((*mat)->traits->aux))[2];
    }
    (*mat)->nrowsPadded = ghost_pad((*mat)->nrows,SELL((*mat))->chunkHeight);;

    return GHOST_SUCCESS;
}

static ghost_error_t SELL_permute(ghost_mat_t *mat , ghost_midx_t *perm, ghost_midx_t *invPerm)
{
    UNUSED(mat);
    UNUSED(perm);
    UNUSED(invPerm);
    return GHOST_ERR_NOT_IMPLEMENTED;

}
static void SELL_printInfo(ghost_mat_t *mat)
{
    ghost_printLine("Max row length (# rows)",NULL,"%d (%d)",SELL(mat)->maxRowLen,SELL(mat)->nMaxRows);
    ghost_printLine("Chunk height (C)",NULL,"%d",SELL(mat)->chunkHeight);
    ghost_printLine("Chunk occupancy (beta)",NULL,"%f",SELL(mat)->beta);
    ghost_printLine("Row length variance",NULL,"%f",SELL(mat)->variance);
    ghost_printLine("Row length standard deviation",NULL,"%f",SELL(mat)->deviation);
    ghost_printLine("Row length coefficient of variation",NULL,"%f",SELL(mat)->cv);
    ghost_printLine("Threads per row (T)",NULL,"%d",SELL(mat)->T);
    if (mat->traits->flags & GHOST_SPM_SORTED) {
        ghost_printLine("Sorted",NULL,"yes");
        ghost_printLine("Scope (sigma)",NULL,"%u",*(unsigned int *)(mat->traits->aux));
        ghost_printLine("Permuted columns",NULL,"%s",mat->traits->flags&GHOST_SPM_PERMUTECOLIDX?"yes":"no");
    } else {
        ghost_printLine("Sorted",NULL,"no");
    }
}

static const char * SELL_formatName(ghost_mat_t *mat)
{
    // TODO format SELL-C-sigma
    UNUSED(mat);
    return "SELL";
}

static ghost_midx_t SELL_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
    if (mat && i<mat->nrows) {
        return SELL(mat)->rowLen[i];
    }

    return 0;
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
    if (mat->data == NULL) {
        return 0;
    }
    return (size_t)((mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_mnnz_t) + 
            mat->nEnts*(sizeof(ghost_midx_t)+ghost_sizeofDataType(mat->traits->datatype)));
}

static int compareNZEPerRow( const void* a, const void* b ) 
{
    /* comparison function for ghost_sorting_t; 
     * sorts rows with higher number of non-zero elements first */

    return  ((ghost_sorting_t*)b)->nEntsInRow - ((ghost_sorting_t*)a)->nEntsInRow;
}
static ghost_error_t SELL_fromRowFunc(ghost_mat_t *mat, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, ghost_spmFromRowFunc_flags_t flags)
{
    //   WARNING_LOG("SELL-%d-%d from row func",SELL(mat)->chunkHeight,SELL(mat)->scope);
    UNUSED(base);
    UNUSED(flags);
    int nprocs = 1;
    int me;
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs));
    GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&me));

    //char *tmpval;
    //ghost_midx_t *tmpcol;    
    ghost_midx_t i,col,row;
    ghost_midx_t chunk,j;
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

    ghost_midx_t *rowPerm = NULL;
    ghost_midx_t *invRowPerm = NULL;

    ghost_sorting_t* rowSort = NULL;
    mat->context->rowPerm = rowPerm;
    mat->context->invRowPerm = invRowPerm;

    ghost_midx_t nChunks = mat->nrowsPadded/SELL(mat)->chunkHeight;
    SELL(mat)->chunkStart = (ghost_mnnz_t *)ghost_malloc((nChunks+1)*sizeof(ghost_mnnz_t));
    SELL(mat)->chunkMin = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->chunkLen = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->chunkLenPadded = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->rowLen = (ghost_midx_t *)ghost_malloc((mat->nrowsPadded)*sizeof(ghost_midx_t));
    SELL(mat)->rowLenPadded = (ghost_midx_t *)ghost_malloc((mat->nrowsPadded)*sizeof(ghost_midx_t));
    SELL(mat)->chunkStart[0] = 0;
    SELL(mat)->maxRowLen = 0;
    mat->nEnts = 0;
    mat->nnz = 0;
    SELL(mat)->maxRowLen = 0;
    SELL(mat)->nMaxRows = 0;
    SELL(mat)->variance = 0.;
    SELL(mat)->deviation = 0.;
    SELL(mat)->cv = 0.;

    ghost_midx_t maxRowLenInChunk = 0;
    ghost_midx_t maxRowLen = 0;
    ghost_mnnz_t nEnts = 0, nnz = 0;
    SELL(mat)->chunkStart[0] = 0;


    if (mat->traits->flags & GHOST_SPM_SORTED) {
#pragma omp parallel private(i)
        { 
            char * tmpval = ghost_malloc(maxrowlen*sizeofdt);
            ghost_midx_t * tmpcol = (ghost_midx_t *)ghost_malloc(maxrowlen*sizeof(ghost_midx_t));
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nChunks; chunk++ ) {
                for (i=0; i<SELL(mat)->chunkHeight; i++) {
                    ghost_midx_t row = chunk*SELL(mat)->chunkHeight+i;

                    if (row < mat->nrows) {
                        func(mat->context->lfRow[me]+row,&SELL(mat)->rowLen[row],tmpcol,tmpval);
                    } else {
                        SELL(mat)->rowLen[row] = 0;
                    }
                }
            }

            free(tmpval);
            free(tmpcol);
        }
        rowPerm = (ghost_midx_t *)ghost_malloc(mat->nrows*sizeof(ghost_midx_t));
        invRowPerm = (ghost_midx_t *)ghost_malloc(mat->nrows*sizeof(ghost_midx_t));

        mat->context->rowPerm = rowPerm;
        mat->context->invRowPerm = invRowPerm;

        DEBUG_LOG(1,"Sorting matrix rows");

        rowSort = (ghost_sorting_t*)ghost_malloc(mat->nrows * sizeof(ghost_sorting_t));

        ghost_midx_t c;
        for (c=0; c<mat->nrows/SELL(mat)->scope; c++)  
        {
            for( i = c*SELL(mat)->scope; i < (c+1)*SELL(mat)->scope; i++ ) 
            {
                rowSort[i].row = i;
                rowSort[i].nEntsInRow = SELL(mat)->rowLen[i];
            } 

            qsort( rowSort+c*SELL(mat)->scope, SELL(mat)->scope, sizeof( ghost_sorting_t  ), compareNZEPerRow );
        }
        for( i = c*SELL(mat)->scope; i < mat->nrows; i++ ) 
        { // remainder
            rowSort[i].row = i;
            rowSort[i].nEntsInRow = SELL(mat)->rowLen[i];
        }
        qsort( rowSort+c*SELL(mat)->scope, mat->nrows-c*SELL(mat)->scope, sizeof( ghost_sorting_t  ), compareNZEPerRow );

        for(i=0; i < mat->nrows; ++i) {
            (invRowPerm)[i] = rowSort[i].row;
            (rowPerm)[rowSort[i].row] = i;
        }

#pragma omp parallel private(maxRowLenInChunk,i) reduction (+:nEnts,nnz)
        { 
            char * tmpval = ghost_malloc(maxrowlen*sizeofdt);
            ghost_midx_t * tmpcol = (ghost_midx_t *)ghost_malloc(maxrowlen*sizeof(ghost_midx_t));
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nChunks; chunk++ ) {
                for (i=0; i<SELL(mat)->chunkHeight; i++) {
                    ghost_midx_t row = chunk*SELL(mat)->chunkHeight+i;

                    if (row < mat->nrows) {
                        func(mat->context->lfRow[me]+invRowPerm[row],&SELL(mat)->rowLen[row],tmpcol,tmpval);
                    } else {
                        SELL(mat)->rowLen[row] = 0;
                    }

                    SELL(mat)->rowLenPadded[row] = ghost_pad(SELL(mat)->rowLen[row],SELL(mat)->T);

                    nnz += SELL(mat)->rowLen[row];
                    maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[row]);
                }
#pragma omp critical
                maxRowLen = MAX(maxRowLen,maxRowLenInChunk);
                SELL(mat)->chunkLen[chunk] = maxRowLenInChunk;
                SELL(mat)->chunkLenPadded[chunk] = ghost_pad(maxRowLenInChunk,SELL(mat)->T);
                nEnts += SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
                maxRowLenInChunk = 0;
            }

            free(tmpval);
            free(tmpcol);
        }
    } else {

#pragma omp parallel private(maxRowLenInChunk,i) reduction (+:nEnts,nnz) 
        { 
            char * tmpval = ghost_malloc(maxrowlen*sizeofdt);
            ghost_midx_t * tmpcol = (ghost_midx_t *)ghost_malloc(maxrowlen*sizeof(ghost_midx_t));
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nChunks; chunk++ ) {
                for (i=0; i<SELL(mat)->chunkHeight; i++) {
                    ghost_midx_t row = chunk*SELL(mat)->chunkHeight+i;

                    if (row < mat->nrows) {
                        func(mat->context->lfRow[me]+row,&SELL(mat)->rowLen[row],tmpcol,tmpval);
                    } else {
                        SELL(mat)->rowLen[row] = 0;
                    }

                    SELL(mat)->rowLenPadded[row] = ghost_pad(SELL(mat)->rowLen[row],SELL(mat)->T);

                    nnz += SELL(mat)->rowLen[row];
                    maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[row]);
                }
#pragma omp critical
                maxRowLen = MAX(maxRowLen,maxRowLenInChunk);
                SELL(mat)->chunkLen[chunk] = maxRowLenInChunk;
                SELL(mat)->chunkLenPadded[chunk] = ghost_pad(maxRowLenInChunk,SELL(mat)->T);
                nEnts += SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
                maxRowLenInChunk = 0;
            }

            free(tmpval);
            free(tmpcol);
        }
    }

    for( chunk = 0; chunk < nChunks; chunk++ ) {
        SELL(mat)->chunkStart[chunk+1] = SELL(mat)->chunkStart[chunk] + SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
    }

    SELL(mat)->maxRowLen = maxRowLen; 
    mat->nEnts = nEnts;
    mat->nnz = nnz;
    SELL(mat)->beta = mat->nnz*1.0/(double)mat->nEnts;

    SELL(mat)->val = (char *)ghost_malloc_align(ghost_sizeofDataType(mat->traits->datatype)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT);
    SELL(mat)->col = (ghost_midx_t *)ghost_malloc_align(sizeof(ghost_midx_t)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT);


#pragma omp parallel for schedule(runtime) private (i,j)
    for (chunk = 0; chunk < nChunks; chunk++) {
        for (j=0; j<SELL(mat)->chunkLenPadded[chunk]; j++) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = 0;
                memset(&SELL(mat)->val[sizeofdt*(SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i)],0,sizeofdt);
            }
        }
    }


#pragma omp parallel private(i,col,row)
    {
        char * tmpval = ghost_malloc(SELL(mat)->chunkHeight*maxrowlen*sizeofdt);
        ghost_midx_t * tmpcol = (ghost_midx_t *)ghost_malloc(SELL(mat)->chunkHeight*maxrowlen*sizeof(ghost_midx_t));
        memset(tmpval,0,sizeofdt*maxrowlen*SELL(mat)->chunkHeight);
        memset(tmpcol,0,sizeof(ghost_midx_t)*maxrowlen*SELL(mat)->chunkHeight);
#pragma omp for schedule(runtime)
        for( chunk = 0; chunk < nChunks; chunk++ ) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                row = chunk*SELL(mat)->chunkHeight+i;

                if (row < mat->nrows) {
                    if (mat->traits->flags & GHOST_SPM_SORTED) {
                        func(mat->context->lfRow[me]+invRowPerm[row],&SELL(mat)->rowLen[row],&tmpcol[maxrowlen*i],&tmpval[maxrowlen*i*sizeofdt]);
                    } else {
                        func(mat->context->lfRow[me]+row,&SELL(mat)->rowLen[row],&tmpcol[maxrowlen*i],&tmpval[maxrowlen*i*sizeofdt]);
                    }
                }

                for (col = 0; col<SELL(mat)->chunkLenPadded[chunk]; col++) {
                    if (mat->traits->flags & GHOST_SPM_SORTED) {
                        if ((tmpcol[i*maxrowlen+col] >= mat->context->lfRow[me]) && (tmpcol[i*maxrowlen+col] < (mat->context->lfRow[me]+mat->nrows))) { // local entry: copy with permutation
                            memcpy(&SELL(mat)->val[sizeofdt*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&tmpval[sizeofdt*(i*maxrowlen+col)],sizeofdt);
                            if (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX) {
                                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = rowPerm[tmpcol[i*maxrowlen+col]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                            } else {
                                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*maxrowlen+col];
                            }
                        } else { // remote entry: copy without permutation
                            memcpy(&SELL(mat)->val[sizeofdt*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&tmpval[sizeofdt*(i*maxrowlen+col)],sizeofdt);
                            SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*maxrowlen+col];
                        }
                    } else {
                        memcpy(&SELL(mat)->val[sizeofdt*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&tmpval[sizeofdt*(i*maxrowlen+col)],sizeofdt);
                        SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*maxrowlen+col];
                    }
                }
            }
            memset(tmpval,0,sizeofdt*maxrowlen*SELL(mat)->chunkHeight);
            memset(tmpcol,0,sizeof(ghost_midx_t)*maxrowlen*SELL(mat)->chunkHeight);
        }
        free(tmpval);
        free(tmpcol);
    }
    if (mat->traits->flags & GHOST_SPM_ASC_COLIDX) {
        WARNING_LOG("Ignoring ASC_COLIDX flag");
    }

    if (!(mat->context->flags & GHOST_CONTEXT_REDUNDANT)) {
#if GHOST_HAVE_MPI

        mat->context->lnEnts[me] = mat->nEnts;

        ghost_mnnz_t nents;
        nents = mat->context->lnEnts[me];
        MPI_safecall(MPI_Allgather(&nents,1,ghost_mpi_dt_mnnz,mat->context->lnEnts,1,ghost_mpi_dt_mnnz,mat->context->mpicomm));

        for (i=0; i<nprocs; i++) {
            mat->context->lfEnt[i] = 0;
        } 

        for (i=1; i<nprocs; i++) {
            mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
        } 

        mat->split(mat);
#endif
    }
#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPM_HOST))
        mat->upload(mat);
#endif
    return GHOST_SUCCESS;

}

static ghost_error_t SELL_split(ghost_mat_t *mat)
{
    if (!mat) {
        ERROR_LOG("Matrix is NULL");
        return GHOST_ERR_INVALID_ARG;
    }
        
        
    SELL_TYPE *fullSELL = SELL(mat);
    SELL_TYPE *localSELL = NULL, *remoteSELL = NULL;
    DEBUG_LOG(1,"Splitting the SELL matrix into a local and remote part");
    ghost_midx_t j;
    ghost_midx_t i;
    int me;
    GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&me));

    ghost_mnnz_t lnEnts_l, lnEnts_r;
    ghost_mnnz_t current_l, current_r;

    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

    ghost_midx_t chunk;
    ghost_midx_t idx, row;

    ghost_setupCommunication(mat->context,fullSELL->col);

    ghost_createMatrix(mat->context,&mat->traits[0],1,&(mat->localPart));
    localSELL = mat->localPart->data;
    mat->localPart->traits->symmetry = mat->traits->symmetry;

    ghost_createMatrix(mat->context,&mat->traits[0],1,&(mat->remotePart));
    remoteSELL = mat->remotePart->data; 

    localSELL->T = fullSELL->T;
    remoteSELL->T = fullSELL->T;

    localSELL->chunkStart = (ghost_midx_t*) ghost_malloc((mat->nrowsPadded/fullSELL->chunkHeight+1)*sizeof( ghost_midx_t )); 
    localSELL->chunkMin = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    localSELL->chunkLen = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    localSELL->chunkLenPadded = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    localSELL->rowLen = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded*sizeof( ghost_midx_t )); 
    localSELL->rowLenPadded = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded*sizeof( ghost_midx_t )); 

    remoteSELL->chunkStart = (ghost_midx_t*) ghost_malloc((mat->nrowsPadded/fullSELL->chunkHeight+1)*sizeof( ghost_midx_t )); 
    remoteSELL->chunkMin = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    remoteSELL->chunkLen = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    remoteSELL->chunkLenPadded = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    remoteSELL->rowLen = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded*sizeof( ghost_midx_t )); 
    remoteSELL->rowLenPadded = (ghost_midx_t*) ghost_malloc(mat->nrowsPadded*sizeof( ghost_midx_t )); 

    for (i=0; i<mat->nrowsPadded; i++) {
        localSELL->rowLen[i] = 0;
        remoteSELL->rowLen[i] = 0;
        localSELL->rowLenPadded[i] = 0;
        remoteSELL->rowLenPadded[i] = 0;
    }
    for(chunk = 0; chunk < mat->nrowsPadded/fullSELL->chunkHeight; chunk++) {
        localSELL->chunkLen[chunk] = 0;
        remoteSELL->chunkLen[chunk] = 0;
        localSELL->chunkLenPadded[chunk] = 0;
        remoteSELL->chunkLenPadded[chunk] = 0;
        localSELL->chunkMin[chunk] = 0;
        remoteSELL->chunkMin[chunk] = 0;
    }
    localSELL->chunkStart[0] = 0;
    remoteSELL->chunkStart[0] = 0;

    mat->localPart->nnz = 0;
    mat->remotePart->nnz = 0;

    if (mat->traits->flags & GHOST_SPM_STORE_SPLIT) { // split computation

        lnEnts_l = 0;
        lnEnts_r = 0;

        for(chunk = 0; chunk < mat->nrowsPadded/fullSELL->chunkHeight; chunk++) {

            for (i=0; i<fullSELL->chunkLen[chunk]; i++) {
                for (j=0; j<fullSELL->chunkHeight; j++) {
                    row = chunk*fullSELL->chunkHeight+j;
                    idx = fullSELL->chunkStart[chunk]+i*fullSELL->chunkHeight+j;

                    if (i < fullSELL->rowLen[row]) {
                        if (fullSELL->col[idx] < mat->context->lnrows[me]) {
                            localSELL->rowLen[row]++;
                            mat->localPart->nnz++;
                        } else {
                            remoteSELL->rowLen[row]++;
                            mat->remotePart->nnz++;
                        }
                        localSELL->rowLenPadded[row] = ghost_pad(localSELL->rowLen[row],localSELL->T);
                        remoteSELL->rowLenPadded[row] = ghost_pad(remoteSELL->rowLen[row],remoteSELL->T);
                    }
                }
            }

            for (j=0; j<fullSELL->chunkHeight; j++) {
                row = chunk*fullSELL->chunkHeight+j;
                localSELL->chunkLen[chunk] = MAX(localSELL->chunkLen[chunk],localSELL->rowLen[row]);
                remoteSELL->chunkLen[chunk] = MAX(remoteSELL->chunkLen[chunk],remoteSELL->rowLen[row]);
            }
            lnEnts_l += localSELL->chunkLen[chunk]*fullSELL->chunkHeight;
            lnEnts_r += remoteSELL->chunkLen[chunk]*fullSELL->chunkHeight;
            localSELL->chunkStart[chunk+1] = lnEnts_l;
            remoteSELL->chunkStart[chunk+1] = lnEnts_r;

            localSELL->chunkLenPadded[chunk] = ghost_pad(localSELL->chunkLen[chunk],localSELL->T);
            remoteSELL->chunkLenPadded[chunk] = ghost_pad(remoteSELL->chunkLen[chunk],remoteSELL->T);

        }



        /*
           for (i=0; i<fullSELL->nEnts;i++) {
           if (fullSELL->col[i]<mat->context->lnrows[me]) lnEnts_l++;
           }
           lnEnts_r = mat->context->lnEnts[me]-lnEnts_l;*/


        localSELL->val = ghost_malloc(lnEnts_l*sizeofdt); 
        localSELL->col = (ghost_midx_t*) ghost_malloc(lnEnts_l*sizeof( ghost_midx_t )); 

        remoteSELL->val = ghost_malloc(lnEnts_r*sizeofdt); 
        remoteSELL->col = (ghost_midx_t*) ghost_malloc(lnEnts_r*sizeof( ghost_midx_t )); 

        mat->localPart->nrows = mat->nrows;
        mat->localPart->nrowsPadded = mat->nrowsPadded;
        mat->localPart->nEnts = lnEnts_l;
        localSELL->chunkHeight = fullSELL->chunkHeight;
        localSELL->scope = fullSELL->scope;

        mat->remotePart->nrows = mat->nrows;
        mat->remotePart->nrowsPadded = mat->nrowsPadded;
        mat->remotePart->nEnts = lnEnts_r;
        remoteSELL->chunkHeight = fullSELL->chunkHeight;
        remoteSELL->scope = 1;

#pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < mat->localPart->nrowsPadded/localSELL->chunkHeight; chunk++) {
            for (i=0; i<localSELL->chunkLenPadded[chunk]; i++) {
                for (j=0; j<localSELL->chunkHeight; j++) {
                    idx = localSELL->chunkStart[chunk]+i*localSELL->chunkHeight+j;
                    memset(&((char *)(localSELL->val))[idx*sizeofdt],0,sizeofdt);
                    localSELL->col[idx] = 0;
                }
            }
        }

#pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < mat->remotePart->nrowsPadded/remoteSELL->chunkHeight; chunk++) {
            for (i=0; i<remoteSELL->chunkLenPadded[chunk]; i++) {
                for (j=0; j<remoteSELL->chunkHeight; j++) {
                    idx = remoteSELL->chunkStart[chunk]+i*remoteSELL->chunkHeight+j;
                    memset(&((char *)(remoteSELL->val))[idx*sizeofdt],0,sizeofdt);
                    remoteSELL->col[idx] = 0;
                }
            }
        }

        current_l = 0;
        current_r = 0;
        ghost_midx_t col_l[fullSELL->chunkHeight], col_r[fullSELL->chunkHeight];

        for(chunk = 0; chunk < mat->nrowsPadded/fullSELL->chunkHeight; chunk++) {

            for (j=0; j<fullSELL->chunkHeight; j++) {
                col_l[j] = 0;
                col_r[j] = 0;
            }

            for (i=0; i<fullSELL->chunkLen[chunk]; i++) {
                for (j=0; j<fullSELL->chunkHeight; j++) {
                    row = chunk*fullSELL->chunkHeight+j;
                    idx = fullSELL->chunkStart[chunk]+i*fullSELL->chunkHeight+j;

                    if (i<fullSELL->rowLen[row]) {
                        if (fullSELL->col[idx] < mat->context->lnrows[me]) {
                            if (col_l[j] < localSELL->rowLen[row]) {
                                ghost_midx_t lidx = localSELL->chunkStart[chunk]+col_l[j]*localSELL->chunkHeight+j;
                                localSELL->col[lidx] = fullSELL->col[idx];
                                memcpy(&localSELL->val[lidx*sizeofdt],&fullSELL->val[idx*sizeofdt],sizeofdt);
                                current_l++;
                            }
                            col_l[j]++;
                        }
                        else{
                            if (col_r[j] < remoteSELL->rowLen[row]) {
                                ghost_midx_t ridx = remoteSELL->chunkStart[chunk]+col_r[j]*remoteSELL->chunkHeight+j;
                                remoteSELL->col[ridx] = fullSELL->col[idx];
                                memcpy(&remoteSELL->val[ridx*sizeofdt],&fullSELL->val[idx*sizeofdt],sizeofdt);
                                current_r++;
                            }
                            col_r[j]++;
                        }
                    }
                }
            }
        }
    }

#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPM_HOST)) {
        mat->localPart->upload(mat->localPart);
        mat->remotePart->upload(mat->remotePart);
    }
#endif

    /*    INFO_LOG("+ local +++++++");
          for(chunk = 0; chunk < localSELL->nrowsPadded/localSELL->chunkHeight; chunk++) {
          INFO_LOG("chunk %d, start %d, len %d",chunk,localSELL->chunkStart[chunk],localSELL->chunkLen[chunk]);
          for (i=0; i<localSELL->chunkLen[chunk]; i++) {
          for (j=0; j<localSELL->chunkHeight; j++) {
          idx = localSELL->chunkStart[chunk]+i*localSELL->chunkHeight+j;
          INFO_LOG("%d,%d,%d:%d: %f|%d",chunk,i,j,idx,((double *)(localSELL->val))[idx],localSELL->col[idx]);
          }
          }
          }

          INFO_LOG("+ remote +++++++");
          for(chunk = 0; chunk < remoteSELL->nrowsPadded/remoteSELL->chunkHeight; chunk++) {
          INFO_LOG("chunk %d, start %d, len %d",chunk,remoteSELL->chunkStart[chunk],remoteSELL->chunkLen[chunk]);
          for (i=0; i<remoteSELL->chunkLen[chunk]; i++) {
          for (j=0; j<remoteSELL->chunkHeight; j++) {
          idx = remoteSELL->chunkStart[chunk]+i*remoteSELL->chunkHeight+j;
          INFO_LOG("%d,%d,%d:%d: %f|%d",chunk,i,j,idx,((double *)(remoteSELL->val))[idx],remoteSELL->col[idx]);
          }
          }
          }*/

    return GHOST_SUCCESS;
}

static ghost_error_t SELL_fromBin(ghost_mat_t *mat, char *matrixPath)
{
    DEBUG_LOG(1,"Creating SELL matrix from binary file");

    ghost_context_t *context = mat->context;
    int nprocs = 1;
    int me;
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs));
    GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&me));

    ghost_midx_t *rpt;
    ghost_midx_t i;
    int proc;
    ghost_midx_t chunk,j;
    ghost_matfile_header_t header;

    ghost_readMatFileHeader(matrixPath,&header);

    if (header.version != 1)
        ABORT("Can not read version %d of binary CRS format!",header.version);

    if (header.base != 0)
        ABORT("Can not read matrix with %d-based indices!",header.base);

    if (!ghost_symmetryValid(header.symmetry))
        ABORT("Symmetry is invalid! (%d)",header.symmetry);
    if (header.symmetry != GHOST_BINCRS_SYMM_GENERAL)
        ABORT("Can not handle symmetry different to general at the moment!");
    mat->traits->symmetry = header.symmetry;

    if (!ghost_datatypeValid(header.datatype))
        ABORT("Datatype is invalid! (%d)",header.datatype);

    if (me == 0) {
        if (context->flags & GHOST_CONTEXT_DIST_ROWS) { // lnents and lfent have to be filled!
            rpt = (ghost_mnnz_t *)ghost_malloc(sizeof(ghost_mnnz_t)*(context->gnrows+1));
            ghost_readRpt(rpt,matrixPath,0,context->gnrows+1);  // read rpt

            /* DEBUG_LOG(1,"Adjust lfrow and lnrows for each process according to the SELL chunk height of %"PRmatIDX,SELL(mat)->chunkHeight);
               for (proc=0; proc<nprocs; proc++) {
               if (context->lfRow[proc] % SELL(mat)->chunkHeight > SELL(mat)->chunkHeight/2) {
               ghost_midx_t old = context->lfRow[proc];
               context->lfRow[proc] += SELL(mat)->chunkHeight - (context->lfRow[proc] % SELL(mat)->chunkHeight); 
               DEBUG_LOG(1,"PE%d: %"PRmatIDX"->%"PRmatIDX,proc,old,context->lfRow[proc]); 
               } else if (context->lfRow[proc] % SELL(mat)->chunkHeight > 0) {
               ghost_midx_t old = context->lfRow[proc];
               context->lfRow[proc] -= context->lfRow[proc] % SELL(mat)->chunkHeight;
               if (proc>0 && (context->lfRow[proc] == context->lfRow[proc-1])) {
               context->lfRow[proc] += SELL(mat)->chunkHeight;
               }
               DEBUG_LOG(1,"PE%d: %"PRmatIDX"->%"PRmatIDX,proc,old,context->lfRow[proc]); 
               } else {
               DEBUG_LOG(1,"PE%d: %"PRmatIDX,proc,context->lfRow[proc]);
               }
               }*/


            mat->context->lfEnt[0] = 0;

            for (proc=1; proc<nprocs; proc++){
                mat->context->lfEnt[proc] = rpt[mat->context->lfRow[proc]];
            }
        } else {
            rpt = context->rpt;
        }
    }
#if GHOST_HAVE_MPI
    MPI_safecall(MPI_Bcast(mat->context->lfRow,  nprocs, ghost_mpi_dt_midx, 0, mat->context->mpicomm));
    MPI_safecall(MPI_Bcast(mat->context->lnrows, nprocs, ghost_mpi_dt_midx, 0, mat->context->mpicomm));
    MPI_safecall(MPI_Bcast(mat->context->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, mat->context->mpicomm));
#endif


    mat->nrows = context->lnrows[me];
    mat->nrowsPadded = ghost_pad(context->lnrows[me],SELL(mat)->chunkHeight);

    ghost_midx_t nChunks =  mat->nrowsPadded/SELL(mat)->chunkHeight;
    SELL(mat)->chunkStart = (ghost_midx_t *)ghost_malloc((nChunks+1)*sizeof(ghost_midx_t));
    SELL(mat)->chunkMin = (ghost_midx_t *)ghost_malloc(nChunks*sizeof(ghost_midx_t));
    SELL(mat)->chunkLen = (ghost_midx_t *)ghost_malloc(nChunks*sizeof(ghost_midx_t));
    SELL(mat)->chunkLenPadded = (ghost_midx_t *)ghost_malloc(nChunks*sizeof(ghost_midx_t));
    SELL(mat)->rowLen = (ghost_midx_t *)ghost_malloc(mat->nrowsPadded*sizeof(ghost_midx_t));
    SELL(mat)->rowLenPadded = (ghost_midx_t *)ghost_malloc(mat->nrowsPadded*sizeof(ghost_midx_t));

    if (me != 0) {
        rpt = (ghost_midx_t *)ghost_malloc((context->lnrows[me]+1)*sizeof(ghost_midx_t));
    }
#if GHOST_HAVE_MPI
    MPI_Request req[nprocs];
    MPI_Status stat[nprocs];
    int msgcount = 0;

    for (proc=0;proc<nprocs;proc++) 
        req[proc] = MPI_REQUEST_NULL;

    if (me != 0) {
        MPI_safecall(MPI_Irecv(rpt,context->lnrows[me]+1,ghost_mpi_dt_midx,0,me,context->mpicomm,&req[msgcount]));
        msgcount++;
    } else {
        for (proc=1;proc<nprocs;proc++) {
            MPI_safecall(MPI_Isend(&rpt[context->lfRow[proc]],context->lnrows[proc]+1,ghost_mpi_dt_midx,proc,proc,context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    }
    MPI_safecall(MPI_Waitall(msgcount,req,stat));
#endif 

    for (i=0;i<context->lnrows[me]+1;i++) {
        rpt[i] -= context->lfEnt[me]; 
    }

    mat->nnz = rpt[context->lnrows[me]];
    mat->nEnts = 0;

    ghost_midx_t *rowPerm = NULL;
    ghost_midx_t *invRowPerm = NULL;

    ghost_sorting_t* rowSort = NULL;

    mat->context->rowPerm = rowPerm;
    mat->context->invRowPerm = invRowPerm;

    ghost_midx_t maxRowLenInChunk = 0;
    ghost_midx_t minRowLenInChunk = INT_MAX;

    SELL(mat)->maxRowLen = 0;
    SELL(mat)->chunkStart[0] = 0;    

    if (mat->traits->flags & GHOST_SPM_SORTED) {
        DEBUG_LOG(1,"Extracting row lenghts");
        rowPerm = (ghost_midx_t *)ghost_malloc(mat->nrows*sizeof(ghost_midx_t));
        invRowPerm = (ghost_midx_t *)ghost_malloc(mat->nrows*sizeof(ghost_midx_t));

        mat->context->rowPerm = rowPerm;
        mat->context->invRowPerm = invRowPerm;

        DEBUG_LOG(1,"Sorting matrix rows");

        rowSort = (ghost_sorting_t*)ghost_malloc(mat->nrows * sizeof(ghost_sorting_t));

        ghost_midx_t c;
        for (c=0; c<mat->nrows/SELL(mat)->scope; c++)  
        {
            for( i = c*SELL(mat)->scope; i < (c+1)*SELL(mat)->scope; i++ ) 
            {
                rowSort[i].row = i;
                if (i<mat->nrows) {
                    rowSort[i].nEntsInRow = rpt[i+1] - rpt[i];
                } else {
                    rowSort[i].nEntsInRow = 0;
                }
            } 

            qsort( rowSort+c*SELL(mat)->scope, SELL(mat)->scope, sizeof( ghost_sorting_t  ), compareNZEPerRow );
        }
        for( i = c*SELL(mat)->scope; i < mat->nrows; i++ ) 
        { // remainder
            rowSort[i].row = i;
            if (i<mat->nrows) {
                rowSort[i].nEntsInRow = rpt[i+1] - rpt[i];
            } else {
                rowSort[i].nEntsInRow = 0;
            }
        }
        qsort( rowSort+c*SELL(mat)->scope, mat->nrows-c*SELL(mat)->scope, sizeof( ghost_sorting_t  ), compareNZEPerRow );

        for(i=0; i < mat->nrows; ++i) {
            /* invRowPerm maps an index in the permuted system to the original index,
             * rowPerm gets the original index and returns the corresponding permuted position.
             */
            (invRowPerm)[i] = rowSort[i].row;
            (rowPerm)[rowSort[i].row] = i;
        }

    } 
    INFO_LOG("%"PRmatIDX" %"PRmatIDX" %"PRmatIDX,mat->nEnts,mat->nrows,mat->nrowsPadded);

    DEBUG_LOG(1,"Extracting row lenghts");
    for( chunk = 0; chunk < nChunks; chunk++ ) {
        for (i=0; i<SELL(mat)->chunkHeight; i++) {
            ghost_midx_t row = chunk*SELL(mat)->chunkHeight+i;
            if (row < mat->nrows) {
                if (mat->traits->flags & GHOST_SPM_SORTED) {
                    SELL(mat)->rowLen[row] = rowSort[row].nEntsInRow;
                } else {
                    SELL(mat)->rowLen[row] = rpt[row+1]-rpt[row];
                }
            } else {
                SELL(mat)->rowLen[row] = 0;
            }
            SELL(mat)->rowLenPadded[row] = ghost_pad(SELL(mat)->rowLen[row],SELL(mat)->T);

            maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[row]);
            minRowLenInChunk = MIN(minRowLenInChunk,SELL(mat)->rowLen[row]);
        }


        SELL(mat)->maxRowLen = MAX(SELL(mat)->maxRowLen,maxRowLenInChunk);
        SELL(mat)->chunkLen[chunk] = maxRowLenInChunk;
        SELL(mat)->chunkMin[chunk] = minRowLenInChunk;
        SELL(mat)->chunkLenPadded[chunk] = ghost_pad(maxRowLenInChunk,SELL(mat)->T);
        mat->nEnts += SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
        SELL(mat)->chunkStart[chunk+1] = mat->nEnts;
        maxRowLenInChunk = 0;
        minRowLenInChunk = INT_MAX;
    }

    mat->context->lnEnts[me] = mat->nEnts;
    SELL(mat)->beta = mat->nnz*1.0/(double)mat->nEnts;

#if GHOST_HAVE_MPI
    ghost_mnnz_t nents;
    nents = mat->context->lnEnts[me];
    MPI_safecall(MPI_Allgather(&nents,1,ghost_mpi_dt_mnnz,mat->context->lnEnts,1,ghost_mpi_dt_mnnz,mat->context->mpicomm));
#endif

    DEBUG_LOG(1,"SELL matrix has %"PRmatIDX" (padded to %"PRmatIDX") rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros and %"PRmatNNZ" entries",mat->nrows,mat->nrowsPadded,mat->ncols,mat->nnz,mat->context->lnEnts[me]);

    SELL(mat)->val = (char *)ghost_malloc_align(ghost_sizeofDataType(mat->traits->datatype)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT);
    SELL(mat)->col = (ghost_midx_t *)ghost_malloc_align(sizeof(ghost_midx_t)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT);

    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

#pragma omp parallel for schedule(runtime) private (i,j)
    for (chunk = 0; chunk < nChunks; chunk++) {
        for (j=0; j<SELL(mat)->chunkLenPadded[chunk]; j++) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = 0;
                memset(&SELL(mat)->val[sizeofdt*(SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i)],0,sizeofdt);
            }
        }
    }

    FILE *filed;

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s",matrixPath);
        return GHOST_ERR_IO;
    }


    WARNING_LOG("Memory usage may be high because read-in of CRS data is done at once and not chunk-wise");
    /*ghost_midx_t *tmpcol = (ghost_midx_t *)ghost_malloc(SELL(mat)->maxRowLen*SELL(mat)->chunkHeight*sizeof(ghost_midx_t));
      char *tmpval = (char *)ghost_malloc(SELL(mat)->maxRowLen*SELL(mat)->chunkHeight*sizeofdt);*/
    ghost_midx_t *tmpcol = (ghost_midx_t *)ghost_malloc(mat->nnz*sizeof(ghost_midx_t));
    char *tmpval = (char *)ghost_malloc(mat->nnz*sizeofdt);
    GHOST_CALL_RETURN(ghost_readCol(tmpcol, matrixPath, mat->context->lfEnt[me], mat->nnz));
    GHOST_CALL_RETURN(ghost_readVal(tmpval, mat->traits->datatype, matrixPath,  mat->context->lfEnt[me], mat->nnz));

    INFO_LOG("%"PRmatIDX" rows, %"PRmatIDX" chunks %"PRmatIDX" chunkheight",mat->nrows,nChunks,SELL(mat)->chunkHeight);
    ghost_midx_t row = 0;
    for (chunk = 0; chunk < nChunks; chunk++) {
        /*    memset(tmpcol,0,SELL(mat)->maxRowLen*SELL(mat)->chunkHeight*sizeof(ghost_midx_t));
              memset(tmpval,0,SELL(mat)->maxRowLen*SELL(mat)->chunkHeight*sizeofdt);

              ghost_midx_t firstNzOfChunk = context->lfEnt[me]+rpt[chunk*SELL(mat)->chunkHeight];
              ghost_midx_t nnzInChunk;

              if ((chunk+1)*SELL(mat)->chunkHeight <= mat->nrows) { // chunk is fully in matrix
              nnzInChunk = rpt[(chunk+1)*SELL(mat)->chunkHeight] - rpt[chunk*SELL(mat)->chunkHeight];
              } else { // parts of the chunk are out of matrix
              nnzInChunk = mat->nnz - rpt[chunk*SELL(mat)->chunkHeight];
              }


              GHOST_CALL_RETURN(ghost_readColOpen(tmpcol,matrixPath,firstNzOfChunk,nnzInChunk,filed));
              GHOST_CALL_RETURN(ghost_readValOpen(tmpval,mat->traits->datatype,matrixPath,firstNzOfChunk,nnzInChunk,filed));
         */
       
        ghost_midx_t col;
        ghost_midx_t *curRowCols;
        char * curRowVals;
        for (i=0; (i<SELL(mat)->chunkHeight) && (row < mat->nrows); i++, row++) {
            if (mat->traits->flags & GHOST_SPM_SORTED) {
                if (!invRowPerm) {
                    WARNING_LOG("invRowPerm is NULL but matrix should be sorted");
                }
                curRowCols = &tmpcol[rpt[invRowPerm[row]]];
                curRowVals = &tmpval[rpt[invRowPerm[row]]*sizeofdt];
            } else {
                curRowCols = &tmpcol[rpt[row]];
                curRowVals = &tmpval[rpt[row]*sizeofdt];
            }
            for (col=0; col<SELL(mat)->rowLen[row]; col++) {


                if ((mat->traits->flags & (GHOST_SPM_SORTED | GHOST_SPM_PERMUTECOLIDX)) &&
                        (curRowCols[col] >= mat->context->lfRow[me]) && 
                        (curRowCols[col] < (mat->context->lfRow[me]+mat->nrows))) {
                    SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = rowPerm[curRowCols[col]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                } else {
                    SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = curRowCols[col];
                }

                memcpy(&SELL(mat)->val[sizeofdt*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&curRowVals[col*sizeofdt],sizeofdt);
            }
            // sort cols and vals ascending by local col idx
            if (mat->traits->flags & GHOST_SPM_ASC_COLIDX) {
                ghost_midx_t n;
                curRowCols = &SELL(mat)->col[SELL(mat)->chunkStart[chunk]+i];
                curRowVals = &SELL(mat)->val[sizeofdt*(SELL(mat)->chunkStart[chunk]+i)];
                ghost_midx_t tmpcol;
                char *tmpval = ghost_malloc(sizeofdt);
                for (n=SELL(mat)->rowLen[row]; n>1; n--) {
                    for (col=0; col<n-1; col++) {
                        if (curRowCols[col*SELL(mat)->chunkHeight] > curRowCols[(col+1)*SELL(mat)->chunkHeight]) {
                            tmpcol = curRowCols[col*SELL(mat)->chunkHeight];
                            curRowCols[col*SELL(mat)->chunkHeight] = curRowCols[(col+1)*SELL(mat)->chunkHeight];
                            curRowCols[(col+1)*SELL(mat)->chunkHeight] = tmpcol; 

                            memcpy(&tmpval,&curRowVals[sizeofdt*(col*SELL(mat)->chunkHeight)],sizeofdt);
                            memcpy(&curRowVals[sizeofdt*(col*SELL(mat)->chunkHeight)],&curRowVals[sizeofdt*((col+1)*SELL(mat)->chunkHeight)],sizeofdt);
                            memcpy(&curRowVals[sizeofdt*((col+1)*SELL(mat)->chunkHeight)],&tmpval,sizeofdt);
                        }
                    }
                }
            }
        }

    }

    mat->split(mat);


#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPM_HOST))
        mat->upload(mat);
#endif


    free(tmpcol);
    free(tmpval);
    free(rpt);
    fclose(filed);

    DEBUG_LOG(1,"SELL matrix successfully created");
    return GHOST_SUCCESS;
}

static const char * SELL_stringify(ghost_mat_t *mat, int dense)
{
    return SELL_stringify_funcs[ghost_dataTypeIdx(mat->traits->datatype)](mat, dense);
}

static void SELL_fromCRS(ghost_mat_t *mat, ghost_mat_t *crs)
{
    SELL_fromCRS_funcs[ghost_dataTypeIdx(mat->traits->datatype)](mat,crs);
}

static ghost_error_t SELL_upload(ghost_mat_t* mat) 
{
#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPM_HOST)) {
        DEBUG_LOG(1,"Creating matrix on CUDA device");
        SELL(mat)->cumat = (CU_SELL_TYPE *)ghost_malloc(sizeof(CU_SELL_TYPE));
        SELL(mat)->cumat->rowLen = CU_allocDeviceMemory((mat->nrows)*sizeof(ghost_midx_t));
        SELL(mat)->cumat->rowLenPadded = CU_allocDeviceMemory((mat->nrows)*sizeof(ghost_midx_t));
        SELL(mat)->cumat->col = CU_allocDeviceMemory((mat->nEnts)*sizeof(ghost_midx_t));
        SELL(mat)->cumat->val = CU_allocDeviceMemory((mat->nEnts)*ghost_sizeofDataType(mat->traits->datatype));
        SELL(mat)->cumat->chunkStart = CU_allocDeviceMemory((mat->nrowsPadded/SELL(mat)->chunkHeight+1)*sizeof(ghost_mnnz_t));
        SELL(mat)->cumat->chunkLen = CU_allocDeviceMemory((mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_midx_t));

        SELL(mat)->cumat->nrows = mat->nrows;
        SELL(mat)->cumat->nrowsPadded = mat->nrowsPadded;
        CU_copyHostToDevice(SELL(mat)->cumat->rowLen, SELL(mat)->rowLen, mat->nrows*sizeof(ghost_midx_t));
        CU_copyHostToDevice(SELL(mat)->cumat->rowLenPadded, SELL(mat)->rowLenPadded, mat->nrows*sizeof(ghost_midx_t));
        CU_copyHostToDevice(SELL(mat)->cumat->col, SELL(mat)->col, mat->nEnts*sizeof(ghost_midx_t));
        CU_copyHostToDevice(SELL(mat)->cumat->val, SELL(mat)->val, mat->nEnts*ghost_sizeofDataType(mat->traits->datatype));
        CU_copyHostToDevice(SELL(mat)->cumat->chunkStart, SELL(mat)->chunkStart, (mat->nrowsPadded/SELL(mat)->chunkHeight+1)*sizeof(ghost_mnnz_t));
        CU_copyHostToDevice(SELL(mat)->cumat->chunkLen, SELL(mat)->chunkLen, (mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_midx_t));
    }
#else
    if (mat->traits->flags & GHOST_SPM_DEVICE) {
        ERROR_LOG("Device matrix cannot be created without CUDA");
        return GHOST_ERR_CUDA;
    }
#endif
    return GHOST_SUCCESS;
}


static void SELL_free(ghost_mat_t *mat)
{
    if (!mat) {
        return;
    }

    if (mat->data) {
#ifdef GHOST_HAVE_CUDA
        if (mat->traits->flags & GHOST_SPM_DEVICE) {
            CU_freeDeviceMemory(SELL(mat)->cumat->rowLen);
            CU_freeDeviceMemory(SELL(mat)->cumat->rowLenPadded);
            CU_freeDeviceMemory(SELL(mat)->cumat->col);
            CU_freeDeviceMemory(SELL(mat)->cumat->val);
            CU_freeDeviceMemory(SELL(mat)->cumat->chunkStart);
            CU_freeDeviceMemory(SELL(mat)->cumat->chunkLen);
            free(SELL(mat)->cumat);
        }
#endif
        free(SELL(mat)->val);
        free(SELL(mat)->col);
        free(SELL(mat)->chunkStart);
        free(SELL(mat)->chunkMin);
        free(SELL(mat)->chunkLen);
        free(SELL(mat)->chunkLenPadded);
        free(SELL(mat)->rowLen);
        free(SELL(mat)->rowLenPadded);
    }

    
    free(mat->data);
    free(mat);

}

static ghost_error_t SELL_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, ghost_spmv_flags_t options)
{
    DEBUG_LOG(1,"Calling plain (maybe intrinsics) SELL kernel");
    DEBUG_LOG(2,"lhs vector has %s data and %"PRvecIDX" sub-vectors",ghost_datatypeName(lhs->traits->datatype),lhs->traits->nvecs);

    ghost_error_t (*kernel) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t) = NULL;

#if GHOST_HAVE_OPENMP
    /*    static int first = 1;
          if ((mat->byteSize(mat) < ghost_getSizeOfLLC()) || (SELL(mat)->deviation*1./(ghost_getMatNnz(mat)*1.0/(double)ghost_getMatNrows(mat)) < 0.4)) {
          if (first) {
          INFO_LOG("Setting OpenMP scheduling to STATIC for SELL SpMVM kernel");
          }
          omp_set_schedule(omp_sched_static,0);
          } else {
          if (first) {
          INFO_LOG("Setting OpenMP scheduling to GUIDED,4 for SELL SpMVM kernel");
          }
          omp_set_schedule(omp_sched_guided,4);
          }
          first=0;*/
#endif


#if GHOST_HAVE_SSE
#if !(GHOST_HAVE_LONGIDX)
    if (!((options & GHOST_SPMVM_AXPBY) ||
                (options & GHOST_SPMVM_APPLY_SCALE) ||
                (options & GHOST_SPMVM_APPLY_SHIFT))) {
        kernel = SELL_kernels_SSE
            [ghost_dataTypeIdx(mat->traits->datatype)]
            [ghost_dataTypeIdx(lhs->traits->datatype)];
    }
#endif
#elif GHOST_HAVE_AVX
        if (SELL(mat)->chunkHeight == 4) {
            kernel = SELL_kernels_AVX
                [ghost_dataTypeIdx(mat->traits->datatype)]
                [ghost_dataTypeIdx(lhs->traits->datatype)];
        } else if (SELL(mat)->chunkHeight == 32) {
            kernel = SELL_kernels_AVX_32
                [ghost_dataTypeIdx(mat->traits->datatype)]
                [ghost_dataTypeIdx(lhs->traits->datatype)];
        }
#elif GHOST_HAVE_MIC
#if !(GHOST_HAVE_LONGIDX)
    if (!((options & GHOST_SPMVM_AXPBY) ||
                (options & GHOST_SPMVM_APPLY_SCALE) ||
                (options & GHOST_SPMVM_APPLY_SHIFT))) {
        if (SELL(mat)->chunkHeight == 16) {
            kernel = SELL_kernels_MIC_16
                [ghost_dataTypeIdx(mat->traits->datatype)]
                [ghost_dataTypeIdx(lhs->traits->datatype)];
        } else if (SELL(mat)->chunkHeight == 32) {
            kernel = SELL_kernels_MIC_32
                [ghost_dataTypeIdx(mat->traits->datatype)]
                [ghost_dataTypeIdx(lhs->traits->datatype)];
        }
    }
#endif
#else
    kernel = SELL_kernels_plain
        [ghost_dataTypeIdx(mat->traits->datatype)]
        [ghost_dataTypeIdx(lhs->traits->datatype)];
#endif

    if (kernel == NULL) {
        //WARNING_LOG("Selected kernel cannot be found. Falling back to plain C version!");
        kernel = SELL_kernels_plain
            [ghost_dataTypeIdx(mat->traits->datatype)]
            [ghost_dataTypeIdx(lhs->traits->datatype)];
    }

    return kernel(mat,lhs,rhs,options);

    
}


#ifdef GHOST_HAVE_CUDA
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

#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
    ghost_midx_t c,j;
    ghost_mnnz_t offs;
    vector double tmp;
    vector double val;
    vector double rhs;


#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
    for (c=0; c<mat->nrowsPadded>>1; c++) 
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

static int ghost_selectSellChunkHeight(int datatype) {
    /* int ch = 1;

       if (datatype & GHOST_BINCRS_DT_FLOAT)
       ch *= 2;

       if (datatype & GHOST_BINCRS_DT_REAL)
       ch *= 2;

#ifdef AVX
ch *= 2;
#endif

#ifdef MIC
ch *= 4;
#if (!GHOST_HAVE_LONGIDX)
ch *= 2;
#endif
#endif

#if defined (CUDA)
ch = 256;
#endif

return ch;*/
    UNUSED(datatype);
    return 32;
}

