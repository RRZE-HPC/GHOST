#include "ghost/sell.h"
#include "ghost/core.h"
#include "ghost/crs.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/sparsemat.h"
#include "ghost/context.h"
#include "ghost/bincrs.h"
#include "ghost/log.h"
#include "ghost/machine.h"

#include <libgen.h>
#include <string.h>
#include <stdlib.h>

#ifdef GHOST_HAVE_OPENMP
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

#ifdef GHOST_HAVE_SSE
static ghost_error_t (*SELL_kernels_SSE[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list argp) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_SSE,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};
#endif

#ifdef GHOST_HAVE_AVX
static ghost_error_t (*SELL_kernels_AVX[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list argp) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_AVX,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

static ghost_error_t (*SELL_kernels_AVX_32[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list argp) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_AVX_32_rich,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};
#endif

#ifdef GHOST_HAVE_MIC
static ghost_error_t (*SELL_kernels_MIC_16[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list argp) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_MIC_16,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

static ghost_error_t (*SELL_kernels_MIC_32[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list argp) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_MIC_32,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};
#endif

static ghost_error_t (*SELL_kernels_plain[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list argp) = 
{{&ss_SELL_kernel_plain,&sd_SELL_kernel_plain,&sc_SELL_kernel_plain,&sz_SELL_kernel_plain},
    {&ds_SELL_kernel_plain,&dd_SELL_kernel_plain,&dc_SELL_kernel_plain,&dz_SELL_kernel_plain},
    {&cs_SELL_kernel_plain,&cd_SELL_kernel_plain,&cc_SELL_kernel_plain,&cz_SELL_kernel_plain},
    {&zs_SELL_kernel_plain,&zd_SELL_kernel_plain,&zc_SELL_kernel_plain,&zz_SELL_kernel_plain}};

#ifdef GHOST_HAVE_CUDA
static ghost_error_t (*SELL_kernels_CU[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list argp) = 
{{&ss_SELL_kernel_CU,&sd_SELL_kernel_CU,&sc_SELL_kernel_CU,&sz_SELL_kernel_CU},
    {&ds_SELL_kernel_CU,&dd_SELL_kernel_CU,&dc_SELL_kernel_CU,&dz_SELL_kernel_CU},
    {&cs_SELL_kernel_CU,&cd_SELL_kernel_CU,&cc_SELL_kernel_CU,&cz_SELL_kernel_CU},
    {&zs_SELL_kernel_CU,&zd_SELL_kernel_CU,&zc_SELL_kernel_CU,&zz_SELL_kernel_CU}};
#endif

static ghost_error_t (*SELL_fromCRS_funcs[4]) (ghost_sparsemat_t *, ghost_sparsemat_t *) = 
{&s_SELL_fromCRS, &d_SELL_fromCRS, &c_SELL_fromCRS, &z_SELL_fromCRS}; 

static ghost_error_t (*SELL_stringify_funcs[4]) (ghost_sparsemat_t *, char **, int) = 
{&s_SELL_stringify, &d_SELL_stringify, &c_SELL_stringify, &z_SELL_stringify}; 

static void SELL_printInfo(ghost_sparsemat_t *mat, char **str);
static const char * SELL_formatName(ghost_sparsemat_t *mat);
static ghost_idx_t SELL_rowLen (ghost_sparsemat_t *mat, ghost_idx_t i);
static size_t SELL_byteSize (ghost_sparsemat_t *mat);
static ghost_error_t SELL_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crs);
static ghost_error_t SELL_stringify(ghost_sparsemat_t *mat, char **str, int dense);
static ghost_error_t SELL_split(ghost_sparsemat_t *mat);
static ghost_error_t SELL_permute(ghost_sparsemat_t *, ghost_idx_t *, ghost_idx_t *);
static ghost_error_t SELL_upload(ghost_sparsemat_t *mat);
static ghost_error_t SELL_fromBin(ghost_sparsemat_t *mat, char *);
static ghost_error_t SELL_fromRowFunc(ghost_sparsemat_t *mat, ghost_idx_t maxrowlen, int base, ghost_sparsemat_fromRowFunc_t func, ghost_sparsemat_fromRowFunc_flags_t flags);
static void SELL_free(ghost_sparsemat_t *mat);
static ghost_error_t SELL_kernel_plain (ghost_sparsemat_t *mat, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
static int ghost_selectSellChunkHeight(int datatype);
#ifdef GHOST_HAVE_CUDA
static ghost_error_t SELL_kernel_CU (ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t flags, va_list argp);
#endif
#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, int options);
#endif

ghost_error_t ghost_sell_init(ghost_sparsemat_t *mat)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->data,sizeof(ghost_sell_t)),err,ret);
    DEBUG_LOG(1,"Setting functions for SELL matrix");
    if (!(mat->traits->flags & (GHOST_SPARSEMAT_HOST | GHOST_SPARSEMAT_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting matrix placement");
        ghost_type_t ghost_type;
        GHOST_CALL_GOTO(ghost_type_get(&ghost_type),err,ret);
        if (ghost_type == GHOST_TYPE_CUDA) {
            mat->traits->flags |= GHOST_SPARSEMAT_DEVICE;
        } else {
            mat->traits->flags |= GHOST_SPARSEMAT_HOST;
        }
    }
    //TODO is it reasonable that a matrix has HOST&DEVICE?

    ghost_type_t ghost_type;
    GHOST_CALL_RETURN(ghost_type_get(&ghost_type));

    mat->upload = &SELL_upload;
    mat->fromFile = &SELL_fromBin;
    mat->fromRowFunc = &SELL_fromRowFunc;
    mat->auxString = &SELL_printInfo;
    mat->formatName = &SELL_formatName;
    mat->rowLen     = &SELL_rowLen;
    mat->byteSize   = &SELL_byteSize;
    mat->spmv     = &SELL_kernel_plain;
    mat->fromCRS    = &SELL_fromCRS;
    mat->string    = &SELL_stringify;
    mat->split = &SELL_split;
    mat->permute = &SELL_permute;
#ifdef VSX_INTR
    mat->spmv = &SELL_kernel_VSX;
#endif
#ifdef GHOST_HAVE_CUDA
    if (ghost_type == GHOST_TYPE_CUDA) {
        mat->spmv   = &SELL_kernel_CU;
    }
#endif
    mat->destroy  = &SELL_free;

    SELL(mat)->val = NULL;
    SELL(mat)->col = NULL;
    SELL(mat)->chunkMin = NULL;
    SELL(mat)->chunkLen = NULL;
    SELL(mat)->chunkLenPadded = NULL;
    SELL(mat)->rowLen = NULL;
    SELL(mat)->rowLenPadded = NULL;
    SELL(mat)->chunkStart = NULL;
    SELL(mat)->maxRowLen = 0;
    SELL(mat)->nMaxRows = 0;
    SELL(mat)->variance = 0.;
    SELL(mat)->deviation = 0.;
    SELL(mat)->cv = 0.;
    SELL(mat)->beta = 0;
    SELL(mat)->cumat = NULL;


    if (mat->traits->aux == NULL) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->traits->aux,sizeof(ghost_sell_aux_t)),err,ret);
        *((ghost_sell_aux_t *)(mat->traits->aux)) = GHOST_SELL_AUX_INITIALIZER;
    }
    SELL(mat)->T = ((ghost_sell_aux_t *)(mat->traits->aux))->T;
    SELL(mat)->chunkHeight = ((ghost_sell_aux_t *)(mat->traits->aux))->C;

    if (SELL(mat)->chunkHeight == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
        SELL(mat)->chunkHeight = PAD(mat->nrows,GHOST_PAD_MAX);
    } else if (SELL(mat)->chunkHeight == GHOST_SELL_CHUNKHEIGHT_AUTO){
        SELL(mat)->chunkHeight = ghost_selectSellChunkHeight(mat->traits->datatype);
    }
    mat->nrowsPadded = PAD(mat->nrows,SELL(mat)->chunkHeight);

    goto out;
err:
    free(mat->data); mat->data = NULL;

out:
    return ret;
}

static ghost_error_t SELL_permute(ghost_sparsemat_t *mat , ghost_idx_t *perm, ghost_idx_t *invPerm)
{
    UNUSED(mat);
    UNUSED(perm);
    UNUSED(invPerm);
    ERROR_LOG("SELL->permute() not implemented");
    return GHOST_ERR_NOT_IMPLEMENTED;

}
static void SELL_printInfo(ghost_sparsemat_t *mat, char **str)
{
    ghost_line_string(str,"Max row length (# rows)",NULL,"%d (%d)",SELL(mat)->maxRowLen,SELL(mat)->nMaxRows);
    ghost_line_string(str,"Chunk height (C)",NULL,"%d",SELL(mat)->chunkHeight);
    ghost_line_string(str,"Chunk occupancy (beta)",NULL,"%f",SELL(mat)->beta);
    ghost_line_string(str,"Row length variance",NULL,"%f",SELL(mat)->variance);
    ghost_line_string(str,"Row length standard deviation",NULL,"%f",SELL(mat)->deviation);
    ghost_line_string(str,"Row length coefficient of variation",NULL,"%f",SELL(mat)->cv);
    ghost_line_string(str,"Threads per row (T)",NULL,"%d",SELL(mat)->T);
}

static const char * SELL_formatName(ghost_sparsemat_t *mat)
{
    // TODO format SELL-C-sigma
    UNUSED(mat);
    return "SELL";
}

static ghost_idx_t SELL_rowLen (ghost_sparsemat_t *mat, ghost_idx_t i)
{
    if (mat && i<mat->nrows) {
        return SELL(mat)->rowLen[i];
    }

    return 0;
}

/*static ghost_dt SELL_entry (ghost_sparsemat_t *mat, ghost_idx_t i, ghost_idx_t j)
  {
  ghost_idx_t e;

  if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE)
  i = mat->permutation->perm[i];
  if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE_COLS)
  j = mat->permutation->perm[j];

  for (e=SELL(mat)->chunkStart[i/SELL_LEN]+i%SELL_LEN; 
  e<SELL(mat)->chunkStart[i/SELL_LEN+1]; 
  e+=SELL_LEN) {
  if (SELL(mat)->col[e] == j)
  return SELL(mat)->val[e];
  }
  return 0.;
  }*/

static size_t SELL_byteSize (ghost_sparsemat_t *mat)
{
    if (mat->data == NULL) {
        return 0;
    }
    return (size_t)((mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_nnz_t) + 
            mat->nEnts*(sizeof(ghost_idx_t)+mat->elSize));
}

/*static int compareNZEPerRow( const void* a, const void* b ) 
{
    return  ((ghost_sorting_t*)b)->nEntsInRow - ((ghost_sorting_t*)a)->nEntsInRow;
}*/

static ghost_error_t SELL_fromRowFunc(ghost_sparsemat_t *mat, ghost_idx_t maxrowlen, int base, ghost_sparsemat_fromRowFunc_t func, ghost_sparsemat_fromRowFunc_flags_t flags)
{
    ghost_error_t ret = GHOST_SUCCESS;
    //   WARNING_LOG("SELL-%d-%d from row func",SELL(mat)->chunkHeight,SELL(mat)->scope);
    UNUSED(base);
    UNUSED(flags);
    int nprocs = 1;
    int me;

    char *tmpval = NULL;
    ghost_idx_t *tmpcol = NULL;    
    ghost_idx_t i,col,row;
    ghost_idx_t chunk,j;

    ghost_sorting_t* rowSort = NULL;

    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);


    ghost_idx_t nChunks = mat->nrowsPadded/SELL(mat)->chunkHeight;
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkStart, (nChunks+1)*sizeof(ghost_nnz_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkMin, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLen, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLenPadded, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLen, (mat->nrowsPadded)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_idx_t)),err,ret);
    SELL(mat)->chunkStart[0] = 0;

    ghost_idx_t maxRowLenInChunk = 0;
    ghost_idx_t maxRowLen = 0;
    ghost_nnz_t nEnts = 0, nnz = 0;
    SELL(mat)->chunkStart[0] = 0;

    GHOST_CALL_GOTO(ghost_sparsemat_fromfunc_common(mat,maxrowlen,func,flags),err,ret);
    /*if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
#pragma omp parallel private(i,tmpval,tmpcol)
        { 
            GHOST_CALL(ghost_malloc((void **)&tmpval,maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,maxrowlen*sizeof(ghost_idx_t)),ret);
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nChunks; chunk++ ) {
                for (i=0; i<SELL(mat)->chunkHeight; i++) {
                    ghost_idx_t row = chunk*SELL(mat)->chunkHeight+i;

                    if (row < mat->nrows) {
                        func(mat->context->lfRow[me]+row,&SELL(mat)->rowLen[row],tmpcol,tmpval);
                    } else {
                        SELL(mat)->rowLen[row] = 0;
                    }
                }
            }

            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        if (ret != GHOST_SUCCESS) {
            goto err;
        }

        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->permutation->perm,mat->nrows*sizeof(ghost_idx_t)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->permutation->invPerm,mat->nrows*sizeof(ghost_idx_t)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&rowSort,mat->nrows * sizeof(ghost_sorting_t)),err,ret);

        ghost_idx_t c;
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
            (mat->permutation->invPerm)[i] = rowSort[i].row;
            (mat->permutation->perm)[rowSort[i].row] = i;
        }

#pragma omp parallel private(maxRowLenInChunk,i,tmpcol,tmpval) reduction (+:nEnts,nnz)
        { 
            maxRowLenInChunk = 0; 
            GHOST_CALL(ghost_malloc((void **)&tmpval,maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,maxrowlen*sizeof(ghost_idx_t)),ret);
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nChunks; chunk++ ) {
                for (i=0; i<SELL(mat)->chunkHeight; i++) {
                    ghost_idx_t row = chunk*SELL(mat)->chunkHeight+i;

                    if (row < mat->nrows) {
                        func(mat->context->lfRow[me]+mat->permutation->invPerm[row],&SELL(mat)->rowLen[row],tmpcol,tmpval);
                    } else {
                        SELL(mat)->rowLen[row] = 0;
                    }

                    SELL(mat)->rowLenPadded[row] = PAD(SELL(mat)->rowLen[row],SELL(mat)->T);

                    nnz += SELL(mat)->rowLen[row];
                    maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[row]);
                }
#pragma omp critical
                maxRowLen = MAX(maxRowLen,maxRowLenInChunk);
                SELL(mat)->chunkLen[chunk] = maxRowLenInChunk;
                SELL(mat)->chunkLenPadded[chunk] = PAD(maxRowLenInChunk,SELL(mat)->T);
                nEnts += SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
                maxRowLenInChunk = 0;
            }

            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        if (ret != GHOST_SUCCESS) {
            goto err;
        }
    } else {
*/
#pragma omp parallel private(maxRowLenInChunk,i,tmpval,tmpcol) reduction (+:nEnts,nnz) 
        {
            int funcret = 0;
            maxRowLenInChunk = 0; 
            GHOST_CALL(ghost_malloc((void **)&tmpval,maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,maxrowlen*sizeof(ghost_idx_t)),ret);
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nChunks; chunk++ ) {
                for (i=0; i<SELL(mat)->chunkHeight; i++) {
                    ghost_idx_t row = chunk*SELL(mat)->chunkHeight+i;

                    if (row < mat->nrows) {
                        if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
                            if (mat->permutation->scope == GHOST_PERMUTATION_GLOBAL) {
                                funcret = func(mat->permutation->invPerm[mat->context->lfRow[me]+row],&SELL(mat)->rowLen[row],tmpcol,tmpval);
                            } else {
                                funcret = func(mat->context->lfRow[me]+mat->permutation->invPerm[row],&SELL(mat)->rowLen[row],tmpcol,tmpval);
                            }
                        } else {
                            funcret = func(mat->context->lfRow[me]+row,&SELL(mat)->rowLen[row],tmpcol,tmpval);
                        }
                        if (funcret) {
                            ERROR_LOG("Matrix construction function returned error");
                            ret = GHOST_ERR_UNKNOWN;
                        }
                    } else {
                        SELL(mat)->rowLen[row] = 0;
                    }

                    SELL(mat)->rowLenPadded[row] = PAD(SELL(mat)->rowLen[row],SELL(mat)->T);

                    nnz += SELL(mat)->rowLen[row];
                    maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[row]);
                }
#pragma omp critical
                maxRowLen = MAX(maxRowLen,maxRowLenInChunk);
                SELL(mat)->chunkLen[chunk] = maxRowLenInChunk;
                SELL(mat)->chunkLenPadded[chunk] = PAD(maxRowLenInChunk,SELL(mat)->T);
                nEnts += SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
                maxRowLenInChunk = 0;
            }

            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        if (ret != GHOST_SUCCESS) {
            goto err;
        }
 //   }

    for( chunk = 0; chunk < nChunks; chunk++ ) {
        SELL(mat)->chunkStart[chunk+1] = SELL(mat)->chunkStart[chunk] + SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
    }

    SELL(mat)->maxRowLen = maxRowLen; 
    mat->nEnts = nEnts;
    mat->nnz = nnz;
    SELL(mat)->beta = mat->nnz*1.0/(double)mat->nEnts;

    GHOST_CALL_GOTO(ghost_malloc_align((void **)&SELL(mat)->val,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&SELL(mat)->col,sizeof(ghost_idx_t)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);


#pragma omp parallel for schedule(runtime) private (i,j)
    for (chunk = 0; chunk < nChunks; chunk++) {
        for (j=0; j<SELL(mat)->chunkLenPadded[chunk]; j++) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = 0;
                memset(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i)],0,mat->elSize);
            }
        }
    }


#pragma omp parallel private(i,col,row,tmpval,tmpcol)
    {
        int funcret = 0;
        GHOST_CALL(ghost_malloc((void **)&tmpval,SELL(mat)->chunkHeight*maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,SELL(mat)->chunkHeight*maxrowlen*sizeof(ghost_idx_t)),ret);
        memset(tmpval,0,mat->elSize*maxrowlen*SELL(mat)->chunkHeight);
        memset(tmpcol,0,sizeof(ghost_idx_t)*maxrowlen*SELL(mat)->chunkHeight);
#pragma omp for schedule(runtime)
        for( chunk = 0; chunk < nChunks; chunk++ ) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                row = chunk*SELL(mat)->chunkHeight+i;

                if (row < mat->nrows) {
                    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
                        if (mat->permutation->scope == GHOST_PERMUTATION_GLOBAL) {
                            funcret = func(mat->permutation->invPerm[mat->context->lfRow[me]+row],&SELL(mat)->rowLen[row],&tmpcol[maxrowlen*i],&tmpval[maxrowlen*i*mat->elSize]);
                        } else {
                            funcret = func(mat->context->lfRow[me]+mat->permutation->invPerm[row],&SELL(mat)->rowLen[row],&tmpcol[maxrowlen*i],&tmpval[maxrowlen*i*mat->elSize]);
                        }
                    } else {
                        funcret = func(mat->context->lfRow[me]+row,&SELL(mat)->rowLen[row],&tmpcol[maxrowlen*i],&tmpval[maxrowlen*i*mat->elSize]);
                    }
                }
                if (funcret) {
                    ERROR_LOG("Matrix construction function returned error");
                    ret = GHOST_ERR_UNKNOWN;
                }

                for (col = 0; col<SELL(mat)->chunkLenPadded[chunk]; col++) {
                    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
                        if ((tmpcol[i*maxrowlen+col] >= mat->context->lfRow[me]) && (tmpcol[i*maxrowlen+col] < (mat->context->lfRow[me]+mat->nrows))) { // local entry: copy with permutation
                            memcpy(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&tmpval[mat->elSize*(i*maxrowlen+col)],mat->elSize);
                            if (mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*maxrowlen+col];
                            } else {
                                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = mat->permutation->perm[tmpcol[i*maxrowlen+col]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                            }
                        } else { // remote entry: copy without permutation
                            memcpy(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&tmpval[mat->elSize*(i*maxrowlen+col)],mat->elSize);
                            SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*maxrowlen+col];
                        }
                    } else {
                        memcpy(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&tmpval[mat->elSize*(i*maxrowlen+col)],mat->elSize);
                        SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*maxrowlen+col];
                    }
                }
                if (!(mat->traits->flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
                    // sort rows by ascending column indices
                    ghost_sparsemat_sortrow(&SELL(mat)->col[SELL(mat)->chunkStart[chunk]+i],&SELL(mat)->val[(SELL(mat)->chunkStart[chunk]+i)*mat->elSize],mat->elSize,SELL(mat)->rowLen[row],SELL(mat)->chunkHeight);
                }
                ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&SELL(mat)->col[SELL(mat)->chunkStart[chunk]+i],SELL(mat)->rowLen[row],SELL(mat)->chunkHeight);
                }
            memset(tmpval,0,mat->elSize*maxrowlen*SELL(mat)->chunkHeight);
            memset(tmpcol,0,sizeof(ghost_idx_t)*maxrowlen*SELL(mat)->chunkHeight);
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }

#ifdef GHOST_HAVE_MPI
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,&mat->lowerBandwidth,1,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,&mat->upperBandwidth,1,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,mat->nzDist,2*mat->context->gnrows-1,ghost_mpi_dt_idx,MPI_SUM,mat->context->mpicomm),err,ret);
#endif
    mat->bandwidth = mat->lowerBandwidth+mat->upperBandwidth+1;

    if (ret != GHOST_SUCCESS) {
        goto err;
    }

    if (!(mat->context->flags & GHOST_CONTEXT_REDUNDANT)) {
#ifdef GHOST_HAVE_MPI

        mat->context->lnEnts[me] = mat->nEnts;

        ghost_nnz_t nents;
        nents = mat->context->lnEnts[me];
        MPI_CALL_GOTO(MPI_Allgather(&nents,1,ghost_mpi_dt_nnz,mat->context->lnEnts,1,ghost_mpi_dt_nnz,mat->context->mpicomm),err,ret);

        for (i=0; i<nprocs; i++) {
            mat->context->lfEnt[i] = 0;
        } 

        for (i=1; i<nprocs; i++) {
            mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
        } 

        GHOST_CALL_GOTO(mat->split(mat),err,ret);
#endif
    }
#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPARSEMAT_HOST))
        mat->upload(mat);
#endif

    goto out;
err:
    free(SELL(mat)->val); SELL(mat)->val = NULL;
    free(SELL(mat)->col); SELL(mat)->col = NULL;
    free(SELL(mat)->chunkMin); SELL(mat)->chunkMin = NULL;
    free(SELL(mat)->chunkLen); SELL(mat)->chunkLen = NULL;
    free(SELL(mat)->chunkLenPadded); SELL(mat)->chunkLenPadded = NULL;
    free(SELL(mat)->rowLen); SELL(mat)->rowLen = NULL;
    free(SELL(mat)->rowLenPadded); SELL(mat)->rowLenPadded = NULL;
    free(SELL(mat)->chunkStart); SELL(mat)->chunkStart = NULL;
    SELL(mat)->maxRowLen = 0;
    SELL(mat)->nMaxRows = 0;
    SELL(mat)->variance = 0.;
    SELL(mat)->deviation = 0.;
    SELL(mat)->cv = 0.;
    SELL(mat)->beta = 0;
    mat->nEnts = 0;
    mat->nnz = 0;
    free(mat->permutation->perm); mat->permutation->perm = NULL;
    free(mat->permutation->invPerm); mat->permutation->invPerm = NULL;

out:
    free(rowSort); rowSort = NULL;
    free(tmpcol); tmpcol = NULL;
    free(tmpval); tmpval = NULL;
    return ret;

}

static ghost_error_t SELL_split(ghost_sparsemat_t *mat)
{
    if (!mat) {
        ERROR_LOG("Matrix is NULL");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_error_t ret = GHOST_SUCCESS;


    ghost_sell_t *fullSELL = SELL(mat);
    ghost_sell_t *localSELL = NULL, *remoteSELL = NULL;
    DEBUG_LOG(1,"Splitting the SELL matrix into a local and remote part");
    ghost_idx_t j;
    ghost_idx_t i;
    int me;
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));

    ghost_nnz_t lnEnts_l, lnEnts_r;
    ghost_nnz_t current_l, current_r;


    ghost_idx_t chunk;
    ghost_idx_t idx, row;

    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED) {
        ghost_context_setupCommunication(mat->context,fullSELL->col);
    }

    ghost_sparsemat_create(&(mat->localPart),mat->context,&mat->traits[0],1);
    localSELL = mat->localPart->data;
    mat->localPart->traits->symmetry = mat->traits->symmetry;

    ghost_sparsemat_create(&(mat->remotePart),mat->context,&mat->traits[0],1);
    remoteSELL = mat->remotePart->data; 

    localSELL->T = fullSELL->T;
    remoteSELL->T = fullSELL->T;

    ghost_idx_t nChunks = mat->nrowsPadded/fullSELL->chunkHeight;
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkStart, (nChunks+1)*sizeof(ghost_nnz_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkMin, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkLen, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkLenPadded, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->rowLen, (mat->nrowsPadded)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_idx_t)),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkStart, (nChunks+1)*sizeof(ghost_nnz_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkMin, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkLen, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkLenPadded, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->rowLen, (mat->nrowsPadded)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_idx_t)),err,ret);

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

    if (!(mat->traits->flags & GHOST_SPARSEMAT_NOT_STORE_SPLIT)) { // split computation

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
                        localSELL->rowLenPadded[row] = PAD(localSELL->rowLen[row],localSELL->T);
                        remoteSELL->rowLenPadded[row] = PAD(remoteSELL->rowLen[row],remoteSELL->T);
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

            localSELL->chunkLenPadded[chunk] = PAD(localSELL->chunkLen[chunk],localSELL->T);
            remoteSELL->chunkLenPadded[chunk] = PAD(remoteSELL->chunkLen[chunk],remoteSELL->T);

        }



        /*
           for (i=0; i<fullSELL->nEnts;i++) {
           if (fullSELL->col[i]<mat->context->lnrows[me]) lnEnts_l++;
           }
           lnEnts_r = mat->context->lnEnts[me]-lnEnts_l;*/


        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->val,lnEnts_l*mat->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->col,lnEnts_l*sizeof(ghost_idx_t)),err,ret); 

        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->val,lnEnts_r*mat->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->col,lnEnts_r*sizeof(ghost_idx_t)),err,ret); 

        mat->localPart->nrows = mat->nrows;
        mat->localPart->nrowsPadded = mat->nrowsPadded;
        mat->localPart->nEnts = lnEnts_l;
        localSELL->chunkHeight = fullSELL->chunkHeight;

        mat->remotePart->nrows = mat->nrows;
        mat->remotePart->nrowsPadded = mat->nrowsPadded;
        mat->remotePart->nEnts = lnEnts_r;
        remoteSELL->chunkHeight = fullSELL->chunkHeight;

#pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < mat->localPart->nrowsPadded/localSELL->chunkHeight; chunk++) {
            for (i=0; i<localSELL->chunkLenPadded[chunk]; i++) {
                for (j=0; j<localSELL->chunkHeight; j++) {
                    idx = localSELL->chunkStart[chunk]+i*localSELL->chunkHeight+j;
                    memset(&((char *)(localSELL->val))[idx*mat->elSize],0,mat->elSize);
                    localSELL->col[idx] = 0;
                }
            }
        }

#pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < mat->remotePart->nrowsPadded/remoteSELL->chunkHeight; chunk++) {
            for (i=0; i<remoteSELL->chunkLenPadded[chunk]; i++) {
                for (j=0; j<remoteSELL->chunkHeight; j++) {
                    idx = remoteSELL->chunkStart[chunk]+i*remoteSELL->chunkHeight+j;
                    memset(&((char *)(remoteSELL->val))[idx*mat->elSize],0,mat->elSize);
                    remoteSELL->col[idx] = 0;
                }
            }
        }

        current_l = 0;
        current_r = 0;
        ghost_idx_t col_l[fullSELL->chunkHeight], col_r[fullSELL->chunkHeight];

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
                                ghost_idx_t lidx = localSELL->chunkStart[chunk]+col_l[j]*localSELL->chunkHeight+j;
                                localSELL->col[lidx] = fullSELL->col[idx];
                                memcpy(&localSELL->val[lidx*mat->elSize],&fullSELL->val[idx*mat->elSize],mat->elSize);
                                current_l++;
                            }
                            col_l[j]++;
                        }
                        else{
                            if (col_r[j] < remoteSELL->rowLen[row]) {
                                ghost_idx_t ridx = remoteSELL->chunkStart[chunk]+col_r[j]*remoteSELL->chunkHeight+j;
                                remoteSELL->col[ridx] = fullSELL->col[idx];
                                memcpy(&remoteSELL->val[ridx*mat->elSize],&fullSELL->val[idx*mat->elSize],mat->elSize);
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
    if (!(mat->traits->flags & GHOST_SPARSEMAT_HOST)) {
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

    goto out;
err:
    mat->localPart->destroy(mat->localPart); mat->localPart = NULL;
    mat->remotePart->destroy(mat->remotePart); mat->remotePart = NULL;

out:
    return ret;
}

static ghost_error_t SELL_fromBin(ghost_sparsemat_t *mat, char *matrixPath)
{
    DEBUG_LOG(1,"Creating SELL matrix from binary file");
    ghost_error_t ret = GHOST_SUCCESS;

    int nprocs = 1;
    int me;
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, mat->context->mpicomm));
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));

    ghost_idx_t *rpt = NULL;
    ghost_idx_t *tmpcol = NULL;
    char *tmpval = NULL;

    ghost_idx_t i;
    ghost_idx_t chunk,j;

    ghost_sparsemat_fromfile_common(mat,matrixPath,&rpt);
    
    mat->nEnts = 0;
    mat->nrowsPadded = PAD(mat->nrows,SELL(mat)->chunkHeight);

    ghost_idx_t nChunks =  mat->nrowsPadded/SELL(mat)->chunkHeight;
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkStart, (nChunks+1)*sizeof(ghost_nnz_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkMin, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLen, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLenPadded, (nChunks)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLen, (mat->nrowsPadded)*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_idx_t)),err,ret);

    ghost_idx_t maxRowLenInChunk = 0;
    ghost_idx_t minRowLenInChunk = INT_MAX;

    SELL(mat)->maxRowLen = 0;
    SELL(mat)->chunkStart[0] = 0;    

    DEBUG_LOG(1,"Extracting row lenghts");
    for( chunk = 0; chunk < nChunks; chunk++ ) {
        for (i=0; i<SELL(mat)->chunkHeight; i++) {
            ghost_idx_t row = chunk*SELL(mat)->chunkHeight+i;
            if (row < mat->nrows) {
                SELL(mat)->rowLen[row] = rpt[row+1]-rpt[row];
            } else {
                SELL(mat)->rowLen[row] = 0;
            }
            SELL(mat)->rowLenPadded[row] = PAD(SELL(mat)->rowLen[row],SELL(mat)->T);

            maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[row]);
            minRowLenInChunk = MIN(minRowLenInChunk,SELL(mat)->rowLen[row]);
        }


        SELL(mat)->maxRowLen = MAX(SELL(mat)->maxRowLen,maxRowLenInChunk);
        SELL(mat)->chunkLen[chunk] = maxRowLenInChunk;
        SELL(mat)->chunkMin[chunk] = minRowLenInChunk;
        SELL(mat)->chunkLenPadded[chunk] = PAD(maxRowLenInChunk,SELL(mat)->T);
        mat->nEnts += SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
        SELL(mat)->chunkStart[chunk+1] = mat->nEnts;
        maxRowLenInChunk = 0;
        minRowLenInChunk = INT_MAX;
    }

    mat->context->lnEnts[me] = mat->nEnts;
    SELL(mat)->beta = mat->nnz*1.0/(double)mat->nEnts;

#ifdef GHOST_HAVE_MPI
    ghost_nnz_t nents;
    nents = mat->context->lnEnts[me];
    MPI_CALL_GOTO(MPI_Allgather(&nents,1,ghost_mpi_dt_nnz,mat->context->lnEnts,1,ghost_mpi_dt_nnz,mat->context->mpicomm),err,ret);
#endif

    DEBUG_LOG(1,"SELL matrix has %"PRIDX" (padded to %"PRIDX") rows, %"PRIDX" cols and %"PRNNZ" nonzeros and %"PRNNZ" entries",mat->nrows,mat->nrowsPadded,mat->ncols,mat->nnz,mat->context->lnEnts[me]);

    GHOST_CALL_GOTO(ghost_malloc_align((void **)&SELL(mat)->val,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&SELL(mat)->col,sizeof(ghost_idx_t)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);

#pragma omp parallel for schedule(runtime) private (i,j)
    for (chunk = 0; chunk < nChunks; chunk++) {
        for (j=0; j<SELL(mat)->chunkLenPadded[chunk]; j++) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = 0;
                memset(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i)],0,mat->elSize);
            }
        }
    }

    WARNING_LOG("Memory usage may be high because read-in of CRS data is done at once and not chunk-wise");
    GHOST_CALL_GOTO(ghost_malloc((void **)&tmpcol,mat->nnz*sizeof(ghost_idx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&tmpval,mat->nnz*mat->elSize),err,ret);
    GHOST_CALL_GOTO(ghost_bincrs_col_read(tmpcol, matrixPath, mat->context->lfRow[me], mat->nrows, mat->permutation, mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS),err,ret);
    GHOST_CALL_GOTO(ghost_bincrs_val_read(tmpval, mat->traits->datatype, matrixPath,  mat->context->lfRow[me], mat->nrows, mat->permutation),err,ret);

    ghost_idx_t row = 0;
    for (chunk = 0; chunk < nChunks; chunk++) {
        ghost_idx_t col;
        ghost_idx_t *curRowCols;
        char * curRowVals;
        for (i=0; (i<SELL(mat)->chunkHeight) && (row < mat->nrows); i++, row++) {
            curRowCols = &tmpcol[rpt[row]];
            curRowVals = &tmpval[rpt[row]*mat->elSize];
            for (col=0; col<SELL(mat)->rowLen[row]; col++) {
                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = curRowCols[col];
                memcpy(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&curRowVals[col*mat->elSize],mat->elSize);
            }
            if (!(mat->traits->flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
                // sort rows by ascending column indices
                ghost_sparsemat_sortrow(&SELL(mat)->col[SELL(mat)->chunkStart[chunk]+i],&SELL(mat)->val[(SELL(mat)->chunkStart[chunk]+i)*mat->elSize],mat->elSize,SELL(mat)->rowLen[row],SELL(mat)->chunkHeight);
            }
            ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&SELL(mat)->col[SELL(mat)->chunkStart[chunk]+i],SELL(mat)->rowLen[row],SELL(mat)->chunkHeight);
        }

    }
#ifdef GHOST_HAVE_MPI
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,&mat->lowerBandwidth,1,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,&mat->upperBandwidth,1,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,mat->nzDist,2*mat->context->gnrows-1,ghost_mpi_dt_idx,MPI_SUM,mat->context->mpicomm),err,ret);
#endif
    mat->bandwidth = mat->lowerBandwidth+mat->upperBandwidth+1;

    mat->split(mat);


#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPARSEMAT_HOST))
        mat->upload(mat);
#endif



    DEBUG_LOG(1,"SELL matrix successfully created");
    goto out;

err:
    free(SELL(mat)->val); SELL(mat)->val = NULL;
    free(SELL(mat)->col); SELL(mat)->col = NULL;
    free(SELL(mat)->chunkMin); SELL(mat)->chunkMin = NULL;
    free(SELL(mat)->chunkLen); SELL(mat)->chunkLen = NULL;
    free(SELL(mat)->chunkLenPadded); SELL(mat)->chunkLenPadded = NULL;
    free(SELL(mat)->rowLen); SELL(mat)->rowLen = NULL;
    free(SELL(mat)->rowLenPadded); SELL(mat)->rowLenPadded = NULL;
    free(SELL(mat)->chunkStart); SELL(mat)->chunkStart = NULL;
    free(mat->permutation->perm); mat->permutation->perm = NULL;
    free(mat->permutation->invPerm); mat->permutation->invPerm = NULL;

out:
    free(tmpcol); tmpcol = NULL;
    free(tmpval); tmpval = NULL;
    free(rpt); rpt = NULL;

    return ret;
}

static ghost_error_t SELL_stringify(ghost_sparsemat_t *mat, char **str, int dense)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,mat->traits->datatype));

    return SELL_stringify_funcs[dtIdx](mat, str, dense);
}

static ghost_error_t SELL_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crs)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,mat->traits->datatype));
    return SELL_fromCRS_funcs[dtIdx](mat,crs);
}

static ghost_error_t SELL_upload(ghost_sparsemat_t* mat) 
{
#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPARSEMAT_HOST)) {
        DEBUG_LOG(1,"Creating matrix on CUDA device");
        GHOST_CALL_RETURN(ghost_malloc((void **)&SELL(mat)->cumat,sizeof(ghost_cu_sell_t)));
        ghost_cu_malloc((void **)&SELL(mat)->cumat->rowLen,(mat->nrows)*sizeof(ghost_idx_t));
        ghost_cu_malloc((void **)&SELL(mat)->cumat->rowLenPadded,(mat->nrows)*sizeof(ghost_idx_t));
        ghost_cu_malloc((void **)&SELL(mat)->cumat->col,(mat->nEnts)*sizeof(ghost_idx_t));
        ghost_cu_malloc((void **)&SELL(mat)->cumat->val,(mat->nEnts)*mat->elSize);
        ghost_cu_malloc((void **)&SELL(mat)->cumat->chunkStart,(mat->nrowsPadded/SELL(mat)->chunkHeight+1)*sizeof(ghost_nnz_t));
        ghost_cu_malloc((void **)&SELL(mat)->cumat->chunkLen,(mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_idx_t));

        ghost_cu_upload(SELL(mat)->cumat->rowLen, SELL(mat)->rowLen, mat->nrows*sizeof(ghost_idx_t));
        ghost_cu_upload(SELL(mat)->cumat->rowLenPadded, SELL(mat)->rowLenPadded, mat->nrows*sizeof(ghost_idx_t));
        ghost_cu_upload(SELL(mat)->cumat->col, SELL(mat)->col, mat->nEnts*sizeof(ghost_idx_t));
        ghost_cu_upload(SELL(mat)->cumat->val, SELL(mat)->val, mat->nEnts*mat->elSize);
        ghost_cu_upload(SELL(mat)->cumat->chunkStart, SELL(mat)->chunkStart, (mat->nrowsPadded/SELL(mat)->chunkHeight+1)*sizeof(ghost_nnz_t));
        ghost_cu_upload(SELL(mat)->cumat->chunkLen, SELL(mat)->chunkLen, (mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_idx_t));
    }
#else
    if (mat->traits->flags & GHOST_SPARSEMAT_DEVICE) {
        ERROR_LOG("Device matrix cannot be created without CUDA");
        return GHOST_ERR_CUDA;
    }
#endif
    return GHOST_SUCCESS;
}


static void SELL_free(ghost_sparsemat_t *mat)
{
    if (!mat) {
        return;
    }

    if (mat->data) {
#ifdef GHOST_HAVE_CUDA
        if (mat->traits->flags & GHOST_SPARSEMAT_DEVICE) {
            ghost_cu_free(SELL(mat)->cumat->rowLen);
            ghost_cu_free(SELL(mat)->cumat->rowLenPadded);
            ghost_cu_free(SELL(mat)->cumat->col);
            ghost_cu_free(SELL(mat)->cumat->val);
            ghost_cu_free(SELL(mat)->cumat->chunkStart);
            ghost_cu_free(SELL(mat)->cumat->chunkLen);
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

static ghost_error_t SELL_kernel_plain (ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp)
{
    DEBUG_LOG(1,"Calling plain (maybe intrinsics) SELL kernel");
    DEBUG_LOG(2,"lhs vector has %s data and %"PRIDX" sub-vectors",ghost_datatype_string(lhs->traits->datatype),lhs->traits->ncols);

    ghost_error_t (*kernel) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list) = NULL;

#ifdef GHOST_HAVE_OPENMP
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
    ghost_datatype_idx_t matDtIdx;
    ghost_datatype_idx_t vecDtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&matDtIdx,mat->traits->datatype));
    GHOST_CALL_RETURN(ghost_datatype_idx(&vecDtIdx,lhs->traits->datatype));


#ifdef GHOST_HAVE_SSE
#if !(GHOST_HAVE_LONGIDX)
    if (!((options & GHOST_SPMV_AXPBY) ||
                (options & GHOST_SPMV_APPLY_SCALE) ||
                (options & GHOST_SPMV_APPLY_SHIFT))) {
        kernel = SELL_kernels_SSE
            [matDtIdx]
            [vecDtIdx];
    }
#endif
#elif defined(GHOST_HAVE_AVX)
    if (SELL(mat)->chunkHeight == 4) {
        kernel = SELL_kernels_AVX
            [matDtIdx]
            [vecDtIdx];
    } else if (SELL(mat)->chunkHeight == 32) {
        kernel = SELL_kernels_AVX_32
            [matDtIdx]
            [vecDtIdx];
    }
#elif defined(GHOST_HAVE_MIC)
#if !(GHOST_HAVE_LONGIDX)
    if (!((options & GHOST_SPMV_AXPBY) ||
                (options & GHOST_SPMV_SCALE) ||
                (options & GHOST_SPMV_SHIFT))) {
        if (SELL(mat)->chunkHeight == 16) {
            kernel = SELL_kernels_MIC_16
                [matDtIdx]
                [vecDtIdx];
        } else if (SELL(mat)->chunkHeight == 32) {
            kernel = SELL_kernels_MIC_32
                [matDtIdx]
                [vecDtIdx];
        }
    }
#endif
#else
    kernel = SELL_kernels_plain
        [matDtIdx]
        [vecDtIdx];
#endif

    if (kernel == NULL) {
        //WARNING_LOG("Selected kernel cannot be found. Falling back to plain C version!");
        kernel = SELL_kernels_plain
            [matDtIdx]
            [vecDtIdx];
    }

    return kernel(mat,lhs,rhs,options,argp);


}


#ifdef GHOST_HAVE_CUDA
static ghost_error_t SELL_kernel_CU (ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t flags,va_list argp)
{
    DEBUG_LOG(1,"Calling SELL CUDA kernel");
    DEBUG_LOG(2,"lhs vector has %s data",ghost_datatype_string(lhs->traits->datatype));
    ghost_datatype_idx_t matDtIdx;
    ghost_datatype_idx_t vecDtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&matDtIdx,mat->traits->datatype));
    GHOST_CALL_RETURN(ghost_datatype_idx(&vecDtIdx,lhs->traits->datatype));

    return SELL_kernels_CU
        [matDtIdx]
        [vecDtIdx](mat,lhs,rhs,flags,argp);


}
#endif

#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * invec, int options)
{
    ghost_idx_t c,j;
    ghost_nnz_t offs;
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
        if (options & GHOST_SPMV_AXPY) {
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

