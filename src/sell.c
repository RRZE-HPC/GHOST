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
#include <math.h>

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

const ghost_sell_aux_t GHOST_SELL_AUX_INITIALIZER = {
    .C = 32, 
    .T = 1
};

static void SELL_printInfo(ghost_sparsemat_t *mat, char **str);
static const char * SELL_formatName(ghost_sparsemat_t *mat);
static ghost_lidx_t SELL_rowLen (ghost_sparsemat_t *mat, ghost_lidx_t i);
static size_t SELL_byteSize (ghost_sparsemat_t *mat);
static ghost_error_t SELL_split(ghost_sparsemat_t *mat);
static ghost_error_t SELL_permute(ghost_sparsemat_t *, ghost_lidx_t *, ghost_lidx_t *);
static ghost_error_t SELL_upload(ghost_sparsemat_t *mat);
static ghost_error_t SELL_fromBin(ghost_sparsemat_t *mat, char *);
static ghost_error_t SELL_toBinCRS(ghost_sparsemat_t *mat, char *matrixPath);
static ghost_error_t SELL_fromRowFunc(ghost_sparsemat_t *mat, ghost_sparsemat_src_rowfunc_t *src);
static void SELL_free(ghost_sparsemat_t *mat);
static int ghost_selectSellChunkHeight(int datatype);

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
            mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_DEVICE;
        } else {
            mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_HOST;
        }
    }
    ghost_type_t ghost_type;
    GHOST_CALL_RETURN(ghost_type_get(&ghost_type));

    mat->upload = &SELL_upload;
    mat->fromFile = &SELL_fromBin;
    mat->fromMM = &ghost_sparsemat_from_mm;
    mat->toFile = &SELL_toBinCRS;
    mat->fromRowFunc = &SELL_fromRowFunc;
    mat->auxString = &SELL_printInfo;
    mat->formatName = &SELL_formatName;
    mat->rowLen     = &SELL_rowLen;
    mat->byteSize   = &SELL_byteSize;
    mat->spmv     = &ghost_sell_spmv_selector;
    mat->kacz   = &ghost_sell_kacz;
    mat->string    = &ghost_sell_stringify_selector;
    mat->split = &SELL_split;
    mat->permute = &SELL_permute;
#ifdef GHOST_HAVE_CUDA
    if ((ghost_type == GHOST_TYPE_CUDA) && (mat->traits->flags & GHOST_SPARSEMAT_DEVICE)) {
        mat->spmv   = &ghost_cu_sell_spmv_selector;
        mat->kacz = NULL;
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

static ghost_error_t SELL_permute(ghost_sparsemat_t *mat , ghost_lidx_t *perm, ghost_lidx_t *invPerm)
{
    UNUSED(mat);
    UNUSED(perm);
    UNUSED(invPerm);
    ERROR_LOG("SELL->permute() not implemented");
    return GHOST_ERR_NOT_IMPLEMENTED;

}
static void SELL_printInfo(ghost_sparsemat_t *mat, char **str)
{
    ghost_line_string(str,"Chunk height (C)",NULL,"%d",SELL(mat)->chunkHeight);
    ghost_line_string(str,"Chunk occupancy (beta)",NULL,"%f",SELL(mat)->beta);
    ghost_line_string(str,"Threads per row (T)",NULL,"%d",SELL(mat)->T);
}

static const char * SELL_formatName(ghost_sparsemat_t *mat)
{
    // TODO format SELL-C-sigma
    UNUSED(mat);
    return "SELL";
}

static ghost_lidx_t SELL_rowLen (ghost_sparsemat_t *mat, ghost_lidx_t i)
{
    if (mat && i<mat->nrows) {
        return SELL(mat)->rowLen[i];
    }

    return 0;
}

/*static ghost_dt SELL_entry (ghost_sparsemat_t *mat, ghost_lidx_t i, ghost_lidx_t j)
  {
  ghost_lidx_t e;

  if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE)
  i = mat->context->permutation->perm[i];
  if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE_COLS)
  j = mat->context->permutation->perm[j];

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
    return (size_t)((mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_lidx_t) + 
            mat->nEnts*(sizeof(ghost_lidx_t)+mat->elSize));
}

/*static int compareNZEPerRow( const void* a, const void* b ) 
{
    return  ((ghost_sorting_t*)b)->nEntsInRow - ((ghost_sorting_t*)a)->nEntsInRow;
}*/

static ghost_error_t SELL_fromRowFunc(ghost_sparsemat_t *mat, ghost_sparsemat_src_rowfunc_t *src)
{
    ghost_error_t ret = GHOST_SUCCESS;
    //   WARNING_LOG("SELL-%d-%d from row func",SELL(mat)->chunkHeight,SELL(mat)->scope);
    int nprocs = 1;
    int me;

    char *tmpval = NULL;
    ghost_gidx_t *tmpcol = NULL;    
    ghost_gidx_t i,col,row;
    ghost_gidx_t chunk,j;

    ghost_sorting_helper_t* rowSort = NULL;

    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);


    ghost_lidx_t nChunks = mat->nrowsPadded/SELL(mat)->chunkHeight;
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkStart, (nChunks+1)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkMin, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLen, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLenPadded, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLen, (mat->nrowsPadded)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_lidx_t)),err,ret);
    SELL(mat)->chunkStart[0] = 0;

    ghost_lidx_t maxRowLenInChunk = 0;
    ghost_lidx_t maxRowLen = 0;
    SELL(mat)->chunkStart[0] = 0;

    GHOST_CALL_GOTO(ghost_sparsemat_fromfunc_common(mat,src),err,ret);
    int funcerrs = 0;
    ghost_gidx_t gnents = 0, gnnz = 0;

#pragma omp parallel private(maxRowLenInChunk,i,tmpval,tmpcol,row) reduction (+:gnents,gnnz,funcerrs) 
        {
            int funcret = 0;
            maxRowLenInChunk = 0; 
            GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t)),ret);
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nChunks; chunk++ ) {
                for (i=0; i<SELL(mat)->chunkHeight; i++) {
                    row = chunk*SELL(mat)->chunkHeight+i;

                    if (row < mat->nrows) {
                        if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && mat->context->permutation) {
                            if (mat->context->permutation->scope == GHOST_PERMUTATION_GLOBAL) {
                                funcret = src->func(mat->context->permutation->invPerm[mat->context->lfRow[me]+row],&SELL(mat)->rowLen[row],tmpcol,tmpval);
                            } else {
                                funcret = src->func(mat->context->lfRow[me]+mat->context->permutation->invPerm[row],&SELL(mat)->rowLen[row],tmpcol,tmpval);
                            }
                        } else {
                            funcret = src->func(mat->context->lfRow[me]+row,&SELL(mat)->rowLen[row],tmpcol,tmpval);
                        }
                        if (funcret) {
                            funcerrs++;
                        }
                    } else {
                        SELL(mat)->rowLen[row] = 0;
                    }

                    SELL(mat)->rowLenPadded[row] = PAD(SELL(mat)->rowLen[row],SELL(mat)->T);

                    gnnz += SELL(mat)->rowLen[row];
                    maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[row]);
                }
#pragma omp critical
                maxRowLen = MAX(maxRowLen,maxRowLenInChunk);
                SELL(mat)->chunkLen[chunk] = maxRowLenInChunk;
                SELL(mat)->chunkLenPadded[chunk] = PAD(maxRowLenInChunk,SELL(mat)->T);
                gnents += SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
                maxRowLenInChunk = 0;
            }

            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        if (funcerrs) {
            ERROR_LOG("Matrix construction function returned error");
            ret = GHOST_ERR_UNKNOWN;
            goto err;
        }
 //   }

    for( chunk = 0; chunk < nChunks; chunk++ ) {
        SELL(mat)->chunkStart[chunk+1] = SELL(mat)->chunkStart[chunk] + SELL(mat)->chunkLenPadded[chunk]*SELL(mat)->chunkHeight;
    }

    if (gnents > (ghost_gidx_t)GHOST_LIDX_MAX) {
        ERROR_LOG("The local number of entries is too large: %"PRGIDX,gnents);
        return GHOST_ERR_DATATYPE;
    }

#ifdef GHOST_HAVE_MPI
    ghost_gidx_t fent = 0;
    for (i=0; i<nprocs; i++) {
        if (i>0 && me==i) {
            MPI_CALL_GOTO(MPI_Recv(&fent,1,ghost_mpi_dt_gidx,me-1,me-1,mat->context->mpicomm,MPI_STATUS_IGNORE),err,ret);
        }
        if (me==i && i<nprocs-1) {
            ghost_gidx_t send = fent+gnents;
            MPI_CALL_GOTO(MPI_Send(&send,1,ghost_mpi_dt_gidx,me+1,me,mat->context->mpicomm),err,ret);
        }
    }
    
    MPI_CALL_GOTO(MPI_Allgather(&gnents,1,ghost_mpi_dt_lidx,mat->context->lnEnts,1,ghost_mpi_dt_lidx,mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Allgather(&fent,1,ghost_mpi_dt_gidx,mat->context->lfEnt,1,ghost_mpi_dt_gidx,mat->context->mpicomm),err,ret);
#endif

    if (gnnz > (ghost_gidx_t)GHOST_LIDX_MAX) {
        ERROR_LOG("The local number of nonzeroes is too large.");
        return GHOST_ERR_DATATYPE;
    }
    mat->nEnts = (ghost_lidx_t)gnents;
    mat->nnz = (ghost_lidx_t)gnnz;
    SELL(mat)->beta = mat->nnz*1.0/(double)mat->nEnts;

    GHOST_CALL_GOTO(ghost_malloc_align((void **)&SELL(mat)->val,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&mat->col_orig,sizeof(ghost_gidx_t)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);


#pragma omp parallel for schedule(runtime) private (i,j)
    for (chunk = 0; chunk < nChunks; chunk++) {
        for (j=0; j<SELL(mat)->chunkLenPadded[chunk]; j++) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                mat->col_orig[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = mat->context->lfRow[me];
                memset(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i)],0,mat->elSize);
            }
        }
    }



#pragma omp parallel private(i,col,row,tmpval,tmpcol)
    {
        int funcret = 0;
        GHOST_CALL(ghost_malloc((void **)&tmpval,SELL(mat)->chunkHeight*src->maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,SELL(mat)->chunkHeight*src->maxrowlen*sizeof(ghost_gidx_t)),ret);
#pragma omp for schedule(runtime)
        for( chunk = 0; chunk < nChunks; chunk++ ) {
            memset(tmpval,0,mat->elSize*src->maxrowlen*SELL(mat)->chunkHeight);
            for (i=0; i<src->maxrowlen*SELL(mat)->chunkHeight; i++) {
                tmpcol[i] = mat->context->lfRow[me];
            }

            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                row = chunk*SELL(mat)->chunkHeight+i;

                if (row < mat->nrows) {
                    if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && mat->context->permutation) {
                        if (mat->context->permutation->scope == GHOST_PERMUTATION_GLOBAL) {
                            funcret = src->func(mat->context->permutation->invPerm[mat->context->lfRow[me]+row],&SELL(mat)->rowLen[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize]);
                        } else {
                            funcret = src->func(mat->context->lfRow[me]+mat->context->permutation->invPerm[row],&SELL(mat)->rowLen[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize]);
                        }
                    } else {
                        funcret = src->func(mat->context->lfRow[me]+row,&SELL(mat)->rowLen[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize]);
                    }
                }
                if (funcret) {
                    ERROR_LOG("Matrix construction function returned error");
                    ret = GHOST_ERR_UNKNOWN;
                }

                for (col = 0; col<SELL(mat)->chunkLenPadded[chunk]; col++) {
                    memcpy(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&tmpval[mat->elSize*(i*src->maxrowlen+col)],mat->elSize);
                    if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && mat->context->permutation) {
                        if (mat->context->permutation->scope == GHOST_PERMUTATION_GLOBAL) { // no distinction between global and local entries
                            mat->col_orig[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = mat->context->permutation->perm[tmpcol[i*src->maxrowlen+col]];
                        } else { // local permutation: distinction between global and local entries
                            if ((tmpcol[i*src->maxrowlen+col] >= mat->context->lfRow[me]) && (tmpcol[i*src->maxrowlen+col] < (mat->context->lfRow[me]+mat->nrows))) { // local entry: copy with permutation
                                if (mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                    mat->col_orig[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*src->maxrowlen+col];
                                } else {
                                    mat->col_orig[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = mat->context->permutation->perm[tmpcol[i*src->maxrowlen+col]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                                }
                            } else { // remote entry: copy without permutation
                                mat->col_orig[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*src->maxrowlen+col];
                            }
                        }
                    } else {
                        mat->col_orig[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = tmpcol[i*src->maxrowlen+col];
                    }
                }
                if (!(mat->traits->flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
                    // sort rows by ascending column indices
                    ghost_sparsemat_sortrow(&mat->col_orig[SELL(mat)->chunkStart[chunk]+i],&SELL(mat)->val[(SELL(mat)->chunkStart[chunk]+i)*mat->elSize],mat->elSize,SELL(mat)->rowLen[row],SELL(mat)->chunkHeight);
                }
#pragma omp critical
                ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&mat->col_orig[SELL(mat)->chunkStart[chunk]+i],SELL(mat)->rowLen[row],SELL(mat)->chunkHeight);
            }
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }

    ghost_sparsemat_registerrow_finalize(mat);

    if (ret != GHOST_SUCCESS) {
        goto err;
    }


    mat->context->lnEnts[me] = mat->nEnts;

    for (i=0; i<nprocs; i++) {
        mat->context->lfEnt[i] = 0;
    } 

    for (i=1; i<nprocs; i++) {
        mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
    } 

    GHOST_CALL_GOTO(mat->split(mat),err,ret);

#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPARSEMAT_HOST))
        mat->upload(mat);
#endif

    goto out;
err:
    free(SELL(mat)->val); SELL(mat)->val = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
    free(SELL(mat)->chunkMin); SELL(mat)->chunkMin = NULL;
    free(SELL(mat)->chunkLen); SELL(mat)->chunkLen = NULL;
    free(SELL(mat)->chunkLenPadded); SELL(mat)->chunkLenPadded = NULL;
    free(SELL(mat)->rowLen); SELL(mat)->rowLen = NULL;
    free(SELL(mat)->rowLenPadded); SELL(mat)->rowLenPadded = NULL;
    free(SELL(mat)->chunkStart); SELL(mat)->chunkStart = NULL;
    SELL(mat)->beta = 0;
    mat->nEnts = 0;
    mat->nnz = 0;
    free(mat->context->permutation->perm); mat->context->permutation->perm = NULL;
    free(mat->context->permutation->invPerm); mat->context->permutation->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
    free(mat->context->permutation->cu_perm); mat->context->permutation->cu_perm = NULL;
#endif

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
    ghost_gidx_t i,j;
    int me;
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));

    ghost_lidx_t lnEnts_l, lnEnts_r;
    ghost_lidx_t current_l, current_r;


    ghost_lidx_t chunk;
    ghost_lidx_t idx, row;

#ifdef GHOST_HAVE_UNIFORM_IDX
    if (!(mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
        DEBUG_LOG(1,"In-place column compression!");
        SELL(mat)->col = mat->col_orig;
    } else 
#endif
    {
        DEBUG_LOG(1,"Duplicate col array!");
        GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->col,sizeof(ghost_lidx_t)*mat->nEnts),err,ret);
    }
   
    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED) {
        GHOST_CALL_GOTO(ghost_context_comm_init(mat->context,mat->col_orig,fullSELL->col),err,ret);

    } else {
        for (i=0; i<mat->nEnts; i++) {
            SELL(mat)->col[i] = (ghost_lidx_t)mat->col_orig[i];
        }
    }
#ifndef GHOST_HAVE_UNIFORM_IDX
    if (!(mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
        DEBUG_LOG(1,"Free orig cols");
        free(mat->col_orig);
        mat->col_orig = NULL;
    }
#endif

    ghost_sparsemat_create(&(mat->localPart),mat->context,&mat->traits[0],1);
    localSELL = mat->localPart->data;
    mat->localPart->traits->symmetry = mat->traits->symmetry;

    ghost_sparsemat_create(&(mat->remotePart),mat->context,&mat->traits[0],1);
    remoteSELL = mat->remotePart->data; 

    localSELL->T = fullSELL->T;
    remoteSELL->T = fullSELL->T;

    ghost_lidx_t nChunks = mat->nrowsPadded/fullSELL->chunkHeight;
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkStart, (nChunks+1)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkMin, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkLen, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkLenPadded, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->rowLen, (mat->nrowsPadded)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_lidx_t)),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkStart, (nChunks+1)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkMin, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkLen, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkLenPadded, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->rowLen, (mat->nrowsPadded)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_lidx_t)),err,ret);

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
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->col,lnEnts_l*sizeof(ghost_lidx_t)),err,ret); 

        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->val,lnEnts_r*mat->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->col,lnEnts_r*sizeof(ghost_lidx_t)),err,ret); 

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
        ghost_lidx_t col_l[fullSELL->chunkHeight], col_r[fullSELL->chunkHeight];

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
                                ghost_lidx_t lidx = localSELL->chunkStart[chunk]+col_l[j]*localSELL->chunkHeight+j;
                                localSELL->col[lidx] = fullSELL->col[idx];
                                memcpy(&localSELL->val[lidx*mat->elSize],&fullSELL->val[idx*mat->elSize],mat->elSize);
                                current_l++;
                            }
                            col_l[j]++;
                        }
                        else{
                            if (col_r[j] < remoteSELL->rowLen[row]) {
                                ghost_lidx_t ridx = remoteSELL->chunkStart[chunk]+col_r[j]*remoteSELL->chunkHeight+j;
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

#ifdef GHOST_HAVE_CUDA
        if (!(mat->traits->flags & GHOST_SPARSEMAT_HOST)) {
            mat->localPart->upload(mat->localPart);
            mat->remotePart->upload(mat->remotePart);
        }
#endif
    }

    goto out;
err:
    mat->localPart->destroy(mat->localPart); mat->localPart = NULL;
    mat->remotePart->destroy(mat->remotePart); mat->remotePart = NULL;

out:
    return ret;
}

static ghost_error_t SELL_toBinCRS(ghost_sparsemat_t *mat, char *matrixPath)
{
    UNUSED(mat);
    UNUSED(matrixPath);

    ERROR_LOG("SELL matrix to binary CRS file not implemented");
    return GHOST_ERR_NOT_IMPLEMENTED;
}

static ghost_error_t SELL_fromBin(ghost_sparsemat_t *mat, char *matrixPath)
{
    DEBUG_LOG(1,"Creating SELL matrix from binary file");
    ghost_error_t ret = GHOST_SUCCESS;

    int nprocs = 1;
    int me;
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, mat->context->mpicomm));
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));

    ghost_lidx_t *rpt = NULL;
    ghost_gidx_t *tmpcol = NULL;
    char *tmpval = NULL;

    ghost_gidx_t i;
    ghost_gidx_t chunk,j;

    ghost_sparsemat_fromfile_common(mat,matrixPath,&rpt);
   
    mat->nEnts = 0;
    mat->nrowsPadded = PAD(mat->nrows,SELL(mat)->chunkHeight);

    ghost_lidx_t nChunks =  mat->nrowsPadded/SELL(mat)->chunkHeight;
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkStart, (nChunks+1)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkMin, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLen, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLenPadded, (nChunks)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLen, (mat->nrowsPadded)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_lidx_t)),err,ret);

    ghost_lidx_t maxRowLenInChunk = 0;
    ghost_lidx_t minRowLenInChunk = INT_MAX;

    SELL(mat)->chunkStart[0] = 0;    

    DEBUG_LOG(1,"Extracting row lenghts");
    for( chunk = 0; chunk < nChunks; chunk++ ) {
        for (i=0; i<SELL(mat)->chunkHeight; i++) {
            ghost_lidx_t row = chunk*SELL(mat)->chunkHeight+i;
            if (row < mat->nrows) {
                SELL(mat)->rowLen[row] = rpt[row+1]-rpt[row];
            } else {
                SELL(mat)->rowLen[row] = 0;
            }
            SELL(mat)->rowLenPadded[row] = PAD(SELL(mat)->rowLen[row],SELL(mat)->T);

            maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[row]);
            minRowLenInChunk = MIN(minRowLenInChunk,SELL(mat)->rowLen[row]);
        }


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

    DEBUG_LOG(1,"SELL matrix has %"PRLIDX" (padded to %"PRLIDX") rows, %"PRGIDX" cols and %"PRLIDX" nonzeros and %"PRLIDX" entries",mat->nrows,mat->nrowsPadded,mat->ncols,mat->nnz,mat->nEnts);

    GHOST_CALL_GOTO(ghost_malloc_align((void **)&SELL(mat)->val,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&mat->col_orig,sizeof(ghost_gidx_t)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);

#pragma omp parallel for schedule(runtime) private (i,j)
    for (chunk = 0; chunk < nChunks; chunk++) {
        for (j=0; j<SELL(mat)->chunkLenPadded[chunk]; j++) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                mat->col_orig[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = mat->context->lfRow[me];
                memset(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i)],0,mat->elSize);
            }
        }
    }

    DEBUG_LOG(1,"Memory usage may be high because read-in of CRS data is done at once and not chunk-wise");
    GHOST_CALL_GOTO(ghost_malloc((void **)&tmpcol,mat->nEnts*sizeof(ghost_gidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&tmpval,mat->nEnts*mat->elSize),err,ret);
    GHOST_CALL_GOTO(ghost_bincrs_col_read(tmpcol, matrixPath, mat->context->lfRow[me], mat->nrows, mat->context->permutation, mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS),err,ret);
    GHOST_CALL_GOTO(ghost_bincrs_val_read(tmpval, mat->traits->datatype, matrixPath,  mat->context->lfRow[me], mat->nrows, mat->context->permutation),err,ret);

    ghost_lidx_t row = 0;
    for (chunk = 0; chunk < nChunks; chunk++) {
        ghost_lidx_t col;
        ghost_gidx_t *curRowCols;
        char * curRowVals;
        for (i=0; (i<SELL(mat)->chunkHeight) && (row < mat->nrows); i++, row++) {
            curRowCols = &tmpcol[rpt[row]];
            curRowVals = &tmpval[rpt[row]*mat->elSize];
            for (col=0; col<SELL(mat)->rowLen[row]; col++) {
                mat->col_orig[SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i] = curRowCols[col];
                memcpy(&SELL(mat)->val[mat->elSize*(SELL(mat)->chunkStart[chunk]+col*SELL(mat)->chunkHeight+i)],&curRowVals[col*mat->elSize],mat->elSize);
            }
            if (!(mat->traits->flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
                // sort rows by ascending column indices
                ghost_sparsemat_sortrow(&mat->col_orig[SELL(mat)->chunkStart[chunk]+i],&SELL(mat)->val[(SELL(mat)->chunkStart[chunk]+i)*mat->elSize],mat->elSize,SELL(mat)->rowLen[row],SELL(mat)->chunkHeight);
            }
            ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&mat->col_orig[SELL(mat)->chunkStart[chunk]+i],SELL(mat)->rowLen[row],SELL(mat)->chunkHeight);
        }

    }
    ghost_sparsemat_registerrow_finalize(mat);

    mat->split(mat);


#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPARSEMAT_HOST))
        mat->upload(mat);
#endif



    DEBUG_LOG(1,"SELL matrix successfully created");
    goto out;

err:
    free(SELL(mat)->val); SELL(mat)->val = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
    free(SELL(mat)->chunkMin); SELL(mat)->chunkMin = NULL;
    free(SELL(mat)->chunkLen); SELL(mat)->chunkLen = NULL;
    free(SELL(mat)->chunkLenPadded); SELL(mat)->chunkLenPadded = NULL;
    free(SELL(mat)->rowLen); SELL(mat)->rowLen = NULL;
    free(SELL(mat)->rowLenPadded); SELL(mat)->rowLenPadded = NULL;
    free(SELL(mat)->chunkStart); SELL(mat)->chunkStart = NULL;
    free(mat->context->permutation->perm); mat->context->permutation->perm = NULL;
    free(mat->context->permutation->invPerm); mat->context->permutation->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
    free(mat->context->permutation->cu_perm); mat->context->permutation->cu_perm = NULL;
#endif

out:
    free(tmpcol); tmpcol = NULL;
    free(tmpval); tmpval = NULL;
    free(rpt); rpt = NULL;

    return ret;
}

static ghost_error_t SELL_upload(ghost_sparsemat_t* mat) 
{
#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPARSEMAT_HOST)) {
        DEBUG_LOG(1,"Creating matrix on CUDA device");
        GHOST_CALL_RETURN(ghost_malloc((void **)&SELL(mat)->cumat,sizeof(ghost_cu_sell_t)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->rowLen,(mat->nrows)*sizeof(ghost_lidx_t)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->rowLenPadded,(mat->nrows)*sizeof(ghost_lidx_t)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->col,(mat->nEnts)*sizeof(ghost_lidx_t)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->val,(mat->nEnts)*mat->elSize));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->chunkStart,(mat->nrowsPadded/SELL(mat)->chunkHeight+1)*sizeof(ghost_lidx_t)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->chunkLen,(mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_lidx_t)));

        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->rowLen, SELL(mat)->rowLen, mat->nrows*sizeof(ghost_lidx_t)));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->rowLenPadded, SELL(mat)->rowLenPadded, mat->nrows*sizeof(ghost_lidx_t)));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->col, SELL(mat)->col, mat->nEnts*sizeof(ghost_lidx_t)));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->val, SELL(mat)->val, mat->nEnts*mat->elSize));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->chunkStart, SELL(mat)->chunkStart, (mat->nrowsPadded/SELL(mat)->chunkHeight+1)*sizeof(ghost_lidx_t)));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->chunkLen, SELL(mat)->chunkLen, (mat->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_lidx_t)));
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
        if (mat->traits->flags & GHOST_SPARSEMAT_DEVICE && SELL(mat)->cumat) {
            ghost_cu_free(SELL(mat)->cumat->rowLen);
            ghost_cu_free(SELL(mat)->cumat->rowLenPadded);
            ghost_cu_free(SELL(mat)->cumat->col);
            ghost_cu_free(SELL(mat)->cumat->val);
            ghost_cu_free(SELL(mat)->cumat->chunkStart);
            ghost_cu_free(SELL(mat)->cumat->chunkLen);
            free(SELL(mat)->cumat);
        }
#endif
        free(SELL(mat)->val); SELL(mat)->val = NULL;
        free(SELL(mat)->col); SELL(mat)->col = NULL;
        free(SELL(mat)->chunkStart); SELL(mat)->chunkStart = NULL;
        free(SELL(mat)->chunkMin); SELL(mat)->chunkMin = NULL;
        free(SELL(mat)->chunkLen); SELL(mat)->chunkLen = NULL;
        free(SELL(mat)->chunkLenPadded); SELL(mat)->chunkLenPadded = NULL;
        free(SELL(mat)->rowLen); SELL(mat)->rowLen = NULL;
        free(SELL(mat)->rowLenPadded); SELL(mat)->rowLenPadded = NULL;
    }

         
    if (mat->localPart) {
        SELL_free(mat->localPart);
    }

    if (mat->remotePart) {
        SELL_free(mat->remotePart);
    }

    ghost_sparsemat_destroy_common(mat);

    free(mat);
}

/*static int ld(int i) 
{
    return (int)log2((double)i);
}*/


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

