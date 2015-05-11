#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/crs.h"
#include "ghost/sell.h"
#include "ghost/sparsemat.h"
#include "ghost/context.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#include "ghost/bincrs.h"
#include "ghost/matrixmarket.h"
#include "ghost/instr.h"

#include <libgen.h>
#include <math.h>

const ghost_sparsemat_src_rowfunc_t GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER = {
    .func = NULL,
    .maxrowlen = 0,
    .base = 0,
    .flags = GHOST_SPARSEMAT_FROMROWFUNC_DEFAULT
};
    

const ghost_sparsemat_traits_t GHOST_SPARSEMAT_TRAITS_INITIALIZER = {
    .format = GHOST_SPARSEMAT_CRS,
    .flags = GHOST_SPARSEMAT_DEFAULT,
    .symmetry = GHOST_SPARSEMAT_SYMM_GENERAL,
    .aux = NULL,
    .scotchStrat = (char*)GHOST_SCOTCH_STRAT_DEFAULT,
    .sortScope = 1,
    .datatype = (ghost_datatype_t) (GHOST_DT_DOUBLE|GHOST_DT_REAL),
    .opt_blockvec_width = 0
};


ghost_error_t ghost_sparsemat_create(ghost_sparsemat_t ** mat, ghost_context_t *context, ghost_sparsemat_traits_t *traits, int nTraits)
{
    UNUSED(nTraits);
    ghost_error_t ret = GHOST_SUCCESS;

    int me;
    GHOST_CALL_GOTO(ghost_rank(&me, context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)mat,sizeof(ghost_sparsemat_t)),err,ret);

    (*mat)->traits = traits;
    (*mat)->context = context;
    (*mat)->localPart = NULL;
    (*mat)->remotePart = NULL;
    (*mat)->name = "Sparse matrix";
    (*mat)->col_orig = NULL;
    (*mat)->data = NULL;
    (*mat)->nzDist = NULL;
    (*mat)->fromFile = NULL;
    (*mat)->toFile = NULL;
    (*mat)->fromRowFunc = NULL;
    (*mat)->auxString = NULL;
    (*mat)->formatName = NULL;
    (*mat)->rowLen = NULL;
    (*mat)->byteSize = NULL;
    (*mat)->permute = NULL;
    (*mat)->destroy = NULL;
    (*mat)->string = NULL;
    (*mat)->upload = NULL;
    (*mat)->permute = NULL;
    (*mat)->spmv = NULL;
    (*mat)->destroy = NULL;
    (*mat)->split = NULL;
    (*mat)->bandwidth = 0;
    (*mat)->lowerBandwidth = 0;
    (*mat)->upperBandwidth = 0;
    (*mat)->avgRowBand = 0.;
    (*mat)->avgAvgRowBand = 0.;
    (*mat)->smartRowBand = 0.;
    (*mat)->maxRowLen = 0;
    (*mat)->nMaxRows = 0;
    (*mat)->variance = 0.;
    (*mat)->deviation = 0.;
    (*mat)->cv = 0.;
    (*mat)->nrows = context->lnrows[me];
    (*mat)->nrowsPadded = (*mat)->nrows;
    (*mat)->ncols = context->gncols;
    (*mat)->nEnts = 0;
    (*mat)->nnz = 0;
    (*mat)->ncolors = 0;
    (*mat)->color_ptr = NULL;

    if ((*mat)->traits->sortScope == GHOST_SPARSEMAT_SORT_GLOBAL) {
        (*mat)->traits->sortScope = (*mat)->context->gnrows;
    } else if ((*mat)->traits->sortScope == GHOST_SPARSEMAT_SORT_LOCAL) {
        (*mat)->traits->sortScope = (*mat)->nrows;
    }

#ifdef GHOST_GATHER_GLOBAL_INFO
    GHOST_CALL_GOTO(ghost_malloc((void **)&((*mat)->nzDist),sizeof(ghost_gidx_t)*(2*context->gnrows-1)),err,ret);
#endif
    GHOST_CALL_GOTO(ghost_datatype_size(&(*mat)->elSize,(*mat)->traits->datatype),err,ret);

    switch (traits->format) {
        case GHOST_SPARSEMAT_CRS:
            GHOST_CALL_GOTO(ghost_crs_init(*mat),err,ret);
            break;
        case GHOST_SPARSEMAT_SELL:
            GHOST_CALL_GOTO(ghost_sell_init(*mat),err,ret);
            break;
        default:
            WARNING_LOG("Invalid sparse matrix format. Falling back to CRS!");
            traits->format = GHOST_SPARSEMAT_CRS;
            GHOST_CALL_GOTO(ghost_crs_init(*mat),err,ret);
    }

    goto out;
err:
    ERROR_LOG("Error. Free'ing resources");
    free(*mat); *mat = NULL;

out:
    return ret;    
}

ghost_error_t ghost_sparsemat_sortrow(ghost_gidx_t *col, char *val, size_t valSize, ghost_lidx_t rowlen, ghost_lidx_t stride)
{
    ghost_lidx_t n;
    ghost_lidx_t c;
    ghost_lidx_t swpcol;
    char swpval[valSize];
    for (n=rowlen; n>1; n--) {
        for (c=0; c<n-1; c++) {
            if (col[c*stride] > col[(c+1)*stride]) {
                swpcol = col[c*stride];
                col[c*stride] = col[(c+1)*stride];
                col[(c+1)*stride] = swpcol; 

                memcpy(&swpval,&val[c*stride*valSize],valSize);
                memcpy(&val[c*stride*valSize],&val[(c+1)*stride*valSize],valSize);
                memcpy(&val[(c+1)*stride*valSize],&swpval,valSize);
            }
        }
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_fromfunc_common(ghost_lidx_t *rl, ghost_lidx_t *rlp, ghost_lidx_t *cl, ghost_lidx_t *clp, ghost_lidx_t *chunkptr, char **val, ghost_gidx_t **col, ghost_sparsemat_src_rowfunc_t *src, ghost_sparsemat_t *mat, ghost_lidx_t C, ghost_lidx_t P)
{
    ghost_error_t ret = GHOST_SUCCESS;
    int funcerrs = 0;
    char *tmpval = NULL;
    ghost_gidx_t *tmpcol = NULL;
    ghost_lidx_t nchunks = (ghost_lidx_t)(ceil((double)mat->nrows/(double)C));
    ghost_lidx_t i,row,chunk,j,colidx;
    ghost_gidx_t gnents = 0, gnnz = 0;
    ghost_lidx_t maxRowLenInChunk = 0, maxRowLen = 0;
    int me,nprocs;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    
    mat->ncols = mat->context->gncols;
    mat->nrows = mat->context->lnrows[me];

#ifdef GHOST_GATHER_GLOBAL_INFO
    memset(mat->nzDist,0,sizeof(ghost_gidx_t)*(2*mat->context->gnrows-1));
#endif
    mat->lowerBandwidth = 0;
    mat->upperBandwidth = 0;
    
    if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_PERMUTE;
    }

    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
        if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_sparsemat_perm_scotch(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } 
        if (mat->traits->flags & GHOST_SPARSEMAT_COLOR) {
            ghost_sparsemat_perm_color(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } 
        if (mat->traits->sortScope > 1) {
            ghost_sparsemat_perm_sort(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC,mat->traits->sortScope);
        }
    } else {
        if (mat->traits->sortScope > 1) {
            WARNING_LOG("Ignoring sorting scope");
        }
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_NOT_PERMUTE_COLS;
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_NOT_SORT_COLS;
    }

    ghost_lidx_t *tmpclp = NULL;
    if (!clp) {
        ghost_malloc((void **)&tmpclp,nchunks*sizeof(ghost_lidx_t));
        clp = tmpclp;
    }
    ghost_lidx_t *tmprl = NULL;
    if (!rl) {
        ghost_malloc((void **)&tmprl,nchunks*sizeof(ghost_lidx_t));
        rl = tmprl;
    }

#pragma omp parallel private(maxRowLenInChunk,i,tmpval,tmpcol,row) shared (maxRowLen) reduction (+:gnents,gnnz,funcerrs) 
    {
        ghost_lidx_t rowlen;
        maxRowLenInChunk = 0; 
        GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t)),ret);
#pragma omp for schedule(runtime)
        for( chunk = 0; chunk < nchunks; chunk++ ) {
            for (i=0; (i<C) && (chunk*C+i < mat->nrows); i++) {
                row = chunk*C+i;

                if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
                    if (mat->context->perm_global && mat->context->perm_local) {
                        INFO_LOG("Global _and_ local permutation");
                        funcerrs += src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[row]],&rowlen,tmpcol,tmpval);
                    } else if (mat->context->perm_global) {
                        funcerrs += src->func(mat->context->perm_global->invPerm[row],&rowlen,tmpcol,tmpval);
                    } else if (mat->context->perm_local) {
                        funcerrs += src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[row],&rowlen,tmpcol,tmpval);
                    }
                } else {
                    funcerrs += src->func(mat->context->lfRow[me]+row,&rowlen,tmpcol,tmpval);
                }

                // rl _must_ not be NULL because we need it for the statistics
                rl[row] = rowlen;
                
                if (rlp) {
                    rlp[row] = PAD(rowlen,P);
                }

                gnnz += rowlen;
                maxRowLenInChunk = MAX(maxRowLenInChunk,rowlen);
            }
            if (cl) {
                cl[chunk] = maxRowLenInChunk;
            }

            // clp _must_ not be NULL because we need it for the chunkptr computation
            clp[chunk] = PAD(maxRowLenInChunk,P);

            gnents += clp[chunk]*C;

#pragma omp critical
            maxRowLen = MAX(maxRowLen,maxRowLenInChunk);

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
    if (gnents > (ghost_gidx_t)GHOST_LIDX_MAX) {
        ERROR_LOG("The local number of entries is too large: %"PRGIDX,gnents);
        return GHOST_ERR_DATATYPE;
    }
    if (gnnz > (ghost_gidx_t)GHOST_LIDX_MAX) {
        ERROR_LOG("The local number of entries is too large: %"PRGIDX,gnents);
        return GHOST_ERR_DATATYPE;
    }

    if (src->maxrowlen != maxRowLen) {
        INFO_LOG("The maximum row length was not correct. Setting it from %"PRLIDX" to %"PRLIDX,src->maxrowlen,maxRowLen); 
        src->maxrowlen = maxRowLen;
    }

    mat->nnz = (ghost_lidx_t)gnnz;
    mat->nEnts = (ghost_lidx_t)gnents;

    // NUMA first touch   
#pragma omp parallel for schedule (runtime)
    for(chunk = 0; chunk < nchunks+1; chunk++ ) {
        chunkptr[chunk] = 0;
    }

    for(chunk = 0; chunk < nchunks; chunk++ ) {
        chunkptr[chunk+1] = chunkptr[chunk] + clp[chunk]*C;
    }

#ifdef GHOST_HAVE_MPI
    ghost_gidx_t fent = 0;
    for (i=0; i<nprocs; i++) {
        if (i>0 && me==i) {
            MPI_CALL_GOTO(MPI_Recv(&fent,1,ghost_mpi_dt_gidx,me-1,me-1,mat->context->mpicomm,MPI_STATUS_IGNORE),err,ret);
        }
        if (me==i && i<nprocs-1) {
            ghost_gidx_t send = fent+mat->nEnts;
            MPI_CALL_GOTO(MPI_Send(&send,1,ghost_mpi_dt_gidx,me+1,me,mat->context->mpicomm),err,ret);
        }
    }
    
    MPI_CALL_GOTO(MPI_Allgather(&mat->nEnts,1,ghost_mpi_dt_lidx,mat->context->lnEnts,1,ghost_mpi_dt_lidx,mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Allgather(&fent,1,ghost_mpi_dt_gidx,mat->context->lfEnt,1,ghost_mpi_dt_gidx,mat->context->mpicomm),err,ret);
#endif
    
    GHOST_CALL_GOTO(ghost_malloc_align((void **)val,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)col,sizeof(ghost_gidx_t)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);

#pragma omp parallel for schedule(runtime) private (i,j)
    for (chunk = 0; chunk < nchunks; chunk++) {
        for (j=0; j<clp[chunk]; j++) {
            for (i=0; i<C; i++) {
                (*col)[chunkptr[chunk]+j*C+i] = mat->context->lfRow[me];
                memset(&((*val)[mat->elSize*(chunkptr[chunk]+j*C+i)]),0,mat->elSize);
            }
        }
    }

#pragma omp parallel private(i,colidx,row,tmpval,tmpcol)
    {
        int funcret = 0;
        GHOST_CALL(ghost_malloc((void **)&tmpval,C*src->maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,C*src->maxrowlen*sizeof(ghost_gidx_t)),ret);
#pragma omp for schedule(runtime)
        for( chunk = 0; chunk < nchunks; chunk++ ) {
            memset(tmpval,0,mat->elSize*src->maxrowlen*C);
            for (i=0; i<src->maxrowlen*C; i++) {
                tmpcol[i] = mat->context->lfRow[me];
            }

            for (i=0; (i<C) && (chunk*C+i < mat->nrows); i++) {
                row = chunk*C+i;

                if (row < mat->nrows) {
                    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
                        if (mat->context->perm_global && mat->context->perm_local) {
                            funcret = src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[row]],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize]);
                        } else if (mat->context->perm_global) {
                            funcret = src->func(mat->context->perm_global->invPerm[row],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize]);
                        } else if (mat->context->perm_local) {
                            funcret = src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[row],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize]);
                        }
                    } else {
                        funcret = src->func(mat->context->lfRow[me]+row,&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize]);
                    }
                }
                if (funcret) {
                    ERROR_LOG("Matrix construction function returned error");
                    ret = GHOST_ERR_UNKNOWN;
                }

                for (colidx = 0; colidx<clp[chunk]; colidx++) {
                    memcpy(*val+mat->elSize*(chunkptr[chunk]+colidx*C+i),&tmpval[mat->elSize*(i*src->maxrowlen+colidx)],mat->elSize);
                    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
                        if (mat->context->perm_global) {
                            // no distinction between global and local entries
                            // global permutation will be done after all rows are read
                            (*col)[chunkptr[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                        } else { 
                            // local permutation: distinction between global and local entries
                            if ((tmpcol[i*src->maxrowlen+colidx] >= mat->context->lfRow[me]) && (tmpcol[i*src->maxrowlen+colidx] < (mat->context->lfRow[me]+mat->nrows))) { // local entry: copy with permutation
                                if (mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                    (*col)[chunkptr[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                                } else {
                                    (*col)[chunkptr[chunk]+colidx*C+i] = mat->context->perm_local->perm[tmpcol[i*src->maxrowlen+colidx]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                                }
                            } else { // remote entry: copy without permutation
                                (*col)[chunkptr[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                            }
                        }
                    } else {
                        (*col)[chunkptr[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                    }
                }
            }
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
    
    if (mat->context->perm_global) {
        ghost_sparsemat_perm_global_cols(*col,mat->nEnts,mat->context);
    }
    
    for( chunk = 0; chunk < nchunks; chunk++ ) {
        for (i=0; (i<C) && (chunk*C+i < mat->nrows); i++) {
            row = chunk*C+i;
            ghost_sparsemat_sortrow(&((*col)[chunkptr[chunk]+i]),&(*val)[(chunkptr[chunk]+i)*mat->elSize],mat->elSize,rl[row],C);
            ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&(*col)[chunkptr[chunk]+i],rl[row],C);
        }
    }

    ghost_sparsemat_registerrow_finalize(mat);
    mat->context->lnEnts[me] = mat->nEnts;

    for (i=0; i<nprocs; i++) {
        mat->context->lfEnt[i] = 0;
    } 

    for (i=1; i<nprocs; i++) {
        mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
    } 


    free(tmpclp);
    free(tmprl);

    goto out;
err:

out:

    return ret;
}

int ghost_cmp_entsperrow(const void* a, const void* b) 
{
    return  ((ghost_sorting_helper_t*)b)->nEntsInRow - ((ghost_sorting_helper_t*)a)->nEntsInRow;
}

ghost_error_t ghost_sparsemat_perm_global_cols(ghost_gidx_t *col, ghost_lidx_t ncols, ghost_context_t *context) 
{
#ifdef GHOST_HAVE_MPI
    int me, nprocs,i;
    ghost_rank(&me,context->mpicomm);
    ghost_nrank(&nprocs,context->mpicomm);

    for (i=0; i<nprocs; i++) {
        ghost_lidx_t nels = 0;
        if (i==me) {
            nels = ncols;
        }
        MPI_Bcast(&nels,1,ghost_mpi_dt_gidx,i,context->mpicomm);

        ghost_gidx_t *colsfromi;
        ghost_malloc((void **)&colsfromi,nels*sizeof(ghost_gidx_t));
    
        if (i==me) {
            memcpy(colsfromi,col,nels*sizeof(ghost_gidx_t));
        }
        MPI_Bcast(colsfromi,nels,ghost_mpi_dt_gidx,i,context->mpicomm);

        ghost_lidx_t el;
        for (el=0; el<nels; el++) {
            if ((colsfromi[el] >= context->lfRow[me]) && (colsfromi[el] < (context->lfRow[me]+context->lnrows[me]))) {
                colsfromi[el] = context->perm_global->perm[colsfromi[el]-context->lfRow[me]];
            } else {
                colsfromi[el] = 0;
            }
        }

        if (i==me) {
            MPI_Reduce(MPI_IN_PLACE,colsfromi,nels,ghost_mpi_dt_gidx,MPI_MAX,i,context->mpicomm);
        } else {
            MPI_Reduce(colsfromi,NULL,nels,ghost_mpi_dt_gidx,MPI_MAX,i,context->mpicomm);
        }

        if (i==me) {
            if (context->perm_local) {
                for (el=0; el<nels; el++) {
                    if ((colsfromi[el] >= context->lfRow[me]) && (colsfromi[el] < context->lfRow[me]+context->lnrows[me])) {
                        col[el] = context->lfRow[me] + context->perm_local->perm[colsfromi[el]-context->lfRow[me]];
                    } else {
                        col[el] = colsfromi[el];
                    }

                }
            } else {
                memcpy(col,colsfromi,nels*sizeof(ghost_gidx_t));
            }
        }

        free(colsfromi);
    }
#else
    ERROR_LOG("This function should not have been called without MPI!");
    UNUSED(col);
    UNUSED(ncols);
    UNUSED(context);
#endif
    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_perm_sort(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType, ghost_gidx_t scope)
{
    ghost_error_t ret = GHOST_SUCCESS;
    if (mat->context->perm_local) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }
    INFO_LOG("in permutation by sorting");
    
    int me;    
    ghost_gidx_t i,c,nrows,rowOffset;
    ghost_sorting_helper_t *rowSort = NULL;
    ghost_gidx_t *rpt = NULL;

    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);

    

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local,sizeof(ghost_permutation_t)),err,ret);
    if (mat->traits->sortScope > mat->nrows) {
        WARNING_LOG("Restricting the sorting scope to the number of matrix rows");
    }
    nrows = mat->nrows;
    rowOffset = mat->context->lfRow[me];
    mat->context->perm_local->scope = GHOST_PERMUTATION_LOCAL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->perm,sizeof(ghost_gidx_t)*nrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->invPerm,sizeof(ghost_gidx_t)*nrows),err,ret);
#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->context->perm_local->cu_perm,sizeof(ghost_gidx_t)*nrows),err,ret);
#endif

    mat->context->perm_local->len = nrows;

    memset(mat->context->perm_local->perm,0,sizeof(ghost_gidx_t)*nrows);
    memset(mat->context->perm_local->invPerm,0,sizeof(ghost_gidx_t)*nrows);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&rowSort,nrows * sizeof(ghost_sorting_helper_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(nrows+1) * sizeof(ghost_gidx_t)),err,ret);

    if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
        ghost_sparsemat_src_rowfunc_t *src = (ghost_sparsemat_src_rowfunc_t *)matrixSource;
        char *tmpval = NULL;
        ghost_gidx_t *tmpcol = NULL;
        rpt[0] = 0;
#pragma omp parallel private(i,tmpval,tmpcol)
        { 
            GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t)),ret);
#pragma omp for schedule(runtime)
            for (i=0; i<nrows; i++) {
                if (mat->context->perm_global) {
                    INFO_LOG("global perm exists");
                    if (src->func(mat->context->perm_global->invPerm[i],&rowSort[i].nEntsInRow,tmpcol,tmpval)) {
                        ERROR_LOG("Matrix construction function returned error");
                        ret = GHOST_ERR_UNKNOWN;
                    }
//                    printf("row %d <- %d rl %d\n",i,mat->context->perm_global->invPerm[i],rowSort[i].nEntsInRow);
                } else {
                    if (src->func(rowOffset+i,&rowSort[i].nEntsInRow,tmpcol,tmpval)) {
                        ERROR_LOG("Matrix construction function returned error");
                        ret = GHOST_ERR_UNKNOWN;
                    }
                }
                rowSort[i].row = i;
            }
            free(tmpval);
            free(tmpcol);
        }
        if (ret != GHOST_SUCCESS) {
            goto err;
        }

    } 
#if 0
    else {
        char *matrixPath = (char *)matrixSource;

        GHOST_CALL_GOTO(ghost_bincrs_rpt_read(rpt, matrixPath, rowOffset, nrows+1, NULL),err,ret);
        for (i=0; i<nrows; i++) {
            rowSort[i].nEntsInRow = rpt[i+1]-rpt[i];
            rowSort[i].row = i;
        }
    }
#endif

    for (c=0; c<nrows/scope; c++) {
        qsort(rowSort+c*scope, scope, sizeof(ghost_sorting_helper_t), ghost_cmp_entsperrow);
    }
    qsort(rowSort+c*scope, nrows-c*scope, sizeof(ghost_sorting_helper_t), ghost_cmp_entsperrow);
    
    for(i=0; i < nrows; ++i) {
        (mat->context->perm_local->invPerm)[i] = rowSort[i].row;
        (mat->context->perm_local->perm)[rowSort[i].row] = i;
    }
    for(i=0; i < nrows; ++i) {
        if (mat->context->perm_global) {
            //INFO_LOG("row %d lp %d gp %d lip %d gip %d",i,mat->context->perm_local->perm[i],mat->context->perm_global->perm[i],mat->context->perm_local->invPerm[i],mat->context->perm_global->invPerm[i]);
        }
    }
#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat->context->perm_local->cu_perm,mat->context->perm_local->perm,mat->context->perm_local->len*sizeof(ghost_gidx_t));
#endif
    
    goto out;

err:
    ERROR_LOG("Deleting permutations");
    if (mat->context->perm_local) {
        free(mat->context->perm_local->perm); mat->context->perm_local->perm = NULL;
        free(mat->context->perm_local->invPerm); mat->context->perm_local->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
        ghost_cu_free(mat->context->perm_local->cu_perm); mat->context->perm_local->cu_perm = NULL;
#endif
    }
    free(mat->context->perm_local); mat->context->perm_local = NULL;

out:

    free(rpt);
    free(rowSort);

    return ret;


}

ghost_error_t ghost_sparsemat_nrows(ghost_gidx_t *nrows, ghost_sparsemat_t *mat)
{
    if (!nrows) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *nrows = mat->context->gnrows;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_nnz(ghost_gidx_t *nnz, ghost_sparsemat_t *mat)
{
    if (!nnz) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_gidx_t lnnz = mat->nnz;

#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Allreduce(&lnnz,nnz,1,ghost_mpi_dt_gidx,MPI_SUM,mat->context->mpicomm));
#else
    *nnz = lnnz;
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_string(char **str, ghost_sparsemat_t *mat)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    int myrank;
    ghost_gidx_t nrows = 0;
    ghost_gidx_t nnz = 0;

    GHOST_CALL_RETURN(ghost_sparsemat_nrows(&nrows,mat));
    GHOST_CALL_RETURN(ghost_sparsemat_nnz(&nnz,mat));
    GHOST_CALL_RETURN(ghost_rank(&myrank, mat->context->mpicomm));


    char *matrixLocation;
    if (mat->traits->flags & GHOST_SPARSEMAT_DEVICE)
        matrixLocation = "Device";
    else if (mat->traits->flags & GHOST_SPARSEMAT_HOST)
        matrixLocation = "Host";
    else
        matrixLocation = "Default";


    ghost_header_string(str,"%s @ rank %d",mat->name,myrank);
    ghost_line_string(str,"Data type",NULL,"%s",ghost_datatype_string(mat->traits->datatype));
    ghost_line_string(str,"Matrix location",NULL,"%s",matrixLocation);
    ghost_line_string(str,"Total number of rows",NULL,"%"PRGIDX,nrows);
    ghost_line_string(str,"Total number of nonzeros",NULL,"%"PRGIDX,nnz);
    ghost_line_string(str,"Avg. nonzeros per row",NULL,"%.3f",(double)nnz/nrows);
    ghost_line_string(str,"Bandwidth",NULL,"%"PRGIDX,mat->bandwidth);
    ghost_line_string(str,"Avg. row band",NULL,"%.3f",mat->avgRowBand);
    ghost_line_string(str,"Avg. avg. row band",NULL,"%.3f",mat->avgAvgRowBand);
    ghost_line_string(str,"Smart row band",NULL,"%.3f",mat->smartRowBand);

    ghost_line_string(str,"Local number of rows",NULL,"%"PRLIDX,mat->nrows);
    ghost_line_string(str,"Local number of rows (padded)",NULL,"%"PRLIDX,mat->nrowsPadded);
    ghost_line_string(str,"Local number of nonzeros",NULL,"%"PRLIDX,mat->nnz);

    ghost_line_string(str,"Full   matrix format",NULL,"%s",mat->formatName(mat));
    if (mat->localPart) {
        ghost_line_string(str,"Local  matrix format",NULL,"%s",mat->localPart->formatName(mat->localPart));
        ghost_line_string(str,"Local  matrix symmetry",NULL,"%s",ghost_sparsemat_symmetry_string(mat->localPart->traits->symmetry));
        ghost_line_string(str,"Local  matrix size","MB","%u",mat->localPart->byteSize(mat->localPart)/(1024*1024));
    }
    if (mat->remotePart) {
        ghost_line_string(str,"Remote matrix format",NULL,"%s",mat->remotePart->formatName(mat->remotePart));
        ghost_line_string(str,"Remote matrix size","MB","%u",mat->remotePart->byteSize(mat->remotePart)/(1024*1024));
    }

    ghost_line_string(str,"Full   matrix size","MB","%u",mat->byteSize(mat)/(1024*1024));
    
    ghost_line_string(str,"Permuted",NULL,"%s",mat->traits->flags&GHOST_SPARSEMAT_PERMUTE?"Yes":"No");
    if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && mat->context->perm_global) {
        if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_line_string(str,"Permutation strategy",NULL,"Scotch%s",mat->traits->sortScope>1?"+Sorting":"");
            ghost_line_string(str,"Scotch ordering strategy",NULL,"%s",mat->traits->scotchStrat);
        } else {
            ghost_line_string(str,"Permutation strategy",NULL,"Sorting");
        }
        if (mat->traits->sortScope > 1) {
            ghost_line_string(str,"Sorting scope",NULL,"%d",mat->traits->sortScope);
        }
#ifdef GHOST_HAVE_MPI
        ghost_line_string(str,"Permutation scope",NULL,"%s",mat->context->perm_global->scope==GHOST_PERMUTATION_GLOBAL?"Across processes":"Local to process");
#endif
        ghost_line_string(str,"Permuted column indices",NULL,"%s",mat->traits->flags&GHOST_SPARSEMAT_NOT_PERMUTE_COLS?"No":"Yes");
        ghost_line_string(str,"Ascending columns in row",NULL,"%s",mat->traits->flags&GHOST_SPARSEMAT_NOT_SORT_COLS?"No":"Yes");
    }
    ghost_line_string(str,"Max row length (# rows)",NULL,"%d (%d)",mat->maxRowLen,mat->nMaxRows);
    ghost_line_string(str,"Row length variance",NULL,"%f",mat->variance);
    ghost_line_string(str,"Row length standard deviation",NULL,"%f",mat->deviation);
    ghost_line_string(str,"Row length coefficient of variation",NULL,"%f",mat->cv);

    mat->auxString(mat,str);
    ghost_footer_string(str);

    return GHOST_SUCCESS;

}

ghost_error_t ghost_sparsemat_tofile_header(ghost_sparsemat_t *mat, char *path)
{
    ghost_gidx_t mnrows,mncols,mnnz;
    GHOST_CALL_RETURN(ghost_sparsemat_nrows(&mnrows,mat));
    mncols = mnrows;
    GHOST_CALL_RETURN(ghost_sparsemat_nnz(&mnnz,mat));
    
    int32_t endianess = ghost_machine_bigendian();
    int32_t version = 1;
    int32_t base = 0;
    int32_t symmetry = GHOST_BINCRS_SYMM_GENERAL;
    int32_t datatype = mat->traits->datatype;
    int64_t nrows = (int64_t)mnrows;
    int64_t ncols = (int64_t)mncols;
    int64_t nnz = (int64_t)mnnz;

    size_t ret;
    FILE *filed;

    if ((filed = fopen64(path, "w")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s: %s",path,strerror(errno));
        return GHOST_ERR_IO;
    }

    if ((ret = fwrite(&endianess,sizeof(endianess),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&version,sizeof(version),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&base,sizeof(base),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&symmetry,sizeof(symmetry),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&datatype,sizeof(datatype),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&nrows,sizeof(nrows),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&ncols,sizeof(ncols),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&nnz,sizeof(nnz),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    fclose(filed);

    return GHOST_SUCCESS;

}

bool ghost_sparsemat_symmetry_valid(ghost_sparsemat_symmetry_t symmetry)
{
    if ((symmetry & (ghost_sparsemat_symmetry_t)GHOST_SPARSEMAT_SYMM_GENERAL) &&
            (symmetry & ~(ghost_sparsemat_symmetry_t)GHOST_SPARSEMAT_SYMM_GENERAL))
        return 0;

    if ((symmetry & (ghost_sparsemat_symmetry_t)GHOST_SPARSEMAT_SYMM_SYMMETRIC) &&
            (symmetry & ~(ghost_sparsemat_symmetry_t)GHOST_SPARSEMAT_SYMM_SYMMETRIC))
        return 0;

    return 1;
}

char * ghost_sparsemat_symmetry_string(ghost_sparsemat_symmetry_t symmetry)
{
    if (symmetry & GHOST_SPARSEMAT_SYMM_GENERAL)
        return "General";

    if (symmetry & GHOST_SPARSEMAT_SYMM_SYMMETRIC)
        return "Symmetric";

    if (symmetry & GHOST_SPARSEMAT_SYMM_SKEW_SYMMETRIC) {
        if (symmetry & GHOST_SPARSEMAT_SYMM_HERMITIAN)
            return "Skew-hermitian";
        else
            return "Skew-symmetric";
    } else {
        if (symmetry & GHOST_SPARSEMAT_SYMM_HERMITIAN)
            return "Hermitian";
    }

    return "Invalid";
}

void ghost_sparsemat_destroy_common(ghost_sparsemat_t *mat)
{
    if (!mat) {
        return;
    }

    free(mat->data); mat->data = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
}

ghost_error_t ghost_sparsemat_from_bincrs(ghost_sparsemat_t *mat, char *path)
{
    PERFWARNING_LOG("The current implementation of binCRS read-in is "
            "unefficient in terms of memory consumption!");
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_sparsemat_rowfunc_bincrs_initargs args;
    ghost_gidx_t dim[2];
    ghost_sparsemat_src_rowfunc_t src = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    
    src.func = &ghost_sparsemat_rowfunc_bincrs;
    args.filename = path;
    args.dt = mat->traits->datatype;
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETDIM,NULL,dim,&args)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_INIT,NULL,NULL,&args)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
   
    src.maxrowlen = dim[1];
    
    GHOST_CALL_GOTO(mat->fromRowFunc(mat,&src),err,ret);
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_FINALIZE,NULL,NULL,NULL)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;

}

ghost_error_t ghost_sparsemat_from_mm(ghost_sparsemat_t *mat, char *path)
{
    PERFWARNING_LOG("The current implementation of Matrix Market read-in is "
            "unefficient!");
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_sparsemat_rowfunc_mm_initargs args;
    ghost_gidx_t dim[2];
    ghost_sparsemat_src_rowfunc_t src = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    
    src.func = &ghost_sparsemat_rowfunc_mm;
    args.filename = path;
    args.dt = mat->traits->datatype;
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETDIM,NULL,dim,&args)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_INIT,NULL,NULL,&args)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    
    src.maxrowlen = dim[1];
    
    GHOST_CALL_GOTO(mat->fromRowFunc(mat,&src),err,ret);
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_FINALIZE,NULL,NULL,NULL)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;

}
