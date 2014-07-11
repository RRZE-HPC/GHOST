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
#include "ghost/instr.h"

#include <libgen.h>
#ifdef GHOST_HAVE_SCOTCH
#ifdef GHOST_HAVE_MPI
#include <ptscotch.h>
#else
#include <scotch.h>
#endif
#endif

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
    (*mat)->fromCRS = NULL;
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
    (*mat)->permutation = NULL;
    (*mat)->bandwidth = 0;
    (*mat)->lowerBandwidth = 0;
    (*mat)->upperBandwidth = 0;
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

    if ((*mat)->traits->sortScope == GHOST_SPARSEMAT_SORT_GLOBAL) {
        (*mat)->traits->sortScope = (*mat)->context->gnrows;
    } else if ((*mat)->traits->sortScope == GHOST_SPARSEMAT_SORT_LOCAL) {
        (*mat)->traits->sortScope = (*mat)->nrows;
    }

#ifdef GHOST_GATHER_GLOBAL_INFO
    GHOST_CALL_GOTO(ghost_malloc((void **)&((*mat)->nzDist),sizeof(ghost_nnz_t)*(2*context->gnrows-1)),err,ret);
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

ghost_error_t ghost_sparsemat_sortrow(ghost_idx_t *col, char *val, size_t valSize, ghost_idx_t rowlen, ghost_idx_t stride)
{
    ghost_idx_t n;
    ghost_idx_t c;
    ghost_idx_t swpcol;
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

ghost_error_t ghost_sparsemat_fromfunc_common(ghost_sparsemat_t *mat, ghost_sparsemat_src_rowfunc_t *src)
{
    ghost_error_t ret = GHOST_SUCCESS;
    
    int nprocs = 1;
    int me;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    
#ifdef GHOST_GATHER_GLOBAL_INFO
    memset(mat->nzDist,0,sizeof(ghost_nnz_t)*(2*mat->context->gnrows-1));
#endif
    mat->lowerBandwidth = 0;
    mat->upperBandwidth = 0;
    
    if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
        mat->traits->flags |= GHOST_SPARSEMAT_PERMUTE;
    }

    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
        if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_sparsemat_perm_scotch(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } else {
            ghost_sparsemat_perm_sort(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC,mat->traits->sortScope);
        }
    } else {
        if (mat->traits->sortScope > 1) {
            WARNING_LOG("Ignoring sorting scope");
        }
        mat->traits->flags |= GHOST_SPARSEMAT_NOT_PERMUTE_COLS;
        mat->traits->flags |= GHOST_SPARSEMAT_NOT_SORT_COLS;
    }

    goto out;

err:

out:


    return ret;
}

ghost_error_t ghost_sparsemat_fromfile_common(ghost_sparsemat_t *mat, char *matrixPath, ghost_idx_t **rpt) 
{
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_idx_t i;
    ghost_bincrs_header_t header;

#ifdef GHOST_HAVE_MPI
    MPI_Request *req = NULL;
    MPI_Status *stat = NULL;
#endif

    ghost_bincrs_header_read(&header,matrixPath);
    int nprocs = 1;
    int me;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);

#ifdef GHOST_HAVE_MPI
    GHOST_CALL_GOTO(ghost_malloc((void **)&req,sizeof(MPI_Request)*nprocs),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&stat,sizeof(MPI_Status)*nprocs),err,ret);
#endif

    if (header.version != 1) {
        ERROR_LOG("Can not read version %d of binary CRS format!",header.version);
        return GHOST_ERR_IO;
    }

    if (header.base != 0) {
        ERROR_LOG("Can not read matrix with %d-based indices!",header.base);
        return GHOST_ERR_IO;
    }

    if (!ghost_sparsemat_symmetry_valid(header.symmetry)) {
        ERROR_LOG("Symmetry is invalid! (%d)",header.symmetry);
        return GHOST_ERR_IO;
    }

    if (header.symmetry != GHOST_BINCRS_SYMM_GENERAL) {
        ERROR_LOG("Can not handle symmetry different to general at the moment!");
        return GHOST_ERR_IO;
    }

    if (!ghost_datatype_valid(header.datatype)) {
        ERROR_LOG("Datatype is invalid! (%d)",header.datatype);
        return GHOST_ERR_IO;
    }

#ifdef GHOST_GATHER_GLOBAL_INFO
    memset(mat->nzDist,0,sizeof(ghost_nnz_t)*(2*mat->context->gnrows-1));
#endif
    mat->lowerBandwidth = 0;
    mat->upperBandwidth = 0;
    mat->name = basename(matrixPath);
    mat->traits->symmetry = header.symmetry;
    mat->ncols = (ghost_idx_t)header.ncols;
        
    if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
        mat->traits->flags |= GHOST_SPARSEMAT_PERMUTE;
    }

    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
        if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_sparsemat_perm_scotch(mat,matrixPath,GHOST_SPARSEMAT_SRC_FILE);
        } else {
            ghost_sparsemat_perm_sort(mat,matrixPath,GHOST_SPARSEMAT_SRC_FILE,mat->traits->sortScope);
        }
    } else {
        if (mat->traits->sortScope > 1) {
            WARNING_LOG("Ignoring sorting scope");
        }
        mat->traits->flags |= GHOST_SPARSEMAT_NOT_PERMUTE_COLS;
        mat->traits->flags |= GHOST_SPARSEMAT_NOT_SORT_COLS;
    }


    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED) {
        if (me == 0) {
            if (mat->context->flags & GHOST_CONTEXT_DIST_NZ) { // rpt has already been read
                *rpt = mat->context->rpt;
            } else { // read rpt and compute first entry and number of entries
                GHOST_CALL_GOTO(ghost_malloc_align((void **)rpt,(header.nrows+1) * sizeof(ghost_nnz_t), GHOST_DATA_ALIGNMENT),err,ret);
#pragma omp parallel for schedule(runtime) 
                for (i = 0; i < header.nrows+1; i++) {
                    (*rpt)[i] = 0;
                }
                GHOST_CALL_GOTO(ghost_bincrs_rpt_read(*rpt, matrixPath, 0, header.nrows+1, mat->permutation),err,ret); 
                mat->context->lfEnt[0] = 0;

                for (i=1; i<nprocs; i++){
                    mat->context->lfEnt[i] = (*rpt)[mat->context->lfRow[i]];
                }
                for (i=0; i<nprocs-1; i++){
                    mat->context->lnEnts[i] = mat->context->lfEnt[i+1] - mat->context->lfEnt[i] ;
                }

                mat->context->lnEnts[nprocs-1] = header.nnz - mat->context->lfEnt[nprocs-1];
            }
        }
#ifdef GHOST_HAVE_MPI
        MPI_CALL_GOTO(MPI_Bcast(mat->context->lfEnt,  nprocs, ghost_mpi_dt_idx, 0, mat->context->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Bcast(mat->context->lnEnts, nprocs, ghost_mpi_dt_idx, 0, mat->context->mpicomm),err,ret);

        if (me != 0) {
            GHOST_CALL_GOTO(ghost_malloc_align((void **)rpt,(mat->context->lnrows[me]+1)*sizeof(ghost_idx_t),GHOST_DATA_ALIGNMENT),err,ret);
#pragma omp parallel for schedule(runtime)
            for (i = 0; i < mat->context->lnrows[me]+1; i++) {
                (*rpt)[i] = 0;
            }
        }

        int msgcount = 0;

        for (i=0;i<nprocs;i++) 
            req[i] = MPI_REQUEST_NULL;

        if (me != 0) {
            MPI_CALL_GOTO(MPI_Irecv(*rpt,mat->context->lnrows[me]+1,ghost_mpi_dt_idx,0,me,mat->context->mpicomm,&req[msgcount]),err,ret);
            msgcount++;
        } else {
            for (i=1;i<nprocs;i++) {
                MPI_CALL_GOTO(MPI_Isend(&(*rpt)[mat->context->lfRow[i]],mat->context->lnrows[i]+1,ghost_mpi_dt_idx,i,i,mat->context->mpicomm,&req[msgcount]),err,ret);
                msgcount++;
            }
        }
        MPI_CALL_GOTO(MPI_Waitall(msgcount,req,stat),err,ret);

        for (i=0;i<mat->context->lnrows[me]+1;i++) {
            (*rpt)[i] -= mat->context->lfEnt[me];
        }

        (*rpt)[mat->context->lnrows[me]] = mat->context->lnEnts[me];
#endif
    }

    else if (mat->context->flags & GHOST_CONTEXT_REDUNDANT) {
        GHOST_CALL_GOTO(ghost_malloc_align((void **)rpt,(header.nrows+1) * sizeof(ghost_nnz_t), GHOST_DATA_ALIGNMENT),err,ret);
#pragma omp parallel for schedule(runtime) 
        for (i = 0; i < header.nrows+1; i++) {
            (*rpt)[i] = 0;
        }
        GHOST_CALL_GOTO(ghost_bincrs_rpt_read(*rpt, matrixPath, 0, header.nrows+1, mat->permutation),err,ret);
        for (i=0; i<nprocs; i++){
            mat->context->lfEnt[i] = 0;
            mat->context->lfRow[i] = 0;
        }
        for (i=0; i<nprocs; i++){
            mat->context->lnEnts[i] = header.nnz;
            mat->context->lnrows[i] = header.nrows;
        }
    }



    DEBUG_LOG(1,"local rows          = %"PRIDX,mat->context->lnrows[me]);
    DEBUG_LOG(1,"local rows (offset) = %"PRIDX,mat->context->lfRow[me]);
    DEBUG_LOG(1,"local entries          = %"PRNNZ,mat->context->lnEnts[me]);
    DEBUG_LOG(1,"local entries (offset) = %"PRNNZ,mat->context->lfEnt[me]);

    mat->nrows = mat->context->lnrows[me];
    mat->nnz = mat->context->lnEnts[me];

    goto out;

err:

out:


    return ret;
}

static int compareNZEPerRow( const void* a, const void* b ) 
{
    return  ((ghost_sorting_t*)b)->nEntsInRow - ((ghost_sorting_t*)a)->nEntsInRow;
}

ghost_error_t ghost_sparsemat_perm_sort(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType, ghost_idx_t scope)
{
    ghost_error_t ret = GHOST_SUCCESS;
    if (mat->permutation) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }
    
    int me;    
    ghost_idx_t i,c,nrows,rowOffset;
    ghost_sorting_t *rowSort = NULL;
    ghost_idx_t *rpt = NULL;

    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);

    

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->permutation,sizeof(ghost_permutation_t)),err,ret);
    if (mat->traits->sortScope > mat->nrows) {
        nrows = mat->context->gnrows;
        rowOffset = 0;
        mat->permutation->scope = GHOST_PERMUTATION_GLOBAL;
    } else {
        nrows = mat->nrows;
        rowOffset = mat->context->lfRow[me];
        mat->permutation->scope = GHOST_PERMUTATION_LOCAL;
    }
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->permutation->perm,sizeof(ghost_idx_t)*nrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->permutation->invPerm,sizeof(ghost_idx_t)*nrows),err,ret);
#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->permutation->cu_perm,sizeof(ghost_idx_t)*nrows),err,ret);
#endif

    mat->permutation->len = nrows;

    memset(mat->permutation->perm,0,sizeof(ghost_idx_t)*nrows);
    memset(mat->permutation->invPerm,0,sizeof(ghost_idx_t)*nrows);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&rowSort,nrows * sizeof(ghost_sorting_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(nrows+1) * sizeof(ghost_nnz_t)),err,ret);

    if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
        ghost_sparsemat_src_rowfunc_t *src = (ghost_sparsemat_src_rowfunc_t *)matrixSource;
        char *tmpval = NULL;
        ghost_idx_t *tmpcol = NULL;
        rpt[0] = 0;
#pragma omp parallel private(i,tmpval,tmpcol)
        { 
            GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_idx_t)),ret);
#pragma omp for schedule(runtime)
            for (i=0; i<nrows; i++) {
                if (src->func(rowOffset+i,&rowSort[i].nEntsInRow,tmpcol,tmpval)) {
                    ERROR_LOG("Matrix construction function returned error");
                    ret = GHOST_ERR_UNKNOWN;
                }
                rowSort[i].row = i;
            }
        }
        if (ret != GHOST_SUCCESS) {
            goto err;
        }

    } else {
        char *matrixPath = (char *)matrixSource;

        GHOST_CALL_GOTO(ghost_bincrs_rpt_read(rpt, matrixPath, rowOffset, nrows+1, NULL),err,ret);
        for (i=0; i<nrows; i++) {
            rowSort[i].nEntsInRow = rpt[i+1]-rpt[i];
            rowSort[i].row = i;
        }
    }

    for (c=0; c<nrows/scope; c++) {
        qsort(rowSort+c*scope, scope, sizeof(ghost_sorting_t), compareNZEPerRow );
    }
    qsort(rowSort+c*scope, nrows-c*scope, sizeof(ghost_sorting_t), compareNZEPerRow);
    
    for(i=0; i < nrows; ++i) {
        (mat->permutation->invPerm)[i] = rowSort[i].row;
        (mat->permutation->perm)[rowSort[i].row] = i;
    }
#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat->permutation->cu_perm,mat->permutation->perm,mat->permutation->len*sizeof(ghost_idx_t));
#endif
    
    goto out;

err:
    ERROR_LOG("Deleting permutations");
    free(mat->permutation->perm); mat->permutation->perm = NULL;
    free(mat->permutation->invPerm); mat->permutation->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
    ghost_cu_free(mat->permutation->cu_perm); mat->permutation->cu_perm = NULL;
#endif
    free(mat->permutation); mat->permutation = NULL;

out:

    free(rpt);
    free(rowSort);

    return ret;


}

ghost_error_t ghost_sparsemat_perm_scotch(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType)
{
#ifndef GHOST_HAVE_SCOTCH
    UNUSED(mat);
    UNUSED(matrixSource);
    UNUSED(srcType);
    WARNING_LOG("Scotch not available. Will not create matrix permutation!");
    return GHOST_SUCCESS;
#else
    GHOST_INSTR_START(scotch)
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_idx_t *rpt = NULL, *col = NULL, i, c;
    ghost_sorting_t *rowSort = NULL;
    ghost_idx_t *grpt = NULL;
    ghost_idx_t *col_loopless = NULL;
    ghost_idx_t *rpt_loopless = NULL;
    ghost_nnz_t nnz = 0;
    int me, nprocs;
#ifdef GHOST_HAVE_MPI
    SCOTCH_Dgraph * dgraph = NULL;
    SCOTCH_Strat * strat = NULL;
    SCOTCH_Dordering *dorder = NULL;
#else 
    SCOTCH_Graph * graph = NULL;
    SCOTCH_Strat * strat = NULL;
    SCOTCH_Ordering *order = NULL;
#endif

    if (mat->permutation) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }

    if (srcType == GHOST_SPARSEMAT_SRC_NONE) {
        ERROR_LOG("A valid matrix source has to be given!");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }

    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(mat->context->lnrows[me]+1) * sizeof(ghost_nnz_t)),err,ret);
    
    GHOST_INSTR_START(scotch_readin)
    if (srcType == GHOST_SPARSEMAT_SRC_FILE) {
        char *matrixPath = (char *)matrixSource;
        GHOST_CALL_GOTO(ghost_bincrs_rpt_read(rpt, matrixPath, mat->context->lfRow[me], mat->context->lnrows[me]+1, NULL),err,ret);
#pragma omp parallel for
        for (i=1;i<mat->context->lnrows[me]+1;i++) {
            rpt[i] -= rpt[0];
        }
        rpt[0] = 0;
        nnz = rpt[mat->context->lnrows[me]];
        GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_idx_t)),err,ret);
        GHOST_CALL_GOTO(ghost_bincrs_col_read(col, matrixPath, mat->context->lfRow[me], mat->context->lnrows[me], NULL,1),err,ret);

    } else if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
        ghost_sparsemat_src_rowfunc_t *src = (ghost_sparsemat_src_rowfunc_t *)matrixSource;
        char * tmpval = NULL;
        ghost_idx_t * tmpcol = NULL;

        ghost_idx_t rowlen;
        rpt[0] = 0;
#pragma omp parallel private (tmpval,tmpcol,i,rowlen) reduction(+:nnz)
        {
            ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
            ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_idx_t));
            
#pragma omp for
            for (i=0; i<mat->context->lnrows[me]; i++) {
                src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
                nnz += rowlen;
            }
            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_idx_t)),err,ret);
        
#pragma omp parallel private (tmpval,tmpcol,i,rowlen) reduction(+:nnz)
        {
            ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
            ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_idx_t));
#pragma omp for ordered
            for (i=0; i<mat->context->lnrows[me]; i++) {
#pragma omp ordered
                {
                    src->func(mat->context->lfRow[me]+i,&rowlen,&col[rpt[i]],tmpval);
                    rpt[i+1] = rpt[i] + rowlen;
                }
            }
            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
            
    }
    GHOST_INSTR_STOP(scotch_readin)

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->permutation,sizeof(ghost_permutation_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->permutation->perm,sizeof(ghost_idx_t)*mat->context->gnrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->permutation->invPerm,sizeof(ghost_idx_t)*mat->context->gnrows),err,ret);
#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->permutation->cu_perm,sizeof(ghost_idx_t)*mat->context->gnrows),err,ret);
#endif
    memset(mat->permutation->perm,0,sizeof(ghost_idx_t)*mat->context->gnrows);
    memset(mat->permutation->invPerm,0,sizeof(ghost_idx_t)*mat->context->gnrows);
    mat->permutation->scope = GHOST_PERMUTATION_GLOBAL;
    mat->permutation->len = mat->context->gnrows;

#ifdef GHOST_HAVE_MPI
    GHOST_INSTR_START(scotch_createperm)
    dgraph = SCOTCH_dgraphAlloc();
    if (!dgraph) {
        ERROR_LOG("Could not alloc SCOTCH graph");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_dgraphInit(dgraph,mat->context->mpicomm),err,ret);
    strat = SCOTCH_stratAlloc();
    if (!strat) {
        ERROR_LOG("Could not alloc SCOTCH strat");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_stratInit(strat),err,ret);
    dorder = SCOTCH_dorderAlloc();
    if (!dorder) {
        ERROR_LOG("Could not alloc SCOTCH order");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_dgraphBuild(dgraph, 0, mat->context->lnrows[me], mat->context->lnrows[me], rpt, rpt+1, NULL, NULL, nnz, nnz, col, NULL, NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphCheck(dgraph),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderInit(dgraph,dorder),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrder(strat,mat->traits->scotchStrat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderCompute(dgraph,dorder,strat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderPerm(dgraph,dorder,mat->permutation->perm+mat->context->lfRow[me]),err,ret);
    GHOST_INSTR_STOP(scotch_createperm)
    

    GHOST_INSTR_START(scotch_combineperm)
    // combine permutation vectors
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,mat->permutation->perm,mat->context->gnrows,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);

    // assemble inverse permutation
    for (i=0; i<mat->context->gnrows; i++) {
        mat->permutation->invPerm[mat->permutation->perm[i]] = i;
    }
    GHOST_INSTR_STOP(scotch_combineperm)
    

#else

    ghost_malloc((void **)&col_loopless,nnz*sizeof(ghost_idx_t));
    ghost_malloc((void **)&rpt_loopless,(mat->nrows+1)*sizeof(ghost_nnz_t));
    rpt_loopless[0] = 0;
    ghost_nnz_t nnz_loopless = 0;
    ghost_idx_t j;

    // eliminate loops by deleting diagonal entries
    for (i=0; i<mat->nrows; i++) {
        for (j=rpt[i]; j<rpt[i+1]; j++) {
            if (col[j] != i) {
                col_loopless[nnz_loopless] = col[j];
                nnz_loopless++;
            }
        }
        rpt_loopless[i+1] = nnz_loopless;
    }

    graph = SCOTCH_graphAlloc();
    if (!graph) {
        ERROR_LOG("Could not alloc SCOTCH graph");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_graphInit(graph),err,ret);
    strat = SCOTCH_stratAlloc();
    if (!strat) {
        ERROR_LOG("Could not alloc SCOTCH strat");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_stratInit(strat),err,ret);
    order = SCOTCH_orderAlloc();
    if (!order) {
        ERROR_LOG("Could not alloc SCOTCH order");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_graphBuild(graph, 0, mat->nrows, rpt_loopless, rpt_loopless+1, NULL, NULL, nnz_loopless, col_loopless, NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphCheck(graph),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderInit(graph,order,mat->permutation->perm,NULL,NULL,NULL,NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_stratGraphOrder(strat,mat->traits->scotchStrat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderCompute(graph,order,strat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderCheck(graph,order),err,ret);

    for (i=0; i<mat->nrows; i++) {
        mat->permutation->invPerm[mat->permutation->perm[i]] = i;
    }

#endif
    
    if (mat->traits->sortScope > 1) {
        GHOST_INSTR_START(post-permutation with sorting)
        ghost_idx_t nrows = mat->context->gnrows;
        ghost_idx_t scope = mat->traits->sortScope;
        
        GHOST_CALL_GOTO(ghost_malloc((void **)&rowSort,nrows * sizeof(ghost_sorting_t)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&grpt,nrows*sizeof(ghost_idx_t)),err,ret);

        memset(grpt,0,nrows*sizeof(ghost_idx_t));
        
        if (srcType == GHOST_SPARSEMAT_SRC_FILE) {
            char *matrixPath = (char *)matrixSource;
            GHOST_CALL_GOTO(ghost_bincrs_rpt_read(grpt, matrixPath, 0, nrows+1, NULL),err,ret);
            for (i=0; i<nrows; i++) {
                rowSort[mat->permutation->perm[i]].row = i;
                rowSort[mat->permutation->perm[i]].nEntsInRow = grpt[i+1]-grpt[i];
            
            }
        
        } else if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
            ghost_sparsemat_src_rowfunc_t *src = (ghost_sparsemat_src_rowfunc_t *)matrixSource;
            ghost_idx_t *dummycol;
            char *dummyval;
            GHOST_CALL_GOTO(ghost_malloc((void **)&dummycol,src->maxrowlen*sizeof(ghost_idx_t)),err,ret);
            GHOST_CALL_GOTO(ghost_malloc((void **)&dummyval,src->maxrowlen*mat->elSize),err,ret);

            for (i=0; i<nrows; i++) {
                rowSort[mat->permutation->perm[i]].row = i;
                src->func(i,&rowSort[mat->permutation->perm[i]].nEntsInRow,dummycol,dummyval);
            }
            
            free(dummyval);
            free(dummycol);
        }
       
        for (c=0; c<nrows/scope; c++) {
            qsort(rowSort+c*scope, scope, sizeof(ghost_sorting_t), compareNZEPerRow );
        }
        qsort(rowSort+c*scope, nrows-c*scope, sizeof(ghost_sorting_t), compareNZEPerRow);
        
        for(i=0; i < nrows; ++i) {
            (mat->permutation->invPerm)[i] =rowSort[i].row;
            (mat->permutation->perm)[rowSort[i].row] = i;
        }
        GHOST_INSTR_STOP(post-permutation with sorting)
    }
#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat->permutation->cu_perm,mat->permutation->perm,mat->permutation->len*sizeof(ghost_idx_t));
#endif
    goto out;
err:
    ERROR_LOG("Deleting permutations");
    free(mat->permutation->perm); mat->permutation->perm = NULL;
    free(mat->permutation->invPerm); mat->permutation->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
    ghost_cu_free(mat->permutation->cu_perm); mat->permutation->cu_perm = NULL;
#endif
    free(mat->permutation); mat->permutation = NULL;

out:
    free(rpt);
    free(grpt);
    free(rowSort);
    free(col);
    free(rpt_loopless);
    free(col_loopless);
#ifdef GHOST_HAVE_MPI
    SCOTCH_dgraphOrderExit(dgraph,dorder);
    SCOTCH_dgraphExit(dgraph);
    SCOTCH_stratExit(strat);
#else
    SCOTCH_graphOrderExit(graph,order);
    SCOTCH_graphExit(graph);
    SCOTCH_stratExit(strat);
#endif
    GHOST_INSTR_STOP(scotch)
    
    
    return ret;

#endif
}

ghost_error_t ghost_sparsemat_nrows(ghost_idx_t *nrows, ghost_sparsemat_t *mat)
{
    if (!nrows) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_nnz_t lnrows = mat->nrows;

    if (mat->context->flags & GHOST_CONTEXT_REDUNDANT) {
        *nrows = lnrows;
    } else {
#ifdef GHOST_HAVE_MPI
        MPI_CALL_RETURN(MPI_Allreduce(&lnrows,nrows,1,ghost_mpi_dt_idx,MPI_SUM,mat->context->mpicomm));
#else
        ERROR_LOG("Trying to get the number of matrix rows in a distributed context without MPI");
        return GHOST_ERR_UNKNOWN;
#endif
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_nnz(ghost_nnz_t *nnz, ghost_sparsemat_t *mat)
{
    if (!nnz) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_nnz_t lnnz = mat->nnz;

    if (mat->context->flags & GHOST_CONTEXT_REDUNDANT) {
        *nnz = lnnz;
    } else {
#ifdef GHOST_HAVE_MPI
        MPI_CALL_RETURN(MPI_Allreduce(&lnnz,nnz,1,ghost_mpi_dt_nnz,MPI_SUM,mat->context->mpicomm));
#else
        ERROR_LOG("Trying to get the number of matrix nonzeros in a distributed context without MPI");
        return GHOST_ERR_UNKNOWN;
#endif
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_string(char **str, ghost_sparsemat_t *mat)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    int myrank;
    ghost_idx_t nrows = 0;
    ghost_idx_t nnz = 0;

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
    ghost_line_string(str,"Total number of rows",NULL,"%"PRIDX,nrows);
    ghost_line_string(str,"Total number of nonzeros",NULL,"%"PRNNZ,nnz);
    ghost_line_string(str,"Avg. nonzeros per row",NULL,"%.3f",(double)nnz/nrows);
    ghost_line_string(str,"Bandwidth",NULL,"%"PRIDX,mat->bandwidth);

    ghost_line_string(str,"Local number of rows",NULL,"%"PRIDX,mat->nrows);
    ghost_line_string(str,"Local number of rows (padded)",NULL,"%"PRIDX,mat->nrowsPadded);
    ghost_line_string(str,"Local number of nonzeros",NULL,"%"PRIDX,mat->nnz);

    ghost_line_string(str,"Full   matrix format",NULL,"%s",mat->formatName(mat));
    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED)
    {
        if (mat->localPart) {
            ghost_line_string(str,"Local  matrix format",NULL,"%s",mat->localPart->formatName(mat->localPart));
            ghost_line_string(str,"Local  matrix symmetry",NULL,"%s",ghost_sparsemat_symmetry_string(mat->localPart->traits->symmetry));
            ghost_line_string(str,"Local  matrix size","MB","%u",mat->localPart->byteSize(mat->localPart)/(1024*1024));
        }
        if (mat->remotePart) {
            ghost_line_string(str,"Remote matrix format",NULL,"%s",mat->remotePart->formatName(mat->remotePart));
            ghost_line_string(str,"Remote matrix size","MB","%u",mat->remotePart->byteSize(mat->remotePart)/(1024*1024));
        }
    } else {
        ghost_line_string(str,"Full   matrix symmetry",NULL,"%s",ghost_sparsemat_symmetry_string(mat->traits->symmetry));
    }

    ghost_line_string(str,"Full   matrix size","MB","%u",mat->byteSize(mat)/(1024*1024));
    
    ghost_line_string(str,"Permuted",NULL,"%s",mat->traits->flags&GHOST_SPARSEMAT_PERMUTE?"Yes":"No");
    if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && mat->permutation) {
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
        ghost_line_string(str,"Permutation scope",NULL,"%s",mat->permutation->scope==GHOST_PERMUTATION_GLOBAL?"Across processes":"Local to process");
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
    ghost_idx_t mnrows,mncols,mnnz;
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
    if ((symmetry & GHOST_SPARSEMAT_SYMM_GENERAL) &&
            (symmetry & ~GHOST_SPARSEMAT_SYMM_GENERAL))
        return 0;

    if ((symmetry & GHOST_SPARSEMAT_SYMM_SYMMETRIC) &&
            (symmetry & ~GHOST_SPARSEMAT_SYMM_SYMMETRIC))
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
    if (mat->permutation) {
        free(mat->permutation->perm); mat->permutation->perm = NULL;
        free(mat->permutation->invPerm); mat->permutation->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
        ghost_cu_free(mat->permutation->cu_perm); mat->permutation->cu_perm = NULL;
#endif
    }
    free(mat->permutation); mat->permutation = NULL;
    free(mat->data); mat->data = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
}


