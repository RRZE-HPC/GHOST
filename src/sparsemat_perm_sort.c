#define _GNU_SOURCE
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/sparsemat.h"
#include "ghost/context.h"
#include "ghost/util.h"
#include "ghost/locality.h"

static int ghost_cmp_entsperrow(const void* a, const void* b, void *arg) 
{
    UNUSED(arg);
    return  ((ghost_sorting_helper*)b)->nEntsInRow - ((ghost_sorting_helper*)a)->nEntsInRow;
}

ghost_error ghost_sparsemat_perm_sort(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType, ghost_gidx scope)
{
    UNUSED(srcType);
    ghost_error ret = GHOST_SUCCESS;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    
    int me;    
    ghost_gidx i,c,nrows,rowOffset;
    ghost_sorting_helper *rowSort = NULL;
    ghost_gidx *rpt = NULL;

    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);

    

    if (mat->traits.sortScope > mat->nrows) {
        WARNING_LOG("Restricting the sorting scope to the number of matrix rows");
    }
    nrows = mat->nrows;
    rowOffset = mat->context->lfRow[me];
    GHOST_CALL_GOTO(ghost_malloc((void **)&rowSort,nrows * sizeof(ghost_sorting_helper)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(nrows+1) * sizeof(ghost_gidx)),err,ret);

    ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
    char *tmpval = NULL;
    ghost_gidx *tmpcol = NULL;
    rpt[0] = 0;
    int funcerrs = 0;

#pragma omp parallel private(i,tmpval,tmpcol)
    { 
        GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx)),ret);
        if (mat->context->perm_global && mat->context->perm_local) {
#pragma omp for schedule(runtime) reduction (+:funcerrs)
            for (i=0; i<nrows; i++) {
                funcerrs += src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowSort[i].nEntsInRow,tmpcol,tmpval,src->arg);
                rowSort[i].row = mat->context->perm_local->invPerm[i];
            }
        } else if (mat->context->perm_global) {
#pragma omp for schedule(runtime) reduction (+:funcerrs)
            for (i=0; i<nrows; i++) {
                funcerrs += src->func(mat->context->perm_global->invPerm[i],&rowSort[i].nEntsInRow,tmpcol,tmpval,src->arg);
                rowSort[i].row = i;
            }
        } else if (mat->context->perm_local) {
#pragma omp for schedule(runtime) reduction (+:funcerrs)
            for (i=0; i<nrows; i++) {
                funcerrs += src->func(rowOffset+mat->context->perm_local->invPerm[i],&rowSort[i].nEntsInRow,tmpcol,tmpval,src->arg);
                rowSort[i].row = mat->context->perm_local->invPerm[i];
            }
        } else {
#pragma omp for schedule(runtime) reduction (+:funcerrs)
            for (i=0; i<nrows; i++) {
                funcerrs += src->func(rowOffset+i,&rowSort[i].nEntsInRow,tmpcol,tmpval,src->arg);
                rowSort[i].row = i;
            }
        }
        free(tmpval);
        free(tmpcol);
    }
    if (funcerrs) {
        ERROR_LOG("Matrix construction function returned error");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }

    
    if (!mat->context->perm_local) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local,sizeof(ghost_permutation)),err,ret);
        mat->context->perm_local->scope = GHOST_PERMUTATION_LOCAL;
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->perm,sizeof(ghost_gidx)*nrows),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->invPerm,sizeof(ghost_gidx)*nrows),err,ret);
#ifdef GHOST_HAVE_CUDA
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->context->perm_local->cu_perm,sizeof(ghost_gidx)*nrows),err,ret);
#endif

        mat->context->perm_local->len = nrows;

        memset(mat->context->perm_local->perm,0,sizeof(ghost_gidx)*nrows);
        memset(mat->context->perm_local->invPerm,0,sizeof(ghost_gidx)*nrows);
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

#pragma omp parallel for
    for (c=0; c<nrows/scope; c++) {
        qsort_r(rowSort+c*scope, scope, sizeof(ghost_sorting_helper), ghost_cmp_entsperrow, NULL);
    }
    qsort_r(rowSort+(nrows/scope)*scope, nrows%scope, sizeof(ghost_sorting_helper), ghost_cmp_entsperrow, NULL);

#pragma omp parallel for    
    for(i=0; i < nrows; ++i) {
        (mat->context->perm_local->invPerm)[i] = rowSort[i].row;
        (mat->context->perm_local->perm)[rowSort[i].row] = i;
    }

#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat->context->perm_local->cu_perm,mat->context->perm_local->perm,mat->context->perm_local->len*sizeof(ghost_gidx));
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;


}

