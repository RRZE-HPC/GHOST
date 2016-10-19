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

ghost_error ghost_sparsemat_perm_sort(ghost_context *ctx, ghost_sparsemat *mat, ghost_lidx scope)
{
    ghost_error ret = GHOST_SUCCESS;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    
    ghost_lidx i,c,nrows;
    ghost_sorting_helper *rowSort = NULL;
    ghost_lidx *rpt = mat->chunkStart;

    if (mat->traits.sortScope > SPM_NROWS(mat)) {
        WARNING_LOG("Restricting the sorting scope to the number of matrix rows");
    }
    nrows = SPM_NROWS(mat);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rowSort,nrows * sizeof(ghost_sorting_helper)),err,ret);

    #pragma omp parallel for 
    for (i=0; i<nrows; i++) {
        
        ghost_lidx orig_row = i;
        if (ctx->row_map->loc_perm) {
            orig_row = ctx->row_map->loc_perm_inv[i];
        }
        rowSort[i].nEntsInRow = rpt[orig_row+1]-rpt[orig_row];
        rowSort[i].row = orig_row;
    }
        
    if (!ctx->row_map->loc_perm) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->row_map->loc_perm,sizeof(ghost_lidx)*nrows),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->row_map->loc_perm_inv,sizeof(ghost_lidx)*nrows),err,ret);
        ctx->col_map->loc_perm = NULL;
        ctx->col_map->loc_perm_inv = NULL;

#ifdef GHOST_HAVE_CUDA
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)ctx->row_map->cu_loc_perm,sizeof(ghost_lidx)*nrows),err,ret);
#endif

        memset(ctx->row_map->loc_perm,0,sizeof(ghost_lidx)*nrows);
        memset(ctx->row_map->loc_perm_inv,0,sizeof(ghost_lidx)*nrows);
    }
    
#pragma omp parallel for
    for (c=0; c<nrows/scope; c++) {
        qsort_r(rowSort+c*scope, scope, sizeof(ghost_sorting_helper), ghost_cmp_entsperrow, NULL);
    }
    qsort_r(rowSort+(nrows/scope)*scope, nrows%scope, sizeof(ghost_sorting_helper), ghost_cmp_entsperrow, NULL);

#pragma omp parallel for    
    for(i=0; i < nrows; ++i) {
        ctx->row_map->loc_perm_inv[i] = rowSort[i].row;
        ctx->row_map->loc_perm[rowSort[i].row] = i;
    }

#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(ctx->row_map->cu_loc_perm,ctx->row_map->loc_perm,SPM_NROWS(mat)*sizeof(ghost_gidx));
#endif
    
    goto out;

err:
    ERROR_LOG("Deleting permutations");
    if (ctx->row_map->loc_perm) {
        free(ctx->row_map->loc_perm); ctx->row_map->loc_perm = NULL;
        free(ctx->row_map->loc_perm_inv); ctx->row_map->loc_perm_inv = NULL;
#ifdef GHOST_HAVE_CUDA
        ghost_cu_free(ctx->row_map->cu_loc_perm); ctx->row_map->cu_loc_perm = NULL;
#endif
    }

out:

    free(rowSort);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;


}

