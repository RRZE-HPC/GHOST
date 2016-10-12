#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#ifdef GHOST_HAVE_COLPACK
#include "ColPack/ColPackHeaders.h"
#endif

extern "C" ghost_error ghost_sparsemat_perm_color(ghost_context *ctx, ghost_sparsemat *mat)
{
    #ifdef GHOST_HAVE_COLPACK
    INFO_LOG("Create permutation from coloring");
    ghost_error ret = GHOST_SUCCESS;
    ghost_lidx *curcol = NULL;
    uint32_t** adolc = new uint32_t*[ctx->row_map->dim];
    std::vector<int>* colvec = NULL;
    uint32_t *adolc_data = NULL;
    ColPack::GraphColoring *GC=new ColPack::GraphColoring();
    bool oldperm = false;
    //ghost_permutation *oldperm = NULL;
    
    int me, i, j;
    ghost_gidx *rpt = NULL;
    ghost_lidx *rptlocal = NULL;
    ghost_gidx *col = NULL;
    ghost_lidx *collocal = NULL;
    ghost_lidx nnz = 0, nnzlocal = 0;
    int64_t pos=0;
    
    ghost_lidx ncols_halo_padded = ctx->row_map->dim;
    if (ctx->flags & GHOST_PERM_NO_DISTINCTION) {
        ncols_halo_padded = ctx->col_map->dimpad;
    }
    
    nnz = SPM_NNZ(mat); 
    
    GHOST_CALL_GOTO(ghost_rank(&me,ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(ctx->row_map->ldim[me]+1) * sizeof(ghost_gidx)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rptlocal,(ctx->row_map->ldim[me]+1) * sizeof(ghost_lidx)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&collocal,nnz * sizeof(ghost_lidx)),err,ret);
    
    rpt[0] = 0;
    rptlocal[0] = 0;
    
    for (ghost_lidx i=0; i<ctx->row_map->dim; i++) {
        rptlocal[i+1] = rptlocal[i];
        
        ghost_lidx orig_row = i;
        if (ctx->row_map->loc_perm) {
            orig_row = ctx->row_map->loc_perm_inv[i];
        }
        ghost_lidx * col = &mat->col[mat->chunkStart[orig_row]];
        ghost_lidx orig_row_len = mat->chunkStart[orig_row+1]-mat->chunkStart[orig_row];

        for(int j=0; j<orig_row_len; ++j) {
            if (col[j] < mat->context->row_map->dim) {
                collocal[nnzlocal] = col[j];
                nnzlocal++;
                rptlocal[i+1]++;
            }
        }
    }
    
    
    adolc_data = new uint32_t[nnzlocal+ctx->row_map->dim];
    
    for (i=0;i<ctx->row_map->dim;i++)
    {   
        adolc[i]=&(adolc_data[pos]);
        adolc_data[pos++]=rptlocal[i+1]-rptlocal[i];
        for (j=rptlocal[i];j<rptlocal[i+1];j++)
        {
            adolc_data[pos++]=collocal[j];
        }
    }
    
    GC->BuildGraphFromRowCompressedFormat(adolc, ctx->row_map->dim);
    
    COLPACK_CALL_GOTO(GC->DistanceTwoColoring(),err,ret);
    
    if (GC->CheckDistanceTwoColoring(2)) {
        ERROR_LOG("Error in coloring!");
        ret = GHOST_ERR_COLPACK;
        goto err;
    }
    
    ctx->ncolors = GC->GetVertexColorCount();
    
    
    if (!ctx->row_map->loc_perm) {
        GHOST_CALL_GOTO(ghost_malloc((void **)ctx->row_map->loc_perm,sizeof(ghost_permutation)),err,ret);
        //ctx->row_map->loc_perm->method = GHOST_PERMUTATION_UNSYMMETRIC; //you can also make it symmetric
        GHOST_CALL_GOTO(ghost_malloc((void **)ctx->row_map->loc_perm,sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)ctx->row_map->loc_perm_inv,sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);   
        GHOST_CALL_GOTO(ghost_malloc((void **)ctx->col_map->loc_perm,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)ctx->col_map->loc_perm_inv,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
        
        for(int i=0; i<ncols_halo_padded; ++i) {
            ctx->col_map->loc_perm[i] = i;
            ctx->col_map->loc_perm_inv[i] = i;
        }
        
        #ifdef GHOST_HAVE_CUDA
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)ctx->row_map->loc_perm->cu_perm,sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);
        #endif
        
    } else if(ctx->row_map->loc_perm == ctx->col_map->loc_perm) { // symmetrix permutation
        oldperm = true; //ctx->row_map->loc_perm;
//        ctx->row_map->loc_perm->method = GHOST_PERMUTATION_UNSYMMETRIC;//change to unsymmetric
        GHOST_CALL_GOTO(ghost_malloc((void **)ctx->col_map->loc_perm,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)ctx->col_map->loc_perm_inv,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
        
        for(int i=0; i<ncols_halo_padded; ++i) {
            ctx->col_map->loc_perm[i] = ctx->row_map->loc_perm[i];
            ctx->col_map->loc_perm_inv[i] = ctx->row_map->loc_perm_inv[i];
        }        
    } else {
        oldperm = true; //ctx->row_map->loc_perm;
    }
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->color_ptr,(ctx->ncolors+1)*sizeof(ghost_lidx)),err,ret);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(ctx->ncolors)*sizeof(ghost_lidx)),err,ret);
    memset(curcol,0,ctx->ncolors*sizeof(ghost_lidx));
    
    colvec = GC->GetVertexColorsPtr();
    
    
    for (i=0;i<ctx->ncolors+1;i++) {
        ctx->color_ptr[i] = 0;
    }
    
    for (i=0;i<ctx->row_map->dim;i++) {
        ctx->color_ptr[(*colvec)[i]+1]++;
    }
    
    for (i=1;i<ctx->ncolors+1;i++) {
        ctx->color_ptr[i] += ctx->color_ptr[i-1];
    }
    
    if (oldperm) {
        for (i=0;i<ctx->row_map->dim;i++) {
            int idx = ctx->row_map->loc_perm_inv[i];
            ctx->row_map->loc_perm[idx]  = curcol[(*colvec)[i]] + ctx->color_ptr[(*colvec)[i]];
            //ctx->row_map->loc_perm[i] = ctx->row_map->loc_perm_inv[curcol[(*colvec)[i]] + ctx->color_ptr[(*colvec)[i]]];
            curcol[(*colvec)[i]]++;
        }
    } else {
        for (i=0;i<ctx->row_map->dim;i++) {
            ctx->row_map->loc_perm[i] = curcol[(*colvec)[i]] + ctx->color_ptr[(*colvec)[i]];
            curcol[(*colvec)[i]]++;
        }
    }
    for (i=0;i<ctx->row_map->dim;i++) {
        ctx->row_map->loc_perm_inv[ctx->row_map->loc_perm[i]] = i;
    }
    
    goto out;
    err:
    
    out:
    delete [] adolc_data;
    delete [] adolc;
    delete GC;
    free(curcol);
    free(rpt);
    free(col);
    free(rptlocal);
    free(collocal);
    
    return ret;
    #else
    UNUSED(mat_out);
    UNUSED(matrixSource);
    UNUSED(srcType);
    ERROR_LOG("ColPack not available!");
    return GHOST_ERR_NOT_IMPLEMENTED;
    #endif
}

