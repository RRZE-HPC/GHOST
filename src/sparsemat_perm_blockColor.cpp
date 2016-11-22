#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#include "ghost/omp.h"
#include <omp.h>
#ifdef GHOST_HAVE_COLPACK
#include "ColPack/ColPackHeaders.h"
#endif
#include <vector>
#include <limits.h>

extern "C" ghost_error ghost_sparsemat_blockColor(ghost_context *ctx, ghost_sparsemat *mat) 
{
    
    INFO_LOG("Create partition and permute (Block coloring)");
    ghost_error ret = GHOST_SUCCESS;
    ghost_lidx *curcol = NULL; 
    bool old_perm = true;
    //a counter for number of rows to be multicolored
    ghost_lidx ctr_nrows_MC = 0;
    ghost_lidx ctr_nnz_MC = 0;
    int total_bw =0;
    int max_col_idx = 0;
    ghost_lidx local_size;
    ghost_lidx *rhs_split = NULL;
    ghost_lidx *zone;
    ghost_lidx nrows =0;
    std::vector<int>  colvec;   
    
    #ifdef GHOST_HAVE_COLPACK
    ghost_lidx offs = 0;
    ghost_lidx pos = 0;
    uint32_t** adolc = NULL; 
    uint32_t *adolc_data = NULL;
    #endif
   
    // We have to access mat->context because this already has communication information 
    ghost_gidx ncols_halo_padded = mat->context->col_map->dim;

    #ifdef GHOST_HAVE_COLPACK
    ColPack::BipartiteGraphPartialColoringInterface *GC;
    adolc = new uint32_t*[ctx->row_map->dim];
    #endif
    
    int nthread;
    
    #ifdef GHOST_HAVE_OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        nthread = ghost_omp_nthread();
    }
    #else
    nthread = 1;
    #endif
    
    int n_zones = nthread;
    ctx->kacz_setting.active_threads = n_zones;
    
    //printf("...NZONES... = %d\n", n_zones);
    int me;
    
    
    GHOST_CALL_GOTO(ghost_rank(&me,ctx->mpicomm),err,ret);
    
    nrows = ctx->row_map->ldim[me];
    
    
    //  ghost_lidx *row_ptr = mat_out->sell->chunkStart;
    //  ghost_lidx *col_ptr = mat_out->sell->col;
    //  ghost_lidx nrows = ctx->row_map->dim;
    
    
    //  ghost_lidx n_t_zones = n_zones-1; 
    //
    /*    if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
     *       ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
     *       ghost_gidx * tmpcol = NULL;
     *       char * tmpval = NULL;
     * 
     *       ghost_lidx rowlen;
     * 
     *     
     * #pragma omp parallel private(tmpval,tmpcol,rowlen) 
     *  {
     *       ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
     *       ghost_malloc((void **)&tmpval,src->maxrowlen*mat_out->elSize); 
     *       int ctr = 0;
     *  #pragma omp parallel for reduction(max:lower_bw) reduction(max:upper_bw) reduction(+:ctr)
     *       for (int i=0; i<ctx->row_map->ldim[me]; i++) {
     *        	if (ctx->perm_global && ctx->perm_local) {
     *                 	src->func(ctx->row_map->glb_perm_inv[ctx->row_map->loc_perm_inv[i]],&rowlen,tmpcol,tmpval,src->arg);
} else if (ctx->perm_global) {
    src->func(ctx->row_map->glb_perm_inv[i],&rowlen,tmpcol,tmpval,src->arg);
} else if (ctx->perm_local) {
    src->func(ctx->row_map->goffs[me]+ctx->row_map->loc_perm_inv[i],&rowlen,tmpcol,tmpval,src->arg);
} else {
    src->func(ctx->row_map->goffs[me]+i,&rowlen,tmpcol,tmpval,src->arg);
}

ghost_gidx start_col = ctx->row_map->dim + ctx->nrowspadded;
ghost_gidx end_col   = 0;

if(ctx->perm_local){
    if(ctx->col_map->loc_perm == NULL) {
        for(int j=0; j<rowlen; ++j) {
            start_col = MIN(start_col, ctx->row_map->loc_perm[tmpcol[j]]);
            end_col   = MAX(end_col, ctx->row_map->loc_perm[tmpcol[j]]);
}
} else {
    for(int j=0; j<rowlen; ++j) {
        start_col = MIN(start_col, ctx->col_map->loc_perm[tmpcol[j]]);
        end_col   = MAX(end_col, ctx->col_map->loc_perm[tmpcol[j]]);
}
}
} else {
    for(int j=0; j<rowlen; ++j) {
        start_col = MIN(start_col, tmpcol[j]);
        end_col   = MAX(end_col, tmpcol[j]);
}
}
lower_bw = MAX(lower_bw, i-start_col);
upper_bw = MAX(upper_bw , end_col - i);
max_col_idx = MAX(max_col_idx, end_col);
}

free(tmpcol);
free(tmpval);
}
}*/
    //std::cout<<"nrows ="<<ctx->row_map->dim<<std::endl;
    //std::cout<<"check"<<row_ptr[ctx->row_map->dim-1]<<std::endl;
    
    total_bw = ctx->bandwidth;
    max_col_idx = ctx->maxColRange;
    
    //approximate, currently last thread is set to not have anything in transitional sweeps;
    //since large bandwidths might create problems (TODO optimise this value)
    local_size = static_cast<int>( ((max_col_idx+1)-0*total_bw) / n_zones); //will have to find a method to balance the load for the last thread, 	
    // even if the last thread gets a bit less load its fine since the excess load 
    // is evenly spread among rest of the threads
    
    rhs_split = new int[n_zones+2];
    
    for(int i=0; i<n_zones+1; ++i) {
        rhs_split[i] = i*local_size;
        //printf("rhs_split[%d] = %d\n",i,rhs_split[i]); 
    }
    
    rhs_split[n_zones+1] = max_col_idx+1;
    //printf("%d\n",(int)rhs_split[n_zones+1]);
    
    
    
    zone = new ghost_lidx[nrows]; 
    
    for(int i=0; i<nrows; ++i) {
        zone[i] = -(n_zones+1) ; //an invalid number
    }
    
    #pragma omp parallel for reduction(+:ctr_nrows_MC,ctr_nnz_MC) 
    for (ghost_lidx i=0; i<ctx->row_map->dim; i++) {
        
        ghost_lidx orig_row = i;
        if (ctx->row_map->loc_perm) {
            orig_row = ctx->row_map->loc_perm_inv[i];
        }
        
        ghost_lidx start_col =  INT_MAX;
        ghost_lidx end_col   =  0;

        ghost_lidx * col = &mat->col[mat->chunkStart[orig_row]];
        ghost_lidx orig_row_len = mat->chunkStart[orig_row+1]-mat->chunkStart[orig_row];
        
        if(ctx->row_map->loc_perm) {
            if(ctx->col_map->loc_perm == NULL) {
                for(int j=0; j<orig_row_len; ++j) {
                    start_col = MIN(start_col, ctx->row_map->loc_perm[col[j]]);
                    end_col   = MAX(end_col, ctx->row_map->loc_perm[col[j]]);
                } 
            } else {
                for(int j=0; j<orig_row_len; ++j) {
                    start_col = MIN(start_col, ctx->col_map->loc_perm[col[j]]);
                    end_col   = MAX(end_col, ctx->col_map->loc_perm[col[j]]);
                }
            }
        } else {
            for(int j=0; j<orig_row_len; ++j) {
                start_col = MIN(start_col, col[j]);
                end_col   = MAX(end_col, col[j]);
            }
        }
        
        
        for(int k=0; k<n_zones; ++k) {
            //pure zone
            if(start_col >= rhs_split[k] && end_col < rhs_split[k+1]) {
                zone[i] = 2*k; 
            }
            //transition zone
            else if(zone[i]<0 && (start_col >= rhs_split[k] && end_col < rhs_split[k+2]) ) {
                zone[i] = 2*k+1;
            }
            //else one can also add the rest of them for Multicoloring
            else if(k==n_zones-1 && zone[i]<0) {
                zone[i] = 2*n_zones;      //put Multicolor as last segment, but for time being we do this zone using single thread          
                ctr_nrows_MC+=1;
                ctr_nnz_MC += orig_row_len;
                //TODO  - increase zone by 1        zone[i] = 2*n_zones + 1;
            } 
        }
    }
    
    INFO_LOG("NO. of Rows Multicolored = %d",ctr_nrows_MC);
            
    
    /*       for(int i=0; i<nrows; ++i){
     *    	    for(int k=0; k<n_zones; ++k){
     *        	    //pure zone
     *            	if(col_ptr[row_ptr[i]] >= rhs_split[k] && col_ptr[row_ptr[i+1]-1] <= rhs_split[k+1]) {
     *            	    zone[i] = 2*k;
}
//transition zone
else if(k>0 && zone[i]<0 && (col_ptr[row_ptr[i]] >= rhs_split[k-1] && col_ptr[row_ptr[i+1]-1] <= rhs_split[k+1]) ){
    zone[i] = 2*k-1;                  
}
}
}
*/
    
    if (!ctx->row_map->loc_perm) {
        //this branch if no local permutations are carried out before
        WARNING_LOG("The matrix has not been RCM permuted, BLOCK coloring works better for matrix with small bandwidths");
        old_perm = false;
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->row_map->loc_perm,sizeof(ghost_gidx)*ctx->row_map->dim), err, ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->row_map->loc_perm_inv,sizeof(ghost_gidx)*ctx->row_map->dim), err, ret);
        
//        ctx->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;
        
        //anyway separate both permutations 
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->col_map->loc_perm,sizeof(ghost_gidx)*ncols_halo_padded), err, ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->col_map->loc_perm_inv,sizeof(ghost_gidx)*ncols_halo_padded), err, ret);
        
        #pragma omp parallel for         
        for(int i=0; i<ncols_halo_padded; ++i) { 
            ctx->col_map->loc_perm[i] = i;
            ctx->col_map->loc_perm_inv[i] = i;
        } 
        
    } else if(ctx->row_map->loc_perm == ctx->col_map->loc_perm) {
        //this branch if no unsymmetric permutations have been carried out before
        
        if(ctx->row_map->dim != ctx->col_map->dim) {
            ERROR_LOG("Trying to do symmetric permutation on non-squared matrix");
        }
        
        //now make it unsymmetric
  //      ctx->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;
        
        //anyway separate both permutations 
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->col_map->loc_perm,sizeof(ghost_gidx)*ncols_halo_padded), err, ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->col_map->loc_perm_inv,sizeof(ghost_gidx)*ncols_halo_padded), err, ret);
        
        #pragma omp parallel for         
        for(int i=0; i<ncols_halo_padded; ++i) {
            ctx->col_map->loc_perm[i] =  ctx->row_map->loc_perm[i];
            ctx->col_map->loc_perm_inv[i] =  ctx->row_map->loc_perm_inv[i];
        } 
    }
    
    ctx->nzones = 2*n_zones ;
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->zone_ptr,(ctx->nzones+2)*sizeof(ghost_lidx)), err, ret);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(ctx->nzones+1)*sizeof(ghost_lidx)), err, ret);
    memset(curcol,0,(ctx->nzones+1)*sizeof(ghost_lidx));
    
    for (int i=0;i<ctx->nzones+2;i++) {
        ctx->zone_ptr[i] = 0;
    }
    
    for (int i=0;i<ctx->row_map->dim;i++) {
        ctx->zone_ptr[zone[i]+1]++;
    }
    
    for (int i=1;i<ctx->nzones+2;i++) {
        ctx->zone_ptr[i] += ctx->zone_ptr[i-1];
    }
    
    ctx->zone_ptr[ctx->nzones+1] = nrows;
    
    if(old_perm){
        for (int i=0;i<ctx->row_map->dim;i++) {
            ctx->row_map->loc_perm[ctx->row_map->loc_perm_inv[i]] = curcol[zone[i]] + ctx->zone_ptr[zone[i]];
            curcol[zone[i]]++;
        }
    } else {  
        for (int i=0;i<ctx->row_map->dim;i++) {
            ctx->row_map->loc_perm[i] = curcol[zone[i]] + ctx->zone_ptr[zone[i]];
            curcol[zone[i]]++;
        }
    }
    
    for (int i=0;i<ctx->row_map->dim;i++) {
        ctx->row_map->loc_perm_inv[ctx->row_map->loc_perm[i]] = i;
    }
    
    #ifdef GHOST_HAVE_COLPACK         
    INFO_LOG("Create permutation from coloring");
    free(curcol);
    //now build adolc for multicoloring
    //int nrows_to_mc = ctx->zone_ptr[ctx->nzones+1]-ctx->zone_ptr[ctx->nzones];
    offs = ctx->zone_ptr[ctx->nzones];
    adolc_data = new uint32_t[ctr_nnz_MC + ctr_nrows_MC];
    pos = 0;
    //printf("MC NNZ %d NROWS %d\n\n",ctr_nnz_MC,ctr_nrows_MC);
    
    //here if a previous permutation like RCM was carried out we either need the original unpermuted matrix or a totally permuted matrix
        
    for(int i=0; i< ctr_nrows_MC; ++i) {
        ghost_lidx orig_row = offs+i;
        if (ctx->row_map->loc_perm) {
            orig_row = ctx->row_map->loc_perm_inv[offs+i];
        }
        ghost_lidx orig_row_len = mat->chunkStart[orig_row+1]-mat->chunkStart[orig_row];
        ghost_lidx * col = &mat->col[mat->chunkStart[orig_row]];
        
        adolc[i] = &(adolc_data[pos]);
        adolc_data[pos++] = orig_row_len;
        
        if (ctx->col_map->loc_perm) {
            for(int j=0; j<orig_row_len; ++j) {
                adolc_data[pos++] = ctx->col_map->loc_perm_inv[col[j]];
            }
        } else {
            for(int j=0; j<orig_row_len; ++j) {
                adolc_data[pos++] = col[j];
            }
        }
    }
    
    //build Bipartite Graph
    GC =  new ColPack::BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC,adolc,ctr_nrows_MC,ncols_halo_padded); 
    COLPACK_CALL_GOTO(GC->PartialDistanceTwoColoring("NATURAL","ROW_PARTIAL_DISTANCE_TWO"),err,ret);
    
    if(!GC->CheckPartialDistanceTwoRowColoring()) {
        ERROR_LOG("Error in coloring!");
        ret = GHOST_ERR_COLPACK;
        goto err;
    }
    
    ctx->ncolors = GC->GetVertexColorCount();
    GC->GetVertexPartialColors(colvec);
    INFO_LOG("No. of Colors = %d",ctx->ncolors);
    
    ghost_malloc((void **)&ctx->color_ptr,(ctx->ncolors+1)*sizeof(ghost_lidx)); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(ctx->ncolors)*sizeof(ghost_lidx)),err,ret);
    memset(curcol,0,ctx->ncolors*sizeof(ghost_lidx));
    
    for (int i=0;i<ctx->ncolors+1;i++) {
        ctx->color_ptr[i] = 0; 
    }
    
    for (int i=0;i<ctr_nrows_MC;i++) {
        ctx->color_ptr[colvec[i]+1]++;
    }
    
    for (int i=1;i<ctx->ncolors+1;i++) {
        ctx->color_ptr[i] += ctx->color_ptr[i-1];
    }
    
    for(int i=0; i<ctx->ncolors+1;++i) {
        ctx->color_ptr[i] = ctx->color_ptr[i] + offs; //add offset to it
    }
    
    for (int i=0;i<ctr_nrows_MC;i++) {
        int idx = ctx->row_map->loc_perm_inv[i+offs];
        ctx->row_map->loc_perm[idx]  = curcol[colvec[i]] + ctx->color_ptr[colvec[i]];
        curcol[colvec[i]]++;
    }
    
    for (int i=0;i<ctx->row_map->dim;i++) {
        ctx->row_map->loc_perm_inv[ctx->row_map->loc_perm[i]] = i;
    }
    
    #else
    WARNING_LOG("COLPACK is not available, only 1 thread would be used")
    ctx->ncolors = 1;
    ghost_malloc((void **)&ctx->color_ptr,(ctx->ncolors+1)*sizeof(ghost_lidx)); 
    
    for(int i=0; i<ctx->ncolors+1; ++i) {
        ctx->color_ptr[i] = ctx->zone_ptr[ctx->nzones+i];
    }
    
    #endif 
    
    double MC_percent;
    MC_percent = ((double)ctr_nrows_MC/ctx->row_map->ldim[me])*100.;
    //TODO : quantify this and give a break point
    if( MC_percent > 5  ) {
        WARNING_LOG("%3.2f %% rows would be Multicolored, try reducing number of threads", MC_percent);
    }
    
    
    goto out;
    
    err:
    
    out:
   
#ifdef GHOST_HAVE_COLPACK 
    delete [] adolc_data;
    delete [] adolc;
    delete GC;
#endif
    delete [] rhs_split;
    delete [] zone;
    free(curcol);
    
    return ret;
}



