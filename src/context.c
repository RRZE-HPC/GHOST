#include "ghost/config.h"
#include "ghost/core.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/context.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#include "ghost/matrixmarket.h"
#include "ghost/log.h"
#include "ghost/omp.h"
#include "ghost/machine.h"
#include "ghost/bench.h"
#include "ghost/map.h"
#include <float.h>
#include <math.h>

ghost_error ghost_context_create(ghost_context **context, ghost_gidx gnrows, ghost_gidx gncols, ghost_context_flags_t context_flags, ghost_mpi_comm comm, double weight) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    if (weight < 0) {
        ERROR_LOG("Negative weight");
        return GHOST_ERR_INVALID_ARG;
    }

    int nranks, me;
    ghost_error ret = GHOST_SUCCESS;

    GHOST_CALL_GOTO(ghost_nrank(&nranks, comm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, comm),err,ret);
    
    if (fabs(weight) < DBL_MIN) {
        double max_bw=0.0;
        double avgweight;
        int withinavg = 0;
        int totalwithinavg = 0;

        GHOST_CALL_GOTO(ghost_bench_bw(GHOST_BENCH_UPDATE,&weight,&max_bw),err,ret);
        MPI_CALL_GOTO(MPI_Allreduce(&weight,&avgweight,1,MPI_DOUBLE,MPI_SUM,comm),err,ret);
        avgweight /= nranks;
        if (fabs(weight-avgweight)/avgweight < 0.1) {
            withinavg = 1;
        }
        MPI_CALL_GOTO(MPI_Allreduce(&withinavg,&totalwithinavg,1,MPI_INT,MPI_SUM,comm),err,ret);
        if (nranks > 1) {
            if (totalwithinavg == nranks) {
                INFO_LOG("The bandwidths of all processes differ by less than 10%%, the weights will be fixed to 1.0 to avoid artifacts.");
                weight = 1.0;
            } else {
                INFO_LOG("The bandwidths of all processes differ by more than 10%%, automatically setting weight to %.2f according to UPDATE bandwidth!",weight);
            }
        }
    }
    if (!gnrows) {
        ERROR_LOG("The global number of rows (and columns for non-square matrices) must not be zero!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (!gncols) {
        gncols = gnrows;
    }
    
    
    ghost_lidx *target_rows = NULL;
    char *tmpval = NULL;
    ghost_gidx *tmpcol = NULL;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)context,sizeof(ghost_context)),err,ret);
    (*context)->weight=weight;
    (*context)->flags = context_flags;
    (*context)->mpicomm = comm;
    (*context)->mpicomm_parent = MPI_COMM_NULL;
    (*context)->wishes   = NULL;
    (*context)->dues     = NULL;
    (*context)->hput_pos = NULL;
    (*context)->cu_duelist = NULL;
    (*context)->duelist = NULL;
    (*context)->wishlist = NULL;
    (*context)->dues = NULL;
    (*context)->wishes = NULL;
    (*context)->duepartners = NULL;
    (*context)->nduepartners = 0;
    (*context)->wishpartners = NULL;
    (*context)->nwishpartners = 0;
    (*context)->entsInCol = NULL;
    (*context)->ncolors = 0;
    (*context)->color_ptr = NULL;
    (*context)->nzones = 0;
    (*context)->zone_ptr = NULL;
    (*context)->kacz_setting.kacz_method = GHOST_KACZ_METHOD_MC;
    (*context)->kacz_setting.active_threads = 0;
    (*context)->bandwidth = 0;
    (*context)->lowerBandwidth = 0;
    (*context)->upperBandwidth = 0;
    (*context)->avg_ptr = NULL;
    (*context)->mapAvg = NULL;
    (*context)->mappedDuelist = NULL;
    (*context)->nrankspresent = NULL;   
    (*context)->nmats = 1; 
        

    GHOST_CALL_GOTO(ghost_map_create(&((*context)->row_map),gnrows,comm,GHOST_MAP_ROW,GHOST_MAP_DEFAULT),err,ret);
    GHOST_CALL_GOTO(ghost_map_create(&((*context)->col_map),gncols,comm,GHOST_MAP_COL,GHOST_MAP_DEFAULT),err,ret);

    if (!((*context)->flags & GHOST_CONTEXT_DIST_NZ)) {
        (*context)->flags |= (ghost_context_flags_t)GHOST_CONTEXT_DIST_ROWS;
    }

    DEBUG_LOG(1,"Context created successfully");
    goto out;
err:
    free(*context); *context = NULL;


out:
    free(tmpval); tmpval = NULL;
    free(tmpcol); tmpcol = NULL;
    free(target_rows); target_rows = NULL;
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;
}

ghost_error ghost_context_string(char **str, ghost_context *context)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);
    int nranks;
    GHOST_CALL_RETURN(ghost_nrank(&nranks, context->mpicomm));

    ghost_header_string(str,"Context");
    ghost_line_string(str,"MPI processes",NULL,"%d",nranks);
    //ghost_line_string(str,"Number of rows",NULL,"%"PRGIDX,context->gnrows);
    ghost_line_string(str,"Work distribution scheme",NULL,"%s",ghost_context_workdist_string(context->flags));
    ghost_footer_string(str);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;

}

void ghost_context_destroy(ghost_context *context)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TEARDOWN);
    
    if (context) {
        if (context->wishlist) {
            free(context->wishlist[0]);
        }
        if (context->duelist) {
            free(context->duelist[0]);
        }
#ifdef GHOST_HAVE_CUDA
        if (context->cu_duelist) {
            ghost_cu_free(context->cu_duelist[0]);
        }
#endif
        free(context->wishlist); context->wishlist = NULL;
        free(context->duelist); context->duelist = NULL;
        free(context->cu_duelist); context->cu_duelist = NULL;
        free(context->wishes); context->wishes = NULL;
        free(context->dues); context->dues = NULL;
        free(context->hput_pos); context->hput_pos = NULL;
        //free(context->lnEnts); context->lnEnts = NULL;
        //free(context->lfEnt); context->lfEnt = NULL;
        free(context->duepartners); context->duepartners = NULL;
        free(context->wishpartners); context->wishpartners = NULL;
        free(context->entsInCol); context->entsInCol = NULL;
    
        if (context->col_map->loc_perm != context->row_map->loc_perm) {
            ghost_cu_free(context->col_map->cu_loc_perm); context->col_map->cu_loc_perm = NULL;
            free(context->col_map->loc_perm); context->col_map->loc_perm = NULL;
            free(context->col_map->loc_perm_inv); context->col_map->loc_perm_inv = NULL;
        }
        ghost_cu_free(context->row_map->cu_loc_perm); context->row_map->cu_loc_perm = NULL;
        free(context->row_map->loc_perm); context->row_map->loc_perm = NULL;
        free(context->row_map->loc_perm_inv); context->row_map->loc_perm_inv = NULL;
        
        if (context->col_map->glb_perm != context->row_map->glb_perm) {
            free(context->col_map->glb_perm); context->col_map->glb_perm = NULL;
            free(context->col_map->glb_perm_inv); context->col_map->glb_perm_inv = NULL;
        }
        free(context->row_map->glb_perm); context->row_map->glb_perm = NULL;
        free(context->row_map->glb_perm_inv); context->row_map->glb_perm_inv = NULL;
        
        ghost_map_destroy(context->row_map);
        ghost_map_destroy(context->col_map);
        context->row_map = NULL;
        context->col_map = NULL;

        if(context->avg_ptr){
          free(context->avg_ptr); context->avg_ptr = NULL;
          free(context->mapAvg); context->mapAvg = NULL;
          free(context->mappedDuelist); context->mappedDuelist = NULL;
          free(context->nrankspresent); context->nrankspresent = NULL;
        }
    
        free(context->color_ptr);
        free(context->zone_ptr);
/*
        if( context->perm_local )
        {
          free(context->perm_local->perm); context->perm_local->perm = NULL;
          free(context->perm_local->invPerm); context->perm_local->invPerm = NULL;

          if(context->perm_local->method == GHOST_PERMUTATION_UNSYMMETRIC) {
          	free(context->perm_local->colPerm);
                context->perm_local->colPerm = NULL;
		free(context->perm_local->colInvPerm);
		context->perm_local->colInvPerm = NULL;
	}
        
	  free(context->perm_local); context->perm_local = NULL;
        }
      */
    }

    free(context);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TEARDOWN);
}

ghost_error ghost_context_comm_init(ghost_context *ctx, ghost_gidx *col_orig, ghost_sparsemat *mat, ghost_lidx *col, ghost_lidx *nhalo)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_SETUP);

    ghost_error ret = GHOST_SUCCESS;
    ghost_gidx j;
    ghost_gidx i;
    ghost_lidx max_loc_elements;
    ghost_lidx *present_values = NULL;
    ghost_lidx acc_dues = 0;
    ghost_lidx acc_wishes;

    ghost_lidx *item_from = NULL;

    ghost_lidx *wishlist_counts = NULL;

    ghost_lidx **wishlist = NULL;
    ghost_lidx **cwishlist = NULL;

    ghost_lidx this_pseudo_col;
    ghost_lidx *pseudocol = NULL;
    ghost_gidx *globcol = NULL;
    ghost_lidx *myrevcol = NULL;

    ghost_lidx *comm_remotePE = NULL;
    ghost_lidx *comm_remoteEl = NULL;
    ghost_lidx *wishl_mem  = NULL;
    ghost_lidx *duel_mem   = NULL;
    ghost_lidx acc_transfer_wishes, acc_transfer_dues;

    size_t size_nint, size_lcol, size_gcol;
    size_t size_nptr, size_pval;  
    size_t size_wish, size_dues;
    
    int nprocs;
    int me;
    
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, ctx->mpicomm));
    GHOST_CALL_RETURN(ghost_rank(&me, ctx->mpicomm));

#ifdef GHOST_HAVE_MPI
    MPI_Request req[2*nprocs];
    MPI_Status stat[2*nprocs];
#endif

    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->entsInCol,ctx->col_map->dim*sizeof(ghost_lidx)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->wishlist,nprocs*sizeof(ghost_lidx *)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->duelist,nprocs*sizeof(ghost_lidx *)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->wishes,nprocs*sizeof(ghost_lidx)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->dues,nprocs*sizeof(ghost_lidx)),err,ret);

    memset(ctx->entsInCol,0,ctx->col_map->dim*sizeof(ghost_lidx));
       
    ghost_lidx chunk,rowinchunk,entinrow,globalent,globalrow;
    for(chunk = 0; chunk < mat->nchunks; chunk++) {
        for (rowinchunk=0; rowinchunk<mat->traits.C; rowinchunk++) {
            globalrow = chunk*mat->traits.C+rowinchunk;
            if (globalrow < ctx->row_map->dim) { // avoid chunk padding rows
                for (entinrow=0; entinrow<mat->rowLen[globalrow]; entinrow++) {
                    globalent = mat->chunkStart[chunk] + entinrow*mat->traits.C + rowinchunk;
		            if (col_orig[globalent] >= ctx->col_map->offs && col_orig[globalent]<(ctx->col_map->offs+ctx->col_map->dim)) {
                        ctx->entsInCol[col_orig[globalent]-ctx->col_map->offs]++;
                    }
                }
            }
        }
    }

    ghost_type type;
    ghost_type_get(&type);
#ifdef GHOST_HAVE_CUDA
    if (type == GHOST_TYPE_CUDA) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->cu_duelist,nprocs*sizeof(ghost_lidx *)),err,ret);
    }
#endif

    for (i=0; i<nprocs; i++){
        ctx->wishes[i] = 0;
        ctx->dues[i] = 0;
        ctx->wishlist[i] = NULL;
        ctx->duelist[i] = NULL;
#ifdef GHOST_HAVE_CUDA
        if (type == GHOST_TYPE_CUDA) {
              ctx->cu_duelist[i] = NULL;
        }
#endif
    }

    size_nint = (size_t)( (size_t)(nprocs)   * sizeof(ghost_lidx)  );
    size_nptr = (size_t)( nprocs             * sizeof(ghost_lidx*) );

#ifdef GHOST_HAVE_MPI
    MPI_CALL_GOTO(MPI_Allreduce(&mat->nEnts,&max_loc_elements,1,ghost_mpi_dt_lidx,MPI_MAX,ctx->mpicomm),err,ret);
#else
    max_loc_elements = mat->nEnts;
#endif

    /*
    max_loc_elements = 0;
    for (i=0;i<nprocs;i++) {
        if (max_loc_elements<ctx->lnEnts[i]) {
            max_loc_elements = ctx->lnEnts[i];
        }
    }
    */

    size_pval = (size_t)( max_loc_elements * sizeof(ghost_lidx) );
    size_lcol  = (size_t)( (size_t)(mat->nEnts)   * sizeof( ghost_lidx ) );
    size_gcol  = (size_t)( (size_t)(mat->nEnts)   * sizeof( ghost_gidx ) );

    /*       / 1  2  .  3  4  . \
     *       | .  5  6  7  .  . |
     * mat = | 8  9  .  .  . 10 |
     *       | . 11 12 13  .  . |
     *       | .  .  .  . 14 15 |
     *       \16  .  .  . 17 18 /
     *
     * nprocs = 3
     * max_loc_elements = 4
     * item_from       = <{0,0,0},{0,0,0},{0,0,0}>
     * wishlist_counts = <{0,0,0},{0,0,0},{0,0,0}>
     * comm_remotePE   = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> PE where element is on
     * comm_remoteEl   = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> local colidx of element
     * present_values  = <{0,0,0,0,0,0,0},{0,0,0,0,0,0,0},{0,0,0,0,0,0,0}> 
     */

      
    GHOST_CALL_GOTO(ghost_malloc((void **)&item_from, size_nint),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&wishlist_counts, nprocs*sizeof(ghost_lidx)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm_remotePE, size_lcol),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm_remoteEl, size_lcol),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&present_values, size_pval),err,ret); 

    for (i=0; i<nprocs; i++) wishlist_counts[i] = 0;

    int nthreads;
    unsigned clsize;
#pragma omp parallel
    {
#pragma omp single
    nthreads = ghost_omp_nthread();
    }
    
    ghost_machine_cacheline_size(&clsize);
    int padding = 8*(int)clsize/sizeof(ghost_lidx);
        
    ghost_lidx *partial_wishlist_counts;
    GHOST_CALL_GOTO(ghost_malloc((void **)&partial_wishlist_counts, nthreads*(nprocs+padding)*sizeof(ghost_lidx)),err,ret); 
    memset(partial_wishlist_counts,0,nthreads*(nprocs+padding)*sizeof(ghost_lidx));

    
    GHOST_INSTR_START("comm_remote*");
#pragma omp parallel shared (partial_wishlist_counts)
    {
        int thread = ghost_omp_threadnum();

#pragma omp for private(j)
        for (i=0;i<mat->nEnts;i++){
            for (j=nprocs-1;j>=0; j--){
                if (ctx->row_map->goffs[j]<col_orig[i]+1) {//is col_orig unpermuted(probably)
                    comm_remotePE[i] = j;//comm_remotePE[colPerm[i]]=j
                    comm_remoteEl[i] = col_orig[i] -ctx->row_map->goffs[j]; //comm_remoteEl[colPerm[i]] = col_orig[i] -ctx->row_map->goffs[j];
                    partial_wishlist_counts[(padding+nprocs)*thread+j]++;
                    break;
                }
            }
        }
    }

    for (j=0; j<nprocs; j++) {
        for (i=0; i<nthreads; i++) {
            wishlist_counts[j] += partial_wishlist_counts[(padding+nprocs)*i+j];
        }
    }
    free(partial_wishlist_counts);
    GHOST_INSTR_STOP("comm_remote*");
    /*
     * wishlist_counts = <{3,3,1},{3,2,1},{1,0,4}>
     * comm_remotePE   = <{0,0,1,2,0,1,1},{0,0,2,0,1,1},{2,2,0,2,2}>
     * comm_remoteEl   = <{0,1,1,0,1,0,1},{0,1,1,1,0,1},{0,1,0,0,1}>
     */

    acc_wishes = 0;
    for (i=0; i<nprocs; i++) {
        acc_wishes += wishlist_counts[i];
    }

    /*
     * acc_wishes = <7,6,5> equal to lnEnts
     */

    GHOST_CALL_GOTO(ghost_malloc((void **)&wishlist,size_nptr),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&cwishlist,size_nptr),err,ret);

    /*
     * wishlist  = <{NULL,NULL,NULL},{NULL,NULL,NULL},{NULL,NULL,NULL}>
     * cwishlist = <{NULL,NULL,NULL},{NULL,NULL,NULL},{NULL,NULL,NULL}>
     */

    for (i=0; i<nprocs; i++){
        GHOST_CALL_GOTO(ghost_malloc((void **)&cwishlist[i],wishlist_counts[i]*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&wishlist[i],wishlist_counts[i]*sizeof(ghost_lidx)),err,ret);
    }
    /*
     * wishlist  = <{{0,0,0},{0,0,0},{0}},{{0,0,0},{0,0},{0}},{{0},NULL,{0,0,0,0}}>
     * cwishlist = <{{0,0,0},{0,0,0},{0}},{{0,0,0},{0,0},{0}},{{0},NULL,{0,0,0,0}}>
     */

    for (i=0;i<nprocs;i++) item_from[i] = 0;

    GHOST_INSTR_START("wishlist");
    for (i=0;i<mat->nEnts;i++){
        wishlist[comm_remotePE[i]][item_from[comm_remotePE[i]]] = comm_remoteEl[i];
        item_from[comm_remotePE[i]]++;
    }
    GHOST_INSTR_STOP("wishlist");
    /*
     * wishlist  = <{{0,1,1},{1,0,1},{0}},{{0,1,1},{0,1},{1}},{{0},NULL,{0,1,0,1}}> local column idx of wishes
     * item_from = <{3,3,1},{3,2,1},{1,0,4}> equal to wishlist_counts
     */

#ifdef GHOST_HAVE_MPI
    MPI_Barrier(ctx->mpicomm);

    GHOST_INSTR_START("wishes_and_dues");
    MPI_Win due_win,nduepartners_win;
    MPI_CALL_GOTO(MPI_Win_create(ctx->dues,nprocs*sizeof(ghost_lidx),sizeof(ghost_lidx),MPI_INFO_NULL,ctx->mpicomm,&due_win),err,ret);
    MPI_CALL_GOTO(MPI_Win_create(&ctx->nduepartners,sizeof(int),sizeof(int),MPI_INFO_NULL,ctx->mpicomm,&nduepartners_win),err,ret);

    ghost_lidx thisentry = 0;
    int one = 1;
    for (i=0; i<nprocs; i++) {

        if ( (i!=me) && (wishlist_counts[i]>0) ){
#pragma omp parallel for
            for (j=0; j<max_loc_elements; j++) {
                present_values[j] = -1;
            }
            thisentry = 0;
            for (j=0; j<wishlist_counts[i]; j++){
                if (present_values[wishlist[i][j]]<0){
                    present_values[wishlist[i][j]] = thisentry;
                    cwishlist[i][thisentry] = wishlist[i][j];
                    thisentry = thisentry + 1;
                }
            }
            ctx->wishes[i] = thisentry;
            ctx->nwishpartners++;

            MPI_CALL_GOTO(MPI_Win_lock(MPI_LOCK_SHARED,i,0,due_win),err,ret);            
            MPI_CALL_GOTO(MPI_Put(&ctx->wishes[i],1,ghost_mpi_dt_lidx,i,me,1,ghost_mpi_dt_lidx,due_win),err,ret);
            MPI_CALL_GOTO(MPI_Win_unlock(i,due_win),err,ret);            
            
            MPI_CALL_GOTO(MPI_Win_lock(MPI_LOCK_SHARED,i,0,nduepartners_win),err,ret);            
            MPI_CALL_GOTO(MPI_Accumulate(&one,1,MPI_INT,i,0,1,MPI_INT,MPI_SUM,nduepartners_win),err,ret);
            MPI_CALL_GOTO(MPI_Win_unlock(i,nduepartners_win),err,ret);            
        } else {
            ctx->wishes[i] = 0; 
        }

    }

    MPI_Win_free(&due_win);
    MPI_Win_free(&nduepartners_win);
#endif

    /* 
     * cwishlist = <{{#,#,#},{1,0,#},{0}},{{0,1,#},{#,#},{1}},{{0},NULL,{#,#,#,#}}> compressed wish list
     * ctx->wishes = <{0,2,1},{2,0,1},{1,0,0}>
     * ctx->dues = <{0,2,1},{2,0,0},{1,1,0}>
     */

    GHOST_INSTR_STOP("wishes_and_dues");


    // now, we now have many due/wish partners we have and can allocate the according arrays
    // it will be filled in a later loop over nprocs 
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->duepartners,sizeof(int)*ctx->nduepartners),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->wishpartners,sizeof(int)*ctx->nwishpartners),err,ret);

    acc_transfer_dues = 0;
    acc_transfer_wishes = 0;
    for (i=0; i<nprocs; i++){
        acc_transfer_wishes += ctx->wishes[i];
        acc_transfer_dues   += ctx->dues[i];
    }
    
    /* 
     * acc_transfer_wishes = <3,3,1>
     * acc_transfer_dues = <3,2,2>
     */

    (*nhalo) = 0;
    ghost_lidx tt = 0;
    i = me;
    int meHandled = 0;

 
    ghost_lidx first_putpos = 0;
     
     if(mat->context->col_map->flags & GHOST_PERM_NO_DISTINCTION) {
        ghost_lidx halo_ctr = 0;
        //we need to know number of halo elements now
        for(int k=0;k<nprocs;++k) {
          if (k != me){ 
            for (j=0;j<ctx->wishes[k];j++){
                ++halo_ctr;
            }
          }
        }

//        ctx->col_map->dimpad = PAD(ctx->row_map->dim+halo_ctr,ghost_densemat_row_padding());
//        ctx->nrowspadded   =  PAD(ctx->row_map->ldim[me]+halo_ctr,rowpadding);
//        rowpaddingoffset   =  ctx->nrowspadded-ctx->row_map->ldim[me];
        first_putpos = PAD(mat->context->col_map->dim,ghost_densemat_row_padding())+halo_ctr;
    } else {
//	    ctx->nrowspadded = PAD(ctx->row_map->ldim[me],rowpadding);// this is set already
//        ctx->col_map->dimpad = PAD(ctx->row_map->dim,ghost_densemat_row_padding());
        first_putpos = PAD(mat->context->col_map->dim,ghost_densemat_row_padding());
    }

//	rowpaddingoffset = MAX(ctx->row_map->dimpad,ctx->col_map->dimpad)-ctx->row_map->dim;
    //first_putpos -= ctx->row_map->dim;

    GHOST_INSTR_START("compress_cols");
    /*
     * col[i] = <{0,1,3,4,1,2,3},{0,1,5,1,2,3},{4,5,0,4,5}>
     */

    GHOST_CALL_GOTO(ghost_malloc((void **)&pseudocol,size_lcol),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&globcol,size_gcol),err,ret);
    
    /*
     * pseudocol = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> PE where element is on
     * globcol   = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> local colidx of element
     */

    this_pseudo_col = ctx->row_map->ldim[me];
    
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, ctx->mpicomm));
    GHOST_CALL_RETURN(ghost_rank(&me, ctx->mpicomm));

    i = me;
    
    for (; i<nprocs; i++) { // iterate i=me,..,nprocs,0,..,me-1
        ghost_lidx t;
        if (meHandled && (i == me)) continue;

        if (i != me){ 
            for (j=0;j<ctx->wishes[i];j++){
                pseudocol[(*nhalo)] = this_pseudo_col; 
                globcol[(*nhalo)]   = ctx->row_map->goffs[i]+cwishlist[i][j]; 
                (*nhalo)++;
                this_pseudo_col++;
            }
            /*
             * pseudocol = <{0,1,2},{0,1,2},{0}> colidx of each halo element starting from 0
             * globcol   = <{2,3,4},{2,3,4},{2}> colidx of each halo element starting from lnrows[me]
             */

            // myrevcol maps the actual colidx to the new colidx
            DEBUG_LOG(2,"Allocating space for myrevcol");
            GHOST_CALL_GOTO(ghost_malloc((void **)&myrevcol,ctx->row_map->ldim[i]*sizeof(ghost_lidx)),err,ret);
            for (j=0;j<ctx->wishes[i];j++){
                myrevcol[globcol[tt]-ctx->row_map->goffs[i]] = tt;
                tt++;
            }
            /*
             * 1st iter: myrevcol = <{1,0},{0,1},{0,#}>
             * 2nd iter: myrevcol = <{2,#},{#,2},{#,#}>
             */

#pragma omp parallel for
            for (t=0; t<mat->nEnts; t++) {
                if (comm_remotePE[t] == i) { // local element for rank i
                    col[t] =  first_putpos-ctx->row_map->dim + pseudocol[myrevcol[col_orig[t]-ctx->row_map->goffs[i]]];
                    //printf("col = %d-%d+%d = %d\n",first_putpos,ctx->row_map->dim, pseudocol[myrevcol[col_orig[t]-ctx->row_map->goffs[i]]],col[t]);
                }
            }
            free(myrevcol); myrevcol = NULL;
        } else { // first i iteration goes here
#pragma omp parallel for
            for (t=0; t<mat->nEnts; t++) {
                if (comm_remotePE[t] == me) { // local element for myself
                    col[t] =  comm_remoteEl[t];
                }
            }
            /*
             * col[i] = <{0,1,3,4,1,2,3},{0,1,5,1,0,1},{0,1,0,0,1}> local idx changed after first iteration
             */

        }

        if (!meHandled) {
            i = -1;
            meHandled = 1;
        }
    }

    GHOST_INSTR_STOP("compress_cols")
    /*
     * col[i] = <{0,1,2,4,1,3,2},{2,3,4,3,0,1},{0,1,2,0,1}>
     */

    GHOST_INSTR_START("final")

    size_wish = (size_t)( acc_transfer_wishes * sizeof(ghost_lidx) );
    size_dues = (size_t)( acc_transfer_dues   * sizeof(ghost_lidx) );

    // we need a contiguous array in memory
    GHOST_CALL_GOTO(ghost_malloc((void **)&wishl_mem,size_wish),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&duel_mem,size_dues),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->hput_pos,size_nptr),err,ret); 
   
    ghost_type_get(&type); 
#ifdef GHOST_HAVE_CUDA
    ghost_lidx *cu_duel_mem;
    if (type == GHOST_TYPE_CUDA) {
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&cu_duel_mem,size_dues),err,ret);
    }
#endif

    acc_dues = 0;
    acc_wishes = 0;

    int duepartneridx = 0, wishpartneridx = 0;

    for (i=0; i<nprocs; i++){
        if (ctx->dues[i]) {
            ctx->duepartners[duepartneridx] = i;
            duepartneridx++;
        }
        if (ctx->wishes[i]) {
            ctx->wishpartners[wishpartneridx] = i;
            wishpartneridx++;
        }

        ctx->duelist[i]    = &(duel_mem[acc_dues]);
#ifdef GHOST_HAVE_CUDA
        if (type == GHOST_TYPE_CUDA) {
            ctx->cu_duelist[i]    = &(cu_duel_mem[acc_dues]);
        }
#endif
        ctx->wishlist[i]   = &(wishl_mem[acc_wishes]);
        ctx->hput_pos[i]   = first_putpos + acc_wishes;

        if  ( (me != i) && !( (i == nprocs-2) && (me == nprocs-1) ) ){
            acc_dues   += ctx->dues[i];
            acc_wishes += ctx->wishes[i];
        }
    }

#ifdef GHOST_HAVE_MPI
    for (i=0;i<2*nprocs;i++) 
        req[i] = MPI_REQUEST_NULL;

    for (i=0; i<nprocs; i++) 
        for (j=0;j<ctx->wishes[i];j++)
            ctx->wishlist[i][j] = cwishlist[i][j]; 

    int msgcount = 0;

    // TODO only loop duepartners
    for(i=0; i<nprocs; i++) 
    { // receive _my_ dues from _other_ processes' wishes
        MPI_CALL_GOTO(MPI_Irecv(ctx->duelist[i],ctx->dues[i],ghost_mpi_dt_lidx,i,i,ctx->mpicomm,&req[msgcount]),err,ret);
        msgcount++;
    }


    for(i=0; i<nprocs; i++) { 
        MPI_CALL_GOTO(MPI_Isend(ctx->wishlist[i],ctx->wishes[i],ghost_mpi_dt_lidx,i,me,ctx->mpicomm,&req[msgcount]),err,ret);
        msgcount++;
    }

    MPI_CALL_GOTO(MPI_Waitall(msgcount,req,stat),err,ret);
#endif

#ifdef GHOST_HAVE_CUDA
    if (type == GHOST_TYPE_CUDA) {
        GHOST_CALL_GOTO(ghost_cu_upload(cu_duel_mem,duel_mem,size_dues),err,ret);
    }
#endif


    GHOST_INSTR_STOP("final")
    goto out;

err:
    free(wishl_mem); wishl_mem = NULL;
    free(duel_mem); duel_mem = NULL;
    for (i=0; i<nprocs; i++) {
        free(ctx->wishlist[i]); ctx->wishlist[i] = NULL;
        free(ctx->duelist[i]); ctx->duelist[i] = NULL;
    }
    free(ctx->hput_pos); ctx->hput_pos = NULL;
    free(ctx->wishes); ctx->wishes = NULL;
    free(ctx->dues); ctx->dues = NULL;
    free(ctx->duepartners); ctx->duepartners = NULL;
    free(ctx->wishpartners); ctx->wishpartners = NULL;
    free(ctx->entsInCol); ctx->entsInCol = NULL;

out:
    for (i=0; i<nprocs; i++) {
        free(wishlist[i]); wishlist[i] = NULL;
    }
    free(wishlist); wishlist = NULL;
    for (i=0; i<nprocs; i++) {
        free(cwishlist[i]); cwishlist[i] = NULL;
    }
    free(cwishlist); cwishlist = NULL;
    free(wishlist_counts); wishlist_counts = NULL;
    free(item_from); item_from = NULL;
    free(comm_remotePE); comm_remotePE = NULL;
    free(comm_remoteEl); comm_remoteEl = NULL;
    free(present_values); present_values = NULL;
    free(pseudocol); pseudocol = NULL;
    free(globcol); globcol = NULL;
    free(myrevcol); myrevcol = NULL;
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_SETUP);
    return ret;
}

char * ghost_context_workdist_string(ghost_context_flags_t flags)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    char *ret;
    if (flags & GHOST_CONTEXT_DIST_NZ) {
        ret = "Equal no. of nonzeros";
    } else if(flags & GHOST_CONTEXT_DIST_ROWS) {
        ret = "Equal no. of rows";
    } else {
        ret = "Invalid";
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);

    return ret;
}

ghost_map *ghost_context_map(const ghost_context *ctx, const ghost_maptype mt) 
{
    return mt==GHOST_MAP_ROW?ctx->row_map:mt==GHOST_MAP_COL?ctx->col_map:ctx->row_map;
}

ghost_map *ghost_context_other_map(const ghost_context *ctx, const ghost_maptype mt)
{
    return mt==GHOST_MAP_ROW?ctx->col_map:mt==GHOST_MAP_COL?ctx->row_map:NULL;
}
    
ghost_map *ghost_context_max_map(const ghost_context *ctx)
{
    return ctx->row_map->dimpad>ctx->col_map->dimpad?ctx->row_map:ctx->col_map;
}

ghost_error ghost_context_set_map(ghost_context *ctx, ghost_maptype which, ghost_map *map)
{
    ghost_map *oldmap = ghost_context_map(ctx,which);

    if (oldmap) {
        oldmap->ref_count--;
    }
    if (which == GHOST_MAP_ROW) {
        ctx->row_map = map;
    } else if (which == GHOST_MAP_COL) {
        ctx->col_map = map;
    } else {
        ERROR_LOG("The map is to be either a column or row map!");
        return GHOST_ERR_INVALID_ARG;
    }

    map->ref_count++;

    return GHOST_SUCCESS;
}
    
ghost_error ghost_context_comm_string(char **str, ghost_context *ctx, int root)
{
#ifdef GHOST_HAVE_MPI
    int nrank, r, p, me,l;
    bool printline = false;
    ghost_lidx maxdues = 0, maxwishes = 0;
    int dueslen = 0, duesprintlen = 4, wisheslen = 0, wishesprintlen = 6, ranklen, rankprintlen, linelen;
    ghost_nrank(&nrank,ctx->mpicomm);
    ghost_rank(&me,ctx->mpicomm);

    ranklen = (int)floor(log10(abs(nrank))) + 1;
    
    for (r=0; r<nrank; r++) {
        maxdues = MAX(maxdues,ctx->dues[r]);
        maxwishes = MAX(maxwishes,ctx->wishes[r]);
    }

    MPI_Allreduce(MPI_IN_PLACE,&maxdues,1,ghost_mpi_dt_lidx,MPI_MAX,ctx->mpicomm);
    MPI_Allreduce(MPI_IN_PLACE,&maxwishes,1,ghost_mpi_dt_lidx,MPI_MAX,ctx->mpicomm);
    
    if (maxdues != 0) {
        dueslen = (int)floor(log10(abs(maxdues))) + 1;
    }
    
    if (maxwishes != 0) {
        wisheslen = (int)floor(log10(abs(maxwishes))) + 1;
    }

    rankprintlen = MAX(4,ranklen);
    duesprintlen = MAX(4,dueslen + 3 + ranklen);
    wishesprintlen = MAX(6,wisheslen + 3 + ranklen);
    
    linelen = rankprintlen + duesprintlen + wishesprintlen + 4;
    
    if (me == root) {
        ghost_malloc((void **)str,1 + (linelen+1)*(3 + (nrank*(nrank+1))));
        memset(*str,'\0',1 + (linelen+1)*(3 + (nrank*(nrank+1))));
        
        for (l=0; l<linelen; l++) {
            sprintf((*str)+strlen(*str),"=");
        }
        sprintf((*str)+strlen(*str),"\n");
        sprintf((*str)+strlen(*str),"%*s  %*s  %*s\n",rankprintlen,"RANK",duesprintlen,"DUES",wishesprintlen,"WISHES");
        for (l=0; l<linelen; l++) {
            sprintf((*str)+strlen(*str),"=");
        }
        sprintf((*str)+strlen(*str),"\n");
    }

    ghost_lidx dues[nrank];
    ghost_lidx wishes[nrank];

    for (r=0; r<nrank; r++) {
        if (r == root && me == root) {
            memcpy(wishes,ctx->wishes,nrank*sizeof(ghost_lidx));
            memcpy(dues,ctx->dues,nrank*sizeof(ghost_lidx));
        } else {
            if (me == root) {
                MPI_Recv(wishes,nrank,ghost_mpi_dt_lidx,r,r,ctx->mpicomm,MPI_STATUS_IGNORE);
                MPI_Recv(dues,nrank,ghost_mpi_dt_lidx,r,nrank+r,ctx->mpicomm,MPI_STATUS_IGNORE);
            }
            if (me == r) {
                MPI_Send(ctx->wishes,nrank,ghost_mpi_dt_lidx,root,me,ctx->mpicomm);
                MPI_Send(ctx->dues,nrank,ghost_mpi_dt_lidx,root,me+nrank,ctx->mpicomm);
            }
        }
        
        if (me == root) {
            for (p=0; p<nrank; p++) {
                if (wishes[p] && dues[p]) {
                    sprintf((*str)+strlen(*str),"%*d",rankprintlen,r);
                    sprintf((*str)+strlen(*str),"  =>%*d %*d",ranklen,p,dueslen,dues[p]);
                    sprintf((*str)+strlen(*str),"%*s<=%*d %*d\n",MAX(2,2+wishesprintlen-(wisheslen+3+ranklen))," ",ranklen,p,wisheslen,wishes[p]);
                } else if (wishes[p]) {
                    sprintf((*str)+strlen(*str),"%*d  %*s",rankprintlen,r,duesprintlen," ");
                    sprintf((*str)+strlen(*str),"%*s<=%*d %*d\n",MAX(2,2+wishesprintlen-(wisheslen+3+ranklen))," ",ranklen,p,wisheslen,wishes[p]);
                } else if (dues[p]) {
                    sprintf((*str)+strlen(*str),"%*d  =>%*d %*d\n",rankprintlen,r,ranklen,p,dueslen,dues[p]);
                }
                if (wishes[p] || dues[p]) {
                    printline = true;
                }

            }
            if (printline && r != nrank-1) {
                for (l=0; l<linelen; l++) {
                    sprintf((*str)+strlen(*str),"-");
                }
                sprintf((*str)+strlen(*str),"\n");
            }
        }
        printline = false;

        //    if (wishes[w])
//        if (me == root) {
            //printf("%.*d %.*d %.*d\n",ranklen,r,dueslen,dues[r],wisheslen,wishes[r]);
//            printf("%*d <-%*d %*d\n",rankprintlen,r,ranklen,r,wisheslen,wishes[r]);
//        }

    }

    if (me == root) {
        for (l=0; l<linelen; l++) {
            sprintf((*str)+strlen(*str),"=");
        }
        sprintf((*str)+strlen(*str),"\n");
    }
#else
    UNUSED(ctx);
    UNUSED(root);
    ghost_malloc((void **)str,8);
    strcpy(*str,"No MPI!");
#endif

    return GHOST_SUCCESS;




}
