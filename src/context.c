#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_util.h>
#include <ghost_context.h>
#include <ghost_constants.h>
#include <ghost_affinity.h>
#include <ghost_crs.h>


ghost_context_t *ghost_createContext(int64_t gnrows, int64_t gncols, int context_flags, char *matrixPath, MPI_Comm comm, double weight) 
{
    ghost_context_t *context;
    int i;

    context = (ghost_context_t *)ghost_malloc(sizeof(ghost_context_t));
    context->flags = context_flags;
    context->rowPerm = NULL;
    context->invRowPerm = NULL;
    context->mpicomm = comm;

    if ((gnrows == GHOST_GET_DIM_FROM_MATRIX) || (gncols == GHOST_GET_DIM_FROM_MATRIX)) {
        ghost_matfile_header_t fileheader;
        ghost_readMatFileHeader(matrixPath,&fileheader);
#ifndef LONGIDX
        if ((fileheader.nrows >= (int64_t)INT_MAX) || (fileheader.ncols >= (int64_t)INT_MAX)) {
            ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
        }
#endif
        if (gnrows == GHOST_GET_DIM_FROM_MATRIX)
            context->gnrows = (ghost_midx_t)fileheader.nrows;
        if (gncols == GHOST_GET_DIM_FROM_MATRIX)
            context->gncols = (ghost_midx_t)fileheader.ncols;

    } else {
#ifndef LONGIDX
        if ((gnrows >= (int64_t)INT_MAX) || (gncols >= (int64_t)INT_MAX)) {
            ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
        }
#endif
        context->gnrows = (ghost_midx_t)gnrows;
        context->gncols = (ghost_midx_t)gncols;
    }
    DEBUG_LOG(1,"Creating context with dimension %"PRmatIDX"x%"PRmatIDX,context->gnrows,context->gncols);

#ifdef GHOST_HAVE_MPI
    if (!(context->flags & GHOST_CONTEXT_DISTRIBUTED) && !(context->flags & GHOST_CONTEXT_GLOBAL)) {
        DEBUG_LOG(1,"Context is set to be distributed");
        context->flags |= GHOST_CONTEXT_DISTRIBUTED;
    }
#else
    if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
        ABORT("Creating a distributed matrix without MPI is not possible");
    } else if (!(context->flags & GHOST_CONTEXT_GLOBAL)) {
        DEBUG_LOG(1,"Context is set to be global");
        context->flags |= GHOST_CONTEXT_GLOBAL;
    }
#endif

    context->spmvsolvers = (ghost_spmvsolver_t *)ghost_malloc(sizeof(ghost_spmvsolver_t)*GHOST_NUM_MODES);
    for (i=0; i<GHOST_NUM_MODES; i++) context->spmvsolvers[i] = NULL;
#ifdef GHOST_HAVE_MPI
    context->spmvsolvers[GHOST_SPMVM_MODE_VECTORMODE_IDX] = &hybrid_kernel_I;
    context->spmvsolvers[GHOST_SPMVM_MODE_GOODFAITH_IDX] = &hybrid_kernel_II;
    context->spmvsolvers[GHOST_SPMVM_MODE_TASKMODE_IDX] = &hybrid_kernel_III;
#else
    context->spmvsolvers[GHOST_SPMVM_MODE_NOMPI_IDX] = &ghost_solver_nompi;
#endif

#ifdef GHOST_HAVE_MPI
    if (context->flags & GHOST_CONTEXT_DISTRIBUTED) {
        context->communicator = (ghost_comm_t*) ghost_malloc( sizeof(ghost_comm_t));
        context->communicator->halo_elements = -1;

        int nprocs = ghost_getNumberOfRanks(context->mpicomm);

        context->communicator->lnEnts   = (ghost_mnnz_t*)       ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
        context->communicator->lfEnt    = (ghost_mnnz_t*)       ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
        context->communicator->lnrows   = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 
        context->communicator->lfRow    = (ghost_midx_t*)       ghost_malloc( nprocs*sizeof(ghost_midx_t)); 

        if (context->flags & GHOST_CONTEXT_WORKDIST_NZE)
        { // read rpt and fill lfrow, lnrows, lfent, lnents
            ghost_midx_t *rpt = NULL;
            ghost_mnnz_t gnnz;

            if (ghost_getRank(context->mpicomm) == 0) {
                rpt = CRS_readRpt(context->gnrows+1,matrixPath);
                context->rpt = rpt;

                gnnz = rpt[context->gnrows];
                ghost_mnnz_t target_nnz;
                target_nnz = (gnnz/nprocs)+1; /* sonst bleiben welche uebrig! */

                context->communicator->lfRow[0]  = 0;
                context->communicator->lfEnt[0] = 0;
                int j = 1;

                for (i=0;i<context->gnrows;i++){
                    if (rpt[i] >= j*target_nnz){
                        context->communicator->lfRow[j] = i;
                        context->communicator->lfEnt[j] = rpt[i];
                        j = j+1;
                    }
                }
                for (i=0; i<nprocs-1; i++){
                    context->communicator->lnrows[i] = context->communicator->lfRow[i+1] - context->communicator->lfRow[i] ;
                    context->communicator->lnEnts[i] = context->communicator->lfEnt[i+1] - context->communicator->lfEnt[i] ;
                }

                context->communicator->lnrows[nprocs-1] = context->gnrows - context->communicator->lfRow[nprocs-1] ;
                context->communicator->lnEnts[nprocs-1] = gnnz - context->communicator->lfEnt[nprocs-1];

                //fclose(filed);
            }
            MPI_safecall(MPI_Bcast(context->communicator->lfRow,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
            MPI_safecall(MPI_Bcast(context->communicator->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
            MPI_safecall(MPI_Bcast(context->communicator->lnrows, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
            MPI_safecall(MPI_Bcast(context->communicator->lnEnts, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));


        } else
        { // don't read rpt, only fill lfrow, lnrows
            UNUSED(matrixPath);
            int me = ghost_getRank(context->mpicomm);
            double allweights;
            MPI_safecall(MPI_Allreduce(&weight,&allweights,1,MPI_DOUBLE,MPI_SUM,context->mpicomm))

                ghost_midx_t my_target_rows = (ghost_midx_t)(context->gnrows*((double)weight/(double)allweights));
            ghost_midx_t target_rows[nprocs];

            MPI_safecall(MPI_Allgather(&my_target_rows,1,ghost_mpi_dt_midx,&target_rows[me],1,ghost_mpi_dt_midx,context->mpicomm));

            context->rpt = NULL;
            context->communicator->lfRow[0] = 0;

            for (i=1; i<nprocs; i++){
                context->communicator->lfRow[i] = context->communicator->lfRow[i-1]+target_rows[i-1];
            }
            for (i=0; i<nprocs-1; i++){
                context->communicator->lnrows[i] = context->communicator->lfRow[i+1] - context->communicator->lfRow[i] ;
            }
            context->communicator->lnrows[nprocs-1] = context->gnrows - context->communicator->lfRow[nprocs-1] ;
            MPI_safecall(MPI_Bcast(context->communicator->lfRow,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
            MPI_safecall(MPI_Bcast(context->communicator->lnrows, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
        }

    } else {
        context->communicator = NULL;
    }

#else
    UNUSED(weight);
    context->communicator = NULL;
#endif

    DEBUG_LOG(1,"Context created successfully");
    return context;
}


void ghost_freeContext(ghost_context_t *context)
{
    DEBUG_LOG(1,"Freeing context");
    if (context != NULL) {
        free(context->spmvsolvers);
        free(context->rowPerm);
        free(context->invRowPerm);
        ghost_freeCommunicator(context->communicator);

        free(context);
    }
    DEBUG_LOG(1,"Context freed successfully");
}

int ghost_setupCommunication(ghost_context_t *ctx, ghost_midx_t *col)
{

    ghost_comm_t *comm = ctx->communicator;
    int hlpi;
    int j;
    int i;
    int me;
    ghost_mnnz_t max_loc_elements, thisentry;
    int *present_values;
    int acc_dues;
    int *tmp_transfers;
    int acc_wishes;

    /* Counter how many entries are requested from each PE */
    int *item_from;

    ghost_mnnz_t *wishlist_counts;

    int **wishlist;
    int **cwishlist;


    int this_pseudo_col;
    int *pseudocol;
    int *globcol;

    int *comm_remotePE, *comm_remoteEl;
    ghost_mnnz_t lnEnts_l, lnEnts_r;
    int current_l, current_r;
    int acc_transfer_wishes, acc_transfer_dues;

    size_t size_nint, size_col;
    size_t size_a2ai, size_nptr, size_pval;  
    size_t size_wish, size_dues;

    int nprocs = ghost_getNumberOfRanks(ctx->mpicomm);

    size_nint = (size_t)( (size_t)(nprocs)   * sizeof(ghost_midx_t)  );
    size_nptr = (size_t)( nprocs             * sizeof(ghost_midx_t*) );
    size_a2ai = (size_t)( nprocs*nprocs * sizeof(ghost_midx_t)  );

    me = ghost_getRank(ctx->mpicomm);

    max_loc_elements = 0;
    for (i=0;i<nprocs;i++)
        if (max_loc_elements<comm->lnEnts[i]) max_loc_elements = comm->lnEnts[i];


    size_pval = (size_t)( max_loc_elements * sizeof(int) );
    size_col  = (size_t)( (size_t)(comm->lnEnts[me])   * sizeof( int ) );

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
     * tmp_transfers   = <{0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0}>
     */

      

    item_from       = (int*) ghost_malloc( size_nint); 
    wishlist_counts = (ghost_mnnz_t *) ghost_malloc( nprocs*sizeof(ghost_mnnz_t)); 
    comm_remotePE   = (int*) ghost_malloc(size_col);
    comm_remoteEl   = (int*) ghost_malloc(size_col);
    present_values  = (int*) ghost_malloc(size_pval); 
    tmp_transfers   = (int*) ghost_malloc(size_a2ai); 

    for (i=0; i<nprocs; i++) wishlist_counts[i] = 0;

    
    for (i=0;i<comm->lnEnts[me];i++){
        for (j=nprocs-1;j>=0; j--){
            if (comm->lfRow[j]<col[i]+1) {
                comm_remotePE[i] = j;
                wishlist_counts[j]++;
                comm_remoteEl[i] = col[i] -comm->lfRow[j];
                break;
            }
        }
    }
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

    wishlist        = (int**) ghost_malloc(size_nptr); 
    cwishlist       = (int**) ghost_malloc(size_nptr);

    /*
     * wishlist  = <{NULL,NULL,NULL},{NULL,NULL,NULL},{NULL,NULL,NULL}>
     * cwishlist = <{NULL,NULL,NULL},{NULL,NULL,NULL},{NULL,NULL,NULL}>
     */

    hlpi = 0;
    for (i=0; i<nprocs; i++){
        cwishlist[i] = (int *)ghost_malloc(wishlist_counts[i]*sizeof(int));
        wishlist[i] = (int *)ghost_malloc(wishlist_counts[i]*sizeof(int));
        hlpi += wishlist_counts[i];
    }
    /*
     * wishlist  = <{{0,0,0},{0,0,0},{0}},{{0,0,0},{0,0},{0}},{{0},NULL,{0,0,0,0}}>
     * cwishlist = <{{0,0,0},{0,0,0},{0}},{{0,0,0},{0,0},{0}},{{0},NULL,{0,0,0,0}}>
     */

    for (i=0;i<nprocs;i++) item_from[i] = 0;

    for (i=0;i<comm->lnEnts[me];i++){
        wishlist[comm_remotePE[i]][item_from[comm_remotePE[i]]] = comm_remoteEl[i];
        item_from[comm_remotePE[i]]++;
    }
    /*
     * wishlist  = <{{0,1,1},{1,0,1},{0}},{{0,1,1},{0,1},{1}},{{0},NULL,{0,1,0,1}}> local column idx of wishes
     * item_from = <{3,3,1},{3,2,1},{1,0,4}> equal to wishlist_counts
     */



    for (i=0; i<nprocs; i++) {
        for (j=0; j<max_loc_elements; j++) 
            present_values[j] = -1;

        if ( (i!=me) && (wishlist_counts[i]>0) ){
            thisentry = 0;
            for (j=0; j<wishlist_counts[i]; j++){
                if (present_values[wishlist[i][j]]<0){
                    present_values[wishlist[i][j]] = thisentry;
                    cwishlist[i][thisentry] = wishlist[i][j];
                    thisentry = thisentry + 1;
                }
            }
            comm->wishes[i] = thisentry;
        } else {
            comm->wishes[i] = 0; 
        }

    }

    /* 
     * cwishlist = <{{#,#,#},{1,0,#},{0}},{{0,1,#},{#,#},{1}},{{0},NULL,{#,#,#,#}}> compressed wish list
     * comm->wishes = <{0,2,1},{2,0,1},{1,0,0}>
     */

    for (i=0; i<nprocs; i++){
        free(wishlist[i]);
    }
    free(wishlist);

    MPI_safecall(MPI_Allgather ( comm->wishes, nprocs, MPI_INTEGER, tmp_transfers, 
                nprocs, MPI_INTEGER, ctx->mpicomm )) ;

    for (i=0; i<nprocs; i++) {
        comm->dues[i] = tmp_transfers[i*nprocs+me];
    }

    comm->dues[me] = 0; 
    
    /* 
     * comm->dues = <{0,2,1},{2,0,0},{1,1,0}>
     */

    acc_transfer_dues = 0;
    acc_transfer_wishes = 0;
    for (i=0; i<nprocs; i++){
        acc_transfer_wishes += comm->wishes[i];
        acc_transfer_dues   += comm->dues[i];
    }
    
    /* 
     * acc_transfer_wishes = <3,3,1>
     * acc_transfer_dues = <3,2,2>
     */

    pseudocol       = (int*) ghost_malloc(size_col);
    globcol         = (int*) ghost_malloc(size_col);
    
    /*
     * pseudocol = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> PE where element is on
     * globcol   = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> local colidx of element
     */

    this_pseudo_col = comm->lnrows[me];
    comm->halo_elements = 0;
    int tt = 0;
    i = me;
    int meHandled = 0;

    /*
     * col[i] = <{0,1,3,4,1,2,3},{0,1,5,1,2,3},{4,5,0,4,5}>
     */

    for (; i<nprocs; i++) { // iterate i=me,..,nprocs,0,..,me-1
        int t = 0;
        if (meHandled && (i == me)) continue;

        if (i != me){ 
            for (j=0;j<comm->wishes[i];j++){
                pseudocol[comm->halo_elements] = this_pseudo_col; 
                globcol[comm->halo_elements]   = comm->lfRow[i]+cwishlist[i][j]; 
                comm->halo_elements++;
                this_pseudo_col++;
            }
            /*
             * pseudocol = <{0,1,2},{0,1,2},{0}> colidx of each halo element starting from 0
             * globcol   = <{2,3,4},{2,3,4},{2}> colidx of each halo element starting from lnrows[me]
             */

            // myrevcol maps the actual colidx to the new colidx
            DEBUG_LOG(2,"Allocating space for myrevcol");
            int * myrevcol = (int *)ghost_malloc(comm->lnrows[i]*sizeof(int));
            for (j=0;j<comm->wishes[i];j++){
                myrevcol[globcol[tt]-comm->lfRow[i]] = tt;
                tt++;
            }
            /*
             * 1st iter: myrevcol = <{1,0},{0,1},{0,#}>
             * 2nd iter: myrevcol = <{2,#},{#,2},{#,#}>
             */

            for (;t<comm->lnEnts[me];t++) {
                if (comm_remotePE[t] == i) { // local element for rank i
                    col[t] =  pseudocol[myrevcol[col[t]-comm->lfRow[i]]];
                }
            }
            free(myrevcol);
        } else { // first i iteration goes here
            for (;t<comm->lnEnts[me];t++) {
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
    /*
     * col[i] = <{0,1,2,4,1,3,2},{2,3,4,3,0,1},{0,1,2,0,1}>
     */

    free(comm_remoteEl);
    free(comm_remotePE);
    free(pseudocol);
    free(globcol);
    free(present_values); 

    size_wish = (size_t)( acc_transfer_wishes * sizeof(int) );
    size_dues = (size_t)( acc_transfer_dues   * sizeof(int) );

    MPI_safecall(MPI_Barrier(ctx->mpicomm));

    comm->wishlist      = (int**) ghost_malloc(nprocs*sizeof(int *)); 
    comm->duelist       = (int**) ghost_malloc(nprocs*sizeof(int *)); 
    int *wishl_mem  = (int *)  ghost_malloc(size_wish); // we need a contiguous array in memory
    int *duel_mem   = (int *)  ghost_malloc(size_dues); // we need a contiguous array in memory
    comm->hput_pos      = (ghost_midx_t*)  ghost_malloc(size_nptr); 

    acc_dues = 0;
    acc_wishes = 0;


    for (i=0; i<nprocs; i++){

        comm->duelist[i]    = &(duel_mem[acc_dues]);
        comm->wishlist[i]   = &(wishl_mem[acc_wishes]);
        comm->hput_pos[i]   = comm->lnrows[me]+acc_wishes;

        if  ( (me != i) && !( (i == nprocs-2) && (me == nprocs-1) ) ){
            acc_dues   += comm->dues[i];
            acc_wishes += comm->wishes[i];
        }
    }

    MPI_Request req[2*nprocs];
    MPI_Status stat[2*nprocs];
    for (i=0;i<2*nprocs;i++) 
        req[i] = MPI_REQUEST_NULL;

    for (i=0; i<nprocs; i++) 
        for (j=0;j<comm->wishes[i];j++)
            comm->wishlist[i][j] = cwishlist[i][j]; 

    int msgcount = 0;
    for(i=0; i<nprocs; i++) 
    { // receive _my_ dues from _other_ processes' wishes
        MPI_safecall(MPI_Irecv(comm->duelist[i],comm->dues[i],MPI_INTEGER,i,i,ctx->mpicomm,&req[msgcount]));
        msgcount++;
    }


    for(i=0; i<nprocs; i++) { 
        MPI_safecall(MPI_Isend(comm->wishlist[i],comm->wishes[i],MPI_INTEGER,i,me,ctx->mpicomm,&req[msgcount]));
        msgcount++;
    }

    MPI_safecall(MPI_Waitall(msgcount,req,stat));

    for (i=0; i<nprocs; i++)
        free(cwishlist[i]);
    free(cwishlist);
    free(tmp_transfers);
    free(wishlist_counts);
    free(item_from);
    return GHOST_SUCCESS;
}
