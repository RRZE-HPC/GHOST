#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/context.h"
#include "ghost/constants.h"
#include "ghost/locality.h"
#include "ghost/crs.h"
#include "ghost/io.h"
#include "ghost/log.h"


ghost_error_t ghost_createContext(ghost_context_t **context, ghost_midx_t gnrows, ghost_midx_t gncols, ghost_context_flags_t context_flags, void *matrixSource, ghost_mpi_comm_t comm, double weight) 
{
    if (weight < 0) {
        ERROR_LOG("Negative weight");
        return GHOST_ERR_INVALID_ARG;
    }
           
    int nranks, me, i;
    ghost_error_t ret = GHOST_SUCCESS;
    
    ghost_midx_t *target_rows = NULL;
    char *tmpval = NULL;
    ghost_midx_t *tmpcol = NULL;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)context,sizeof(ghost_context_t)),err,ret);
    (*context)->flags = context_flags;
    (*context)->rowPerm = NULL;
    (*context)->invRowPerm = NULL;
    (*context)->mpicomm = comm;
    (*context)->wishes   = NULL;
    (*context)->dues     = NULL;
    (*context)->hput_pos = NULL;

    
    GHOST_CALL_GOTO(ghost_getNumberOfRanks((*context)->mpicomm,&nranks),err,ret);
    GHOST_CALL_GOTO(ghost_getRank((*context)->mpicomm,&me),err,ret);

    
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->wishlist,nranks*sizeof(ghost_midx_t *)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->duelist,nranks*sizeof(ghost_midx_t *)),err,ret);
    for (i=0; i<nranks; i++){
        (*context)->wishlist[i] = NULL;
        (*context)->duelist[i] = NULL;
    }

    if (!((*context)->flags & GHOST_CONTEXT_DIST_NZ)) {
        (*context)->flags |= GHOST_CONTEXT_DIST_ROWS;
    }

    if ((gnrows < 1) || (gncols < 1)) {
        ghost_matfile_header_t fileheader;
        GHOST_CALL_GOTO(ghost_readMatFileHeader((char *)matrixSource,&fileheader),err,ret);
#ifndef GHOST_HAVE_LONGIDX
        if ((fileheader.nrows >= (int64_t)INT_MAX) || (fileheader.ncols >= (int64_t)INT_MAX)) {
            ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with LONGIDX enabled!");
            return GHOST_ERR_IO;
        }
#endif
        if (gnrows < 1)
            (*context)->gnrows = (ghost_midx_t)fileheader.nrows;
        if (gncols < 1)
            (*context)->gncols = (ghost_midx_t)fileheader.ncols;

    } else {
#ifndef GHOST_HAVE_LONGIDX
        if ((gnrows >= (int64_t)INT_MAX) || (gncols >= (int64_t)INT_MAX)) {
            ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with LONGIDX enabled!");
            return GHOST_ERR_IO;
        }
#endif
        (*context)->gnrows = (ghost_midx_t)gnrows;
        (*context)->gncols = (ghost_midx_t)gncols;
    }
    DEBUG_LOG(1,"Creating context with dimension %"PRmatIDX"x%"PRmatIDX,(*context)->gnrows,(*context)->gncols);

#ifdef GHOST_HAVE_MPI
    if (!((*context)->flags & (GHOST_CONTEXT_DISTRIBUTED|GHOST_CONTEXT_REDUNDANT))) { // neither flag is set
        DEBUG_LOG(1,"Context is set to be distributed");
        (*context)->flags |= GHOST_CONTEXT_DISTRIBUTED;
    }
#else
    if ((*context)->flags & GHOST_CONTEXT_DISTRIBUTED) {
        WARNING_LOG("Creating a distributed matrix without MPI is not possible. Forcing redundant.");
        (*context)->flags &= ~GHOST_CONTEXT_DISTRIBUTED;
        (*context)->flags |= GHOST_CONTEXT_REDUNDANT;
    } else if (!((*context)->flags & GHOST_CONTEXT_REDUNDANT)) {
        DEBUG_LOG(1,"Context is set to be redundant");
        (*context)->flags |= GHOST_CONTEXT_REDUNDANT;
    }
#endif

    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->lnEnts, nranks*sizeof(ghost_mnnz_t)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->lfEnt, nranks*sizeof(ghost_mnnz_t)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->lnrows, nranks*sizeof(ghost_midx_t)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->lfRow, nranks*sizeof(ghost_midx_t)),err,ret);

#ifdef GHOST_HAVE_MPI
    ghost_midx_t row;
    if ((*context)->flags & GHOST_CONTEXT_DISTRIBUTED) {
        (*context)->halo_elements = -1;


        if ((*context)->flags & GHOST_CONTEXT_DIST_NZ)
        { // read rpt and fill lfrow, lnrows, lfent, lnents
            ghost_mnnz_t gnnz;
            if (!matrixSource) {
                ERROR_LOG("If distribution by nnz a matrix source has to be given!");
                ret = GHOST_ERR_INVALID_ARG;
                goto err;
            }


            if (me == 0) {
                GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->rpt,sizeof(ghost_mnnz_t)*((*context)->gnrows+1)),err,ret);
#pragma omp parallel for schedule(runtime)
                for( row = 0; row < (*context)->gnrows+1; row++ ) {
                    (*context)->rpt[row] = 0;
                }
                if ((*context)->flags & GHOST_CONTEXT_ROWS_FROM_FILE) {
                    GHOST_CALL_GOTO(ghost_readRpt((*context)->rpt,(char *)matrixSource,0,(*context)->gnrows+1),err,ret);
                } else if ((*context)->flags & GHOST_CONTEXT_ROWS_FROM_FUNC) {
                    ghost_sparsemat_fromRowFunc_t func;
                    if (sizeof(void *) != sizeof(ghost_sparsemat_fromRowFunc_t)) {
                        ERROR_LOG("Object pointers are of different size than function pointers. That probably means that you are not running on a POSIX-compatible OS.");
                        goto err;
                    }

                    memcpy(&func,&matrixSource,sizeof(ghost_sparsemat_fromRowFunc_t));
                    GHOST_CALL_GOTO(ghost_malloc((void **)&tmpval,(*context)->gncols*sizeof(complex double)),err,ret);
                    GHOST_CALL_GOTO(ghost_malloc((void **)&tmpcol,(*context)->gncols*sizeof(ghost_midx_t)),err,ret);
                    (*context)->rpt[0] = 0;
                    ghost_midx_t rowlen;
                    for(row = 0; row < (*context)->gnrows; row++) {
                        func(row,&rowlen,tmpcol,tmpval);
                        (*context)->rpt[row+1] = (*context)->rpt[row]+rowlen;
                    }
                    free(tmpval); tmpval = NULL;
                    free(tmpcol); tmpcol = NULL;
                } else {
                    ERROR_LOG("If distribution by nnz the type of the matrix source has to be specified in the flags!");
                    ret = GHOST_ERR_INVALID_ARG;
                    goto err;
                }

                gnnz = (*context)->rpt[(*context)->gnrows];
                ghost_mnnz_t target_nnz;
                target_nnz = (gnnz/nranks)+1; /* sonst bleiben welche uebrig! */

                (*context)->lfRow[0]  = 0;
                (*context)->lfEnt[0] = 0;
                int j = 1;

                for (row=0;row<(*context)->gnrows;row++){
                    if ((*context)->rpt[row] >= j*target_nnz){
                        (*context)->lfRow[j] = row;
                        (*context)->lfEnt[j] = (*context)->rpt[row];
                        j = j+1;
                    }
                }
                for (i=0; i<nranks-1; i++){
                    (*context)->lnrows[i] = (*context)->lfRow[i+1] - (*context)->lfRow[i] ;
                    (*context)->lnEnts[i] = (*context)->lfEnt[i+1] - (*context)->lfEnt[i] ;
                }

                (*context)->lnrows[nranks-1] = (*context)->gnrows - (*context)->lfRow[nranks-1] ;
                (*context)->lnEnts[nranks-1] = gnnz - (*context)->lfEnt[nranks-1];

                //fclose(filed);
            }
            MPI_CALL_GOTO(MPI_Bcast((*context)->lfRow,  nranks, ghost_mpi_dt_midx, 0, (*context)->mpicomm),err,ret);
            MPI_CALL_GOTO(MPI_Bcast((*context)->lfEnt,  nranks, ghost_mpi_dt_midx, 0, (*context)->mpicomm),err,ret);
            MPI_CALL_GOTO(MPI_Bcast((*context)->lnrows, nranks, ghost_mpi_dt_midx, 0, (*context)->mpicomm),err,ret);
            MPI_CALL_GOTO(MPI_Bcast((*context)->lnEnts, nranks, ghost_mpi_dt_midx, 0, (*context)->mpicomm),err,ret);

        } else
        { // don't read rpt, only fill lfrow, lnrows, rest will be done after some matrix from*() function
            UNUSED(matrixSource);
            double allweights;
            MPI_CALL_GOTO(MPI_Allreduce(&weight,&allweights,1,MPI_DOUBLE,MPI_SUM,(*context)->mpicomm),err,ret)

            ghost_midx_t my_target_rows = (ghost_midx_t)((*context)->gnrows*((double)weight/(double)allweights));
            GHOST_CALL_GOTO(ghost_malloc((void **)&target_rows,nranks*sizeof(ghost_midx_t)),err,ret);

            MPI_CALL_GOTO(MPI_Allgather(&my_target_rows,1,ghost_mpi_dt_midx,target_rows,1,ghost_mpi_dt_midx,(*context)->mpicomm),err,ret);
                       
            (*context)->rpt = NULL;
            (*context)->lfRow[0] = 0;

            for (i=1; i<nranks; i++){
                (*context)->lfRow[i] = (*context)->lfRow[i-1]+target_rows[i-1];
            }
            for (i=0; i<nranks-1; i++){
                (*context)->lnrows[i] = (*context)->lfRow[i+1] - (*context)->lfRow[i] ;
            }
            (*context)->lnrows[nranks-1] = (*context)->gnrows - (*context)->lfRow[nranks-1] ;
            MPI_CALL_GOTO(MPI_Bcast((*context)->lfRow,  nranks, ghost_mpi_dt_midx, 0, (*context)->mpicomm),err,ret);
            MPI_CALL_GOTO(MPI_Bcast((*context)->lnrows, nranks, ghost_mpi_dt_midx, 0, (*context)->mpicomm),err,ret);
            (*context)->lnEnts[0] = -1;
            (*context)->lfEnt[0] = -1;

            free(target_rows); target_rows = NULL;
        }

    } else {
        (*context)->lnrows[0] = (*context)->gnrows;
        (*context)->lfRow[0] = 0;
        (*context)->lnEnts[0] = 0;
        (*context)->lfEnt[0] = 0;
    }

#else
    UNUSED(weight);
    (*context)->lnrows[0] = (*context)->gnrows;
    (*context)->lfRow[0] = 0;
    (*context)->lnEnts[0] = 0;
    (*context)->lfEnt[0] = 0;
#endif

    DEBUG_LOG(1,"Context created successfully");
    goto out;
err:
    if (*context) {
        free((*context)->lnEnts); (*context)->lnEnts = NULL;
        free((*context)->lfEnt); (*context)->lfEnt = NULL;
        free((*context)->lnrows); (*context)->lnrows = NULL;
        free((*context)->lfRow); (*context)->lfRow = NULL;
        free((*context)->wishlist); (*context)->wishlist = NULL;
        free((*context)->duelist); (*context)->duelist = NULL;
        free((*context)->rpt); (*context)->rpt = NULL;
    }
    free(*context); *context = NULL;


out:
    free(tmpval); tmpval = NULL;
    free(tmpcol); tmpcol = NULL;
    free(target_rows); target_rows = NULL;
    return ret;
}

ghost_error_t ghost_printContextInfo(char **str, ghost_context_t *context)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);
    int nranks;
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(context->mpicomm,&nranks));

    char *contextType = "";
    if (context->flags & GHOST_CONTEXT_DISTRIBUTED)
        contextType = "Distributed";
    else if (context->flags & GHOST_CONTEXT_REDUNDANT)
        contextType = "Redundant";


    ghost_printHeader(str,"Context");
    ghost_printLine(str,"MPI processes",NULL,"%d",nranks);
    ghost_printLine(str,"Number of rows",NULL,"%"PRmatIDX,context->gnrows);
    ghost_printLine(str,"Type",NULL,"%s",contextType);
    ghost_printLine(str,"Work distribution scheme",NULL,"%s",ghost_workdistName(context->flags));
    ghost_printFooter(str);
    return GHOST_SUCCESS;

}

void ghost_destroyContext(ghost_context_t *context)
{
    DEBUG_LOG(1,"Freeing context");
    if (context != NULL) {
        free(context->rowPerm);
        free(context->invRowPerm);

        free(context);
    }
    DEBUG_LOG(1,"Context freed successfully");
}

ghost_error_t ghost_globalIndex(ghost_context_t *ctx, ghost_midx_t lidx, ghost_midx_t *gidx)
{
    int rank;
    GHOST_CALL_RETURN(ghost_getRank(ctx->mpicomm,&rank));

    if (ctx->flags & GHOST_CONTEXT_DISTRIBUTED) {
        *gidx = ctx->lfRow[rank] + lidx;
    } else {
        *gidx = lidx;
    }

    return GHOST_SUCCESS;    
}
ghost_error_t ghost_setupCommunication(ghost_context_t *ctx, ghost_midx_t *col)
{

    ghost_error_t ret = GHOST_SUCCESS;
    ghost_mnnz_t j;
    ghost_mnnz_t i;
    ghost_mnnz_t max_loc_elements, thisentry;
    ghost_mnnz_t *present_values = NULL;
    ghost_mnnz_t acc_dues;
    ghost_mnnz_t *tmp_transfers = NULL;
    ghost_mnnz_t acc_wishes;

    ghost_mnnz_t *item_from = NULL;

    ghost_mnnz_t *wishlist_counts = NULL;

    ghost_midx_t **wishlist = NULL;
    ghost_midx_t **cwishlist = NULL;


    ghost_midx_t this_pseudo_col;
    ghost_midx_t *pseudocol = NULL;
    ghost_midx_t *globcol = NULL;
    ghost_midx_t *myrevcol = NULL;

    ghost_midx_t *comm_remotePE = NULL;
    ghost_midx_t *comm_remoteEl = NULL;
    ghost_midx_t *wishl_mem  = NULL;
    ghost_midx_t *duel_mem   = NULL;
    ghost_mnnz_t acc_transfer_wishes, acc_transfer_dues;

    size_t size_nint, size_col;
    size_t size_a2ai, size_nptr, size_pval;  
    size_t size_wish, size_dues;

    int nprocs;
    int me;
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(ctx->mpicomm,&nprocs));
    GHOST_CALL_RETURN(ghost_getRank(ctx->mpicomm,&me));

#ifdef GHOST_HAVE_MPI
    MPI_Request req[2*nprocs];
    MPI_Status stat[2*nprocs];
#endif

    size_nint = (size_t)( (size_t)(nprocs)   * sizeof(ghost_midx_t)  );
    size_nptr = (size_t)( nprocs             * sizeof(ghost_midx_t*) );
    size_a2ai = (size_t)( nprocs*nprocs * sizeof(ghost_midx_t)  );


    max_loc_elements = 0;
    for (i=0;i<nprocs;i++) {
        if (max_loc_elements<ctx->lnEnts[i]) {
            max_loc_elements = ctx->lnEnts[i];
        }
    }

    size_pval = (size_t)( max_loc_elements * sizeof(ghost_mnnz_t) );
    size_col  = (size_t)( (size_t)(ctx->lnEnts[me])   * sizeof( ghost_midx_t ) );

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

      

    GHOST_CALL_GOTO(ghost_malloc((void **)&item_from, size_nint),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&wishlist_counts, nprocs*sizeof(ghost_mnnz_t)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm_remotePE, size_col),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm_remoteEl, size_col),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&present_values, size_pval),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&tmp_transfers,  size_a2ai),err,ret); 

    for (i=0; i<nprocs; i++) wishlist_counts[i] = 0;

    
    for (i=0;i<ctx->lnEnts[me];i++){
        for (j=nprocs-1;j>=0; j--){
            if (ctx->lfRow[j]<col[i]+1) {
                comm_remotePE[i] = j;
                wishlist_counts[j]++;
                comm_remoteEl[i] = col[i] -ctx->lfRow[j];
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

    GHOST_CALL_GOTO(ghost_malloc((void **)&wishlist,size_nptr),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&cwishlist,size_nptr),err,ret);

    /*
     * wishlist  = <{NULL,NULL,NULL},{NULL,NULL,NULL},{NULL,NULL,NULL}>
     * cwishlist = <{NULL,NULL,NULL},{NULL,NULL,NULL},{NULL,NULL,NULL}>
     */

    for (i=0; i<nprocs; i++){
        GHOST_CALL_GOTO(ghost_malloc((void **)&cwishlist[i],wishlist_counts[i]*sizeof(ghost_midx_t)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&wishlist[i],wishlist_counts[i]*sizeof(ghost_midx_t)),err,ret);
    }
    /*
     * wishlist  = <{{0,0,0},{0,0,0},{0}},{{0,0,0},{0,0},{0}},{{0},NULL,{0,0,0,0}}>
     * cwishlist = <{{0,0,0},{0,0,0},{0}},{{0,0,0},{0,0},{0}},{{0},NULL,{0,0,0,0}}>
     */

    for (i=0;i<nprocs;i++) item_from[i] = 0;

    for (i=0;i<ctx->lnEnts[me];i++){
        wishlist[comm_remotePE[i]][item_from[comm_remotePE[i]]] = comm_remoteEl[i];
        item_from[comm_remotePE[i]]++;
    }
    /*
     * wishlist  = <{{0,1,1},{1,0,1},{0}},{{0,1,1},{0,1},{1}},{{0},NULL,{0,1,0,1}}> local column idx of wishes
     * item_from = <{3,3,1},{3,2,1},{1,0,4}> equal to wishlist_counts
     */


    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->wishes,nprocs*sizeof(ghost_mnnz_t)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->dues,nprocs*sizeof(ghost_mnnz_t)),err,ret); 

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
            ctx->wishes[i] = thisentry;
        } else {
            ctx->wishes[i] = 0; 
        }

    }

    /* 
     * cwishlist = <{{#,#,#},{1,0,#},{0}},{{0,1,#},{#,#},{1}},{{0},NULL,{#,#,#,#}}> compressed wish list
     * ctx->wishes = <{0,2,1},{2,0,1},{1,0,0}>
     */

#ifdef GHOST_HAVE_MPI
    MPI_CALL_GOTO(MPI_Allgather(ctx->wishes, nprocs, ghost_mpi_dt_midx, tmp_transfers, 
                nprocs, ghost_mpi_dt_midx, ctx->mpicomm),err,ret);
#endif

    for (i=0; i<nprocs; i++) {
        ctx->dues[i] = tmp_transfers[i*nprocs+me];
    }

    ctx->dues[me] = 0; 
    
    /* 
     * ctx->dues = <{0,2,1},{2,0,0},{1,1,0}>
     */

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

    GHOST_CALL_GOTO(ghost_malloc((void **)&pseudocol,size_col),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&globcol,size_col),err,ret);
    
    /*
     * pseudocol = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> PE where element is on
     * globcol   = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> local colidx of element
     */

    this_pseudo_col = ctx->lnrows[me];
    ctx->halo_elements = 0;
    ghost_midx_t tt = 0;
    i = me;
    int meHandled = 0;

    /*
     * col[i] = <{0,1,3,4,1,2,3},{0,1,5,1,2,3},{4,5,0,4,5}>
     */

    for (; i<nprocs; i++) { // iterate i=me,..,nprocs,0,..,me-1
        int t = 0;
        if (meHandled && (i == me)) continue;

        if (i != me){ 
            for (j=0;j<ctx->wishes[i];j++){
                pseudocol[ctx->halo_elements] = this_pseudo_col; 
                globcol[ctx->halo_elements]   = ctx->lfRow[i]+cwishlist[i][j]; 
                ctx->halo_elements++;
                this_pseudo_col++;
            }
            /*
             * pseudocol = <{0,1,2},{0,1,2},{0}> colidx of each halo element starting from 0
             * globcol   = <{2,3,4},{2,3,4},{2}> colidx of each halo element starting from lnrows[me]
             */

            // myrevcol maps the actual colidx to the new colidx
            DEBUG_LOG(2,"Allocating space for myrevcol");
            GHOST_CALL_GOTO(ghost_malloc((void **)&myrevcol,ctx->lnrows[i]*sizeof(ghost_midx_t)),err,ret);
            for (j=0;j<ctx->wishes[i];j++){
                myrevcol[globcol[tt]-ctx->lfRow[i]] = tt;
                tt++;
            }
            /*
             * 1st iter: myrevcol = <{1,0},{0,1},{0,#}>
             * 2nd iter: myrevcol = <{2,#},{#,2},{#,#}>
             */

            for (;t<ctx->lnEnts[me];t++) {
                if (comm_remotePE[t] == i) { // local element for rank i
                    col[t] =  pseudocol[myrevcol[col[t]-ctx->lfRow[i]]];
                }
            }
            free(myrevcol); myrevcol = NULL;
        } else { // first i iteration goes here
            for (;t<ctx->lnEnts[me];t++) {
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


    size_wish = (size_t)( acc_transfer_wishes * sizeof(ghost_midx_t) );
    size_dues = (size_t)( acc_transfer_dues   * sizeof(ghost_midx_t) );

    // we need a contiguous array in memory
    GHOST_CALL_GOTO(ghost_malloc((void **)&wishl_mem,size_wish),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&duel_mem,size_dues),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->hput_pos,size_nptr),err,ret); 

    acc_dues = 0;
    acc_wishes = 0;


    for (i=0; i<nprocs; i++){

        ctx->duelist[i]    = &(duel_mem[acc_dues]);
        ctx->wishlist[i]   = &(wishl_mem[acc_wishes]);
        ctx->hput_pos[i]   = ctx->lnrows[me]+acc_wishes;

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
    for(i=0; i<nprocs; i++) 
    { // receive _my_ dues from _other_ processes' wishes
        MPI_CALL_GOTO(MPI_Irecv(ctx->duelist[i],ctx->dues[i],ghost_mpi_dt_midx,i,i,ctx->mpicomm,&req[msgcount]),err,ret);
        msgcount++;
    }


    for(i=0; i<nprocs; i++) { 
        MPI_CALL_GOTO(MPI_Isend(ctx->wishlist[i],ctx->wishes[i],ghost_mpi_dt_midx,i,me,ctx->mpicomm,&req[msgcount]),err,ret);
        msgcount++;
    }

    MPI_CALL_GOTO(MPI_Waitall(msgcount,req,stat),err,ret);
#endif

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

out:
    for (i=0; i<nprocs; i++) {
        free(wishlist[i]); wishlist[i] = NULL;
    }
    free(wishlist); wishlist = NULL;
    for (i=0; i<nprocs; i++) {
        free(cwishlist[i]); cwishlist[i] = NULL;
    }
    free(cwishlist); cwishlist = NULL;
    free(tmp_transfers); tmp_transfers = NULL;
    free(wishlist_counts); wishlist_counts = NULL;
    free(item_from); item_from = NULL;
    free(comm_remotePE); comm_remotePE = NULL;
    free(comm_remoteEl); comm_remoteEl = NULL;
    free(present_values); present_values = NULL;
    free(pseudocol); pseudocol = NULL;
    free(globcol); globcol = NULL;
    free(myrevcol); myrevcol = NULL;
    
    return ret;
}
