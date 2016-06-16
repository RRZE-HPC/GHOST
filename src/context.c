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
#include <float.h>
#include <math.h>

ghost_error ghost_context_create(ghost_context **context, ghost_gidx gnrows, ghost_gidx gncols, ghost_context_flags_t context_flags, void *matrixSource, ghost_sparsemat_src srcType, ghost_mpi_comm comm, double weight) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    if (weight < 0) {
        ERROR_LOG("Negative weight");
        return GHOST_ERR_INVALID_ARG;
    }
    if (fabs(weight) < DBL_MIN) {
        double max_bw=0.0;
        ghost_bench_stream(GHOST_BENCH_STREAM_COPY,&weight,&max_bw);
        INFO_LOG("Automatically setting weight to %f according to STREAM copy bandwidth!",weight);
    }
          
    int nranks, me, i;
    ghost_error ret = GHOST_SUCCESS;
    
    ghost_lidx *target_rows = NULL;
    char *tmpval = NULL;
    ghost_gidx *tmpcol = NULL;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)context,sizeof(ghost_context)),err,ret);
    (*context)->flags = context_flags;
    (*context)->mpicomm = comm;
    (*context)->mpicomm_parent = MPI_COMM_NULL;
    (*context)->perm_local = NULL;
    (*context)->perm_global = NULL;
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
   

    GHOST_CALL_GOTO(ghost_nrank(&nranks, (*context)->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, (*context)->mpicomm),err,ret);

    if (!((*context)->flags & GHOST_CONTEXT_DIST_NZ)) {
        (*context)->flags |= (ghost_context_flags_t)GHOST_CONTEXT_DIST_ROWS;
    }

    if ((gnrows == 0) || (gncols == 0)) {
        if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
            ERROR_LOG("The correct dimensions have to be given if the sparsemat source is a function!");
            return GHOST_ERR_INVALID_ARG;
        } else if (srcType == GHOST_SPARSEMAT_SRC_FILE) {
            ghost_sparsemat_rowfunc_bincrs_initargs args;
            args.filename = (char *)matrixSource;
            
            ghost_gidx dim[2]; 
            ghost_sparsemat_rowfunc_bincrs(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETDIM,NULL,dim,&args,NULL);
#ifndef GHOST_IDX64_GLOBAL
            if (dim[0] >= (int64_t)INT_MAX) {
                ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with 64 bit indices enabled!");
                return GHOST_ERR_DATATYPE;
            }
#endif
            if (gnrows == 0) {
                (*context)->gnrows = (ghost_gidx)dim[0];
            }
            if (gncols == 0) {
                (*context)->gncols = (ghost_gidx)dim[1];
            }
#if 0
            ghost_bincrs_header_t fileheader;
            GHOST_CALL_GOTO(ghost_bincrs_header_read(&fileheader,(char *)matrixSource),err,ret);
#ifndef GHOST_IDX64_GLOBAL
            if (fileheader.nrows >= (int64_t)INT_MAX) {
                ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with 64 bit indices enabled!");
                return GHOST_ERR_DATATYPE;
            }
#endif
            if (gnrows == 0) {
                (*context)->gnrows = (ghost_gidx)fileheader.nrows;
            }
            if (gncols == 0) {
                (*context)->gncols = (ghost_gidx)fileheader.ncols;
            }
#endif
        } else if (srcType == GHOST_SPARSEMAT_SRC_MM) {
            ghost_sparsemat_rowfunc_mm_initargs args;
            args.filename = (char *)matrixSource;
            
            ghost_gidx dim[2]; 
            ghost_sparsemat_rowfunc_mm(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETDIM,NULL,dim,&args,NULL);
#ifndef GHOST_IDX64_GLOBAL
            if (dim[0] >= (int64_t)INT_MAX) {
                ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with 64 bit indices enabled!");
                return GHOST_ERR_DATATYPE;
            }
#endif
            if (gnrows == 0) {
                (*context)->gnrows = (ghost_gidx)dim[0];
            }
            if (gncols == 0) {
                (*context)->gncols = (ghost_gidx)dim[1];
            }
        }


    } else if ((gnrows < 0) || (gncols < 0)) {
            ERROR_LOG("The given context dimensions are smaller than zero which may be due to an integer overlow. Check your idx types!");
            return GHOST_ERR_DATATYPE;
    } else {
#ifndef GHOST_IDX64_GLOBAL
        if (gnrows >= (int64_t)INT_MAX) {
            ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with 64 bit indices enabled!");
            return GHOST_ERR_DATATYPE;
        }
#endif
        (*context)->gnrows = (ghost_gidx)gnrows;
        (*context)->gncols = (ghost_gidx)gncols;
    }
    DEBUG_LOG(1,"Creating context with dimension %"PRGIDX"x%"PRGIDX,(*context)->gnrows,(*context)->gncols);

    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->lnEnts, nranks*sizeof(ghost_lidx)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->lfEnt, nranks*sizeof(ghost_gidx)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->lnrows, nranks*sizeof(ghost_lidx)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->lfRow, nranks*sizeof(ghost_gidx)),err,ret);

#ifdef GHOST_HAVE_MPI
    ghost_lidx row;
    (*context)->halo_elements = -1;
/*
    if ((*context)->flags & GHOST_CONTEXT_PERMUTED) {
        INFO_LOG("Reducing matrix bandwidth");
        ghost_error ret = GHOST_SUCCESS;
        if ((*context)->rowPerm || (*context)->invRowPerm) {
            WARNING_LOG("Existing permutations will be overwritten!");
        }

        char *matrixPath = (char *)matrixSource;
        ghost_idx_t *rpt = NULL, *col = NULL, i;
        int me, nprocs;
        
        ghost_bincrs_header_t header;
        ghost_bincrs_header_read(matrixPath,&header);
        MPI_Request *req = NULL;
        MPI_Status *stat = NULL;

        GHOST_CALL_GOTO(ghost_rank(&me, (*context)->mpicomm),err,ret);
        GHOST_CALL_GOTO(ghost_nrank(&nprocs, (*context)->mpicomm),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&req,sizeof(MPI_Request)*nprocs),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&stat,sizeof(MPI_Status)*nprocs),err,ret);
            
        ghost_idx_t target_rows = (ghost_idx_t)((*context)->gnrows/nranks);

        (*context)->lfRow[0] = 0;

        for (i=1; i<nranks; i++){
            (*context)->lfRow[i] = (*context)->lfRow[i-1]+target_rows;
        }
        for (i=0; i<nranks-1; i++){
            (*context)->lnrows[i] = (*context)->lfRow[i+1] - (*context)->lfRow[i] ;
        }
        (*context)->lnrows[nranks-1] = (*context)->gnrows - (*context)->lfRow[nranks-1] ;
                
        GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,((*context)->lnrows[me]+1) * sizeof(ghost_idx_t)),err,ret);
        GHOST_CALL_GOTO(ghost_bincrs_rpt_read(rpt, matrixPath, (*context)->lfRow[me], (*context)->lnrows[me]+1, NULL),err,ret);


        ghost_idx_t nnz = rpt[(*context)->lnrows[me]]-rpt[0];
            
        GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_idx_t)),err,ret);
        GHOST_CALL_GOTO(ghost_readCol(col, matrixPath, (*context)->lfRow[me], (*context)->lnrows[me], NULL,NULL),err,ret);
        
        WARNING_LOG("nnz: %d",nnz); 
        for (i=1;i<(*context)->lnrows[me]+1;i++) {
            rpt[i] -= rpt[0];
            WARNING_LOG("rpt[%d]: %d",i,rpt[i]); 
        }
        rpt[0] = 0;

        ghost_idx_t j;
        for (i=0;i<(*context)->lnrows[me];i++) {
            for (j=rpt[i];j<rpt[i+1];j++) {
                WARNING_LOG("col[%d]: %d",j,col[j]);
            }
        }
        
        GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->rowPerm,sizeof(ghost_idx_t)*(*context)->gnrows),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->invRowPerm,sizeof(ghost_idx_t)*(*context)->gnrows),err,ret);
        memset((*context)->rowPerm,0,sizeof(ghost_idx_t)*(*context)->gnrows);
        memset((*context)->rowPerm,0,sizeof(ghost_idx_t)*(*context)->gnrows);
        
        SCOTCH_Dgraph * dgraph = SCOTCH_dgraphAlloc();
        if (!dgraph) {
            ERROR_LOG("Could not alloc SCOTCH graph");
            ret = GHOST_ERR_SCOTCH;
            goto err;
        }
        SCOTCH_CALL_GOTO(SCOTCH_dgraphInit(dgraph,(*context)->mpicomm),err,ret);
        SCOTCH_Strat * strat = SCOTCH_stratAlloc();
        if (!strat) {
            ERROR_LOG("Could not alloc SCOTCH strat");
            ret = GHOST_ERR_SCOTCH;
            goto err;
        }
        SCOTCH_CALL_GOTO(SCOTCH_stratInit(strat),err,ret);
        SCOTCH_Dordering *dorder = SCOTCH_dorderAlloc();
        if (!dorder) {
            ERROR_LOG("Could not alloc SCOTCH order");
            ret = GHOST_ERR_SCOTCH;
            goto err;
        }
        SCOTCH_CALL_GOTO(SCOTCH_dgraphBuild(dgraph, 0, (*context)->lnrows[me], (*context)->lnrows[me], rpt, rpt+1, NULL, NULL, nnz, nnz, col, NULL, NULL),err,ret);
        SCOTCH_CALL_GOTO(SCOTCH_dgraphCheck(dgraph),err,ret);
        SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderInit(dgraph,dorder),err,ret);
        SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrder(strat,"n{sep=m{asc=b,low=b},ole=q{strat=g},ose=q{strat=g},osq=g}"),err,ret);
        SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderCompute(dgraph,dorder,strat),err,ret);
        SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderPerm(dgraph,dorder,(*context)->rowPerm+(*context)->lfRow[me]),err,ret);

        // combine permutation vectors
        MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,(*context)->rowPerm,(*context)->gnrows,ghost_mpi_dt_idx,MPI_MAX,(*context)->mpicomm),err,ret);

        // assemble inverse permutation
        for (i=0; i<(*context)->gnrows; i++) {
            (*context)->invRowPerm[(*context)->rowPerm[i]] = i;
        }
        for (i=0; i<(*context)->gnrows; i++) {
            INFO_LOG("perm[%d] = %d",i,(*context)->rowPerm[i]);
        }
    }
                
*/

    if ((*context)->flags & GHOST_CONTEXT_DIST_NZ)
    { // read rpt and fill lfrow, lnrows, lfent, lnents
        ghost_gidx gnnz;
        if (!matrixSource) {
            ERROR_LOG("If distribution by nnz a matrix source has to be given!");
            ret = GHOST_ERR_INVALID_ARG;
            goto err;
        }
        WARNING_LOG("Will not take into account possible matrix re-ordering when dividing the matrix by number of non-zeros!");


        if (me == 0) {
            GHOST_CALL_GOTO(ghost_malloc((void **)&(*context)->rpt,sizeof(ghost_gidx)*((*context)->gnrows+1)),err,ret);
#pragma omp parallel for schedule(runtime)
            for( row = 0; row < (*context)->gnrows+1; row++ ) {
                (*context)->rpt[row] = 0;
            }
            if (srcType == GHOST_SPARSEMAT_SRC_FILE) {
                ghost_sparsemat_rowfunc_bincrs_initargs args;
                args.filename = (char *)matrixSource;
                
                ghost_sparsemat_rowfunc_bincrs(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETRPT,NULL,(*context)->rpt,&args,NULL);
                //GHOST_CALL_GOTO(ghost_bincrs_rpt_read((*context)->rpt,(char *)matrixSource,0,(*context)->gnrows+1,NULL),err,ret);
            } else if (srcType == GHOST_SPARSEMAT_SRC_MM) {
                ghost_sparsemat_rowfunc_mm_initargs args;
                args.filename = (char *)matrixSource;
                
                ghost_sparsemat_rowfunc_mm(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETRPT,NULL,(*context)->rpt,&args,NULL);


            } else if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
                ghost_sparsemat_src_rowfunc *matsrc = (ghost_sparsemat_src_rowfunc *)matrixSource;
                GHOST_CALL_GOTO(ghost_malloc((void **)&tmpval,matsrc->maxrowlen*GHOST_DT_MAX_SIZE),err,ret);
                GHOST_CALL_GOTO(ghost_malloc((void **)&tmpcol,matsrc->maxrowlen*sizeof(ghost_gidx)),err,ret);
                (*context)->rpt[0] = 0;
                ghost_lidx rowlen;
                for(row = 0; row < (*context)->gnrows; row++) {
                    matsrc->func(row,&rowlen,tmpcol,tmpval,matsrc->arg);
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
            ghost_lidx target_nnz;
            target_nnz = (gnnz/nranks)+1; /* sonst bleiben welche uebrig! */

            (*context)->lfRow[0]  = 0;
            (*context)->lfEnt[0] = 0;
            ghost_lidx j = 1;

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
        MPI_CALL_GOTO(MPI_Bcast((*context)->lfRow,  nranks, ghost_mpi_dt_gidx, 0, (*context)->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Bcast((*context)->lfEnt,  nranks, ghost_mpi_dt_gidx, 0, (*context)->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Bcast((*context)->lnrows, nranks, ghost_mpi_dt_lidx, 0, (*context)->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Bcast((*context)->lnEnts, nranks, ghost_mpi_dt_lidx, 0, (*context)->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Allreduce(&((*context)->lnEnts[me]),&((*context)->gnnz),1,ghost_mpi_dt_gidx,MPI_SUM,(*context)->mpicomm),err,ret);

    } else
    { // don't read rpt, only fill lfrow, lnrows, rest will be done after some matrix from*() function
        UNUSED(matrixSource);
        double allweights;
        MPI_CALL_GOTO(MPI_Allreduce(&weight,&allweights,1,MPI_DOUBLE,MPI_SUM,(*context)->mpicomm),err,ret)

        ghost_lidx my_target_rows = (ghost_lidx)((*context)->gnrows*((double)weight/(double)allweights));
        if (my_target_rows == 0) {
            WARNING_LOG("This rank will have zero rows assigned!");
        }

        GHOST_CALL_GOTO(ghost_malloc((void **)&target_rows,nranks*sizeof(ghost_lidx)),err,ret);

        MPI_CALL_GOTO(MPI_Allgather(&my_target_rows,1,ghost_mpi_dt_lidx,target_rows,1,ghost_mpi_dt_lidx,(*context)->mpicomm),err,ret);
                   
        (*context)->rpt = NULL;
        (*context)->lfRow[0] = 0;

        for (i=1; i<nranks; i++){
            (*context)->lfRow[i] = (*context)->lfRow[i-1]+target_rows[i-1];
        }
        for (i=0; i<nranks-1; i++){
            ghost_gidx lnrows = (*context)->lfRow[i+1] - (*context)->lfRow[i];
            if (lnrows > (ghost_gidx)GHOST_LIDX_MAX) {
                ERROR_LOG("Re-compile with 64-bit local indices!");
                return GHOST_ERR_UNKNOWN;
            }
            (*context)->lnrows[i] = (ghost_lidx)lnrows;
        }
        ghost_gidx lnrows = (*context)->gnrows - (*context)->lfRow[nranks-1];
        if (lnrows > (ghost_gidx)GHOST_LIDX_MAX) {
            ERROR_LOG("The local number of rows (%"PRGIDX") exceeds the maximum range. Re-compile with 64-bit local indices!",lnrows);
            return GHOST_ERR_DATATYPE;
        }
        (*context)->lnrows[nranks-1] = (ghost_lidx)lnrows;
        
        //MPI_CALL_GOTO(MPI_Bcast((*context)->lfRow,  nranks, ghost_mpi_dt_gidx, 0, (*context)->mpicomm),err,ret);
        //MPI_CALL_GOTO(MPI_Bcast((*context)->lnrows, nranks, ghost_mpi_dt_lidx, 0, (*context)->mpicomm),err,ret);
        (*context)->lnEnts[0] = -1;
        (*context)->lfEnt[0] = -1;
        (*context)->gnnz = -1;

        free(target_rows); target_rows = NULL;
    }


#else
    UNUSED(i);
    UNUSED(srcType);
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
    ghost_line_string(str,"Number of rows",NULL,"%"PRGIDX,context->gnrows);
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
        free(context->lfRow); context->lfRow = NULL;
        free(context->lnrows); context->lnrows = NULL;
        free(context->lnEnts); context->lnEnts = NULL;
        free(context->lfEnt); context->lfEnt = NULL;
        free(context->duepartners); context->duepartners = NULL;
        free(context->wishpartners); context->wishpartners = NULL;
        free(context->entsInCol); context->entsInCol = NULL;
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
    }

    free(context);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TEARDOWN);
}

ghost_error ghost_context_comm_init(ghost_context *ctx, ghost_gidx *col_orig, ghost_sparsemat *mat, ghost_lidx *col)
{
    if (ctx->wishlist != NULL) {
        INFO_LOG("The context already has communication information. This will not be done again! Destroy the context in case the matrix has changed!");
        return GHOST_SUCCESS;
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_SETUP);

    ghost_error ret = GHOST_SUCCESS;
    ghost_gidx j;
    ghost_gidx i;
    ghost_lidx max_loc_elements, thisentry;
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
    
    ghost_lidx rowpadding = ghost_densemat_row_padding();
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, ctx->mpicomm));
    GHOST_CALL_RETURN(ghost_rank(&me, ctx->mpicomm));

#ifdef GHOST_HAVE_MPI
    MPI_Request req[2*nprocs];
    MPI_Status stat[2*nprocs];
#endif

    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->entsInCol,ctx->lnrows[me]*sizeof(ghost_lidx)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->wishlist,nprocs*sizeof(ghost_lidx *)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->duelist,nprocs*sizeof(ghost_lidx *)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->wishes,nprocs*sizeof(ghost_lidx)),err,ret); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->dues,nprocs*sizeof(ghost_lidx)),err,ret);

    memset(ctx->entsInCol,0,ctx->lnrows[me]*sizeof(ghost_lidx));
       
    ghost_lidx chunk,rowinchunk,entinrow,globalent,globalrow;
    for(chunk = 0; chunk < mat->nrowsPadded/mat->traits.C; chunk++) {
        for (rowinchunk=0; rowinchunk<mat->traits.C; rowinchunk++) {
            globalrow = chunk*mat->traits.C+rowinchunk;
            if (globalrow < ctx->lnrows[me]) {
                for (entinrow=0; entinrow<SELL(mat)->rowLen[globalrow]; entinrow++) {
                    globalent = SELL(mat)->chunkStart[chunk] + entinrow*mat->traits.C + rowinchunk;
		    if (col_orig[globalent] >= ctx->lfRow[me] && col_orig[globalent]<(ctx->lfRow[me]+ctx->lnrows[me])) {
                        ctx->entsInCol[col_orig[globalent]-ctx->lfRow[me]]++;
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


    max_loc_elements = 0;
    for (i=0;i<nprocs;i++) {
        if (max_loc_elements<ctx->lnEnts[i]) {
            max_loc_elements = ctx->lnEnts[i];
        }
    }

    size_pval = (size_t)( max_loc_elements * sizeof(ghost_lidx) );
    size_lcol  = (size_t)( (size_t)(ctx->lnEnts[me])   * sizeof( ghost_lidx ) );
    size_gcol  = (size_t)( (size_t)(ctx->lnEnts[me])   * sizeof( ghost_gidx ) );

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
        for (i=0;i<ctx->lnEnts[me];i++){
            for (j=nprocs-1;j>=0; j--){
                if (ctx->lfRow[j]<col_orig[i]+1) {//is col_orig unpermuted(probably)
                    comm_remotePE[i] = j;//comm_remotePE[colPerm[i]]=j
                    comm_remoteEl[i] = col_orig[i] -ctx->lfRow[j]; //comm_remoteEl[colPerm[i]] = col_orig[i] -ctx->lfRow[j];
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
    for (i=0;i<ctx->lnEnts[me];i++){
        wishlist[comm_remotePE[i]][item_from[comm_remotePE[i]]] = comm_remoteEl[i];
        item_from[comm_remotePE[i]]++;
    }
    GHOST_INSTR_STOP("wishlist");
    /*
     * wishlist  = <{{0,1,1},{1,0,1},{0}},{{0,1,1},{0,1},{1}},{{0},NULL,{0,1,0,1}}> local column idx of wishes
     * item_from = <{3,3,1},{3,2,1},{1,0,4}> equal to wishlist_counts
     */

    MPI_Barrier(ctx->mpicomm);

    GHOST_INSTR_START("wishes_and_dues");
    MPI_Win due_win,nduepartners_win;
    MPI_CALL_GOTO(MPI_Win_create(ctx->dues,nprocs*sizeof(ghost_lidx),sizeof(ghost_lidx),MPI_INFO_NULL,ctx->mpicomm,&due_win),err,ret);
    MPI_CALL_GOTO(MPI_Win_create(&ctx->nduepartners,sizeof(int),sizeof(int),MPI_INFO_NULL,ctx->mpicomm,&nduepartners_win),err,ret);

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

    GHOST_CALL_GOTO(ghost_malloc((void **)&pseudocol,size_lcol),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&globcol,size_gcol),err,ret);
    
    /*
     * pseudocol = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> PE where element is on
     * globcol   = <{0,0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0}> local colidx of element
     */

    this_pseudo_col = ctx->lnrows[me];
    ctx->halo_elements = 0;
    ghost_lidx tt = 0;
    i = me;
    int meHandled = 0;

 
    ghost_lidx rowpaddingoffset = 0;
     
     if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                ghost_lidx halo_ctr = 0;
		//we need to know number of halo elements now
 		for(int k=0;k<nprocs;++k) {
		    if (k != me){ 
        		    for (int j=0;j<ctx->wishes[k];j++){
				++halo_ctr;
 			    }
		    }
		}
  			
		ctx->nrowspadded   =  PAD(ctx->lnrows[me]+halo_ctr+1,rowpadding);
		rowpaddingoffset   =  ctx->nrowspadded-ctx->lnrows[me];
    } else {
	 //ctx->nrowspadded = PAD(ctx->lnrows[me],rowpadding);// this is set already
	 rowpaddingoffset = PAD(ctx->lnrows[me],rowpadding)-ctx->lnrows[me];
    }

    GHOST_INSTR_START("compress_cols")

    /*
     * col[i] = <{0,1,3,4,1,2,3},{0,1,5,1,2,3},{4,5,0,4,5}>
     */

    for (; i<nprocs; i++) { // iterate i=me,..,nprocs,0,..,me-1
        ghost_lidx t;
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
            GHOST_CALL_GOTO(ghost_malloc((void **)&myrevcol,ctx->lnrows[i]*sizeof(ghost_lidx)),err,ret);
            for (j=0;j<ctx->wishes[i];j++){
                myrevcol[globcol[tt]-ctx->lfRow[i]] = tt;
                tt++;
            }
            /*
             * 1st iter: myrevcol = <{1,0},{0,1},{0,#}>
             * 2nd iter: myrevcol = <{2,#},{#,2},{#,#}>
             */

#pragma omp parallel for
            for (t=0; t<ctx->lnEnts[me]; t++) {
                if (comm_remotePE[t] == i) { // local element for rank i
                    col[t] =  rowpaddingoffset + pseudocol[myrevcol[col_orig[t]-ctx->lfRow[i]]];
                }
            }
            free(myrevcol); myrevcol = NULL;
        } else { // first i iteration goes here
#pragma omp parallel for
            for (t=0; t<ctx->lnEnts[me]; t++) {
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

        if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {	
		ctx->hput_pos[i]   = ctx->nrowspadded + acc_wishes;
	} else {
        	ctx->hput_pos[i]   = PAD(ctx->lnrows[me],rowpadding)+acc_wishes;
	}

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
