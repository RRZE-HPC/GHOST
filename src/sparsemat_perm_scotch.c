#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"

#ifdef GHOST_HAVE_SCOTCH
#ifdef GHOST_HAVE_MPI
#include <ptscotch.h>
#else
#include <scotch.h>
#endif
#endif

#ifdef GHOST_HAVE_MPI
#if 0
//! MPI_Allreduce for more than INT_MAX elements
static int MPI_Allreduce64_in_place ( void *buf, int64_t count,
                      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm )
  {
    int ierr;
    int64_t i;
    int sz;
    ierr=MPI_Type_size(datatype,&sz);
    if (ierr) return ierr;
    int chunksize=count>INT_MAX? INT_MAX: (int)count;
    for (i=0; i<count; i+=chunksize)
    {
      char *rbuf=(char*)buf+i*sz;
      
      int icount= i+chunksize>count? (count-i): chunksize;
        ierr=MPI_Allreduce(MPI_IN_PLACE, (void*)rbuf, icount,
                        datatype, op, comm);
      if (ierr) return ierr;
    }
    return ierr;
  }
#endif
#endif

ghost_error ghost_sparsemat_perm_scotch(ghost_context *ctx, ghost_sparsemat *mat)
{
#ifndef GHOST_HAVE_SCOTCH
    UNUSED(ctx);
    UNUSED(mat);
    WARNING_LOG("Scotch not available. Will not create matrix permutation!");
    return GHOST_SUCCESS;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    ghost_error ret = GHOST_SUCCESS;
    ghost_gidx k;
    ghost_lidx i,j;
    ghost_sorting_helper *rowSort = NULL;
    ghost_gidx *col_loopless = NULL;
    ghost_gidx *rpt_loopless = NULL;
    ghost_gidx nnz = 0;
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

    if (mat->context->row_map->glb_perm) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }


    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(mat->context->row_map->ldim[me]+1) * sizeof(ghost_gidx)),err,ret);
    
    GHOST_INSTR_START("scotch_readin")

    nnz = SPM_NNZ(mat); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_gidx)),err,ret);
    ghost_lidx *rpt = mat->chunkStart;
    ghost_gidx *col = mat->col_orig;
    
    ghost_malloc((void **)&col_loopless,nnz*sizeof(ghost_gidx));
    ghost_malloc((void **)&rpt_loopless,(SPM_NROWS(mat)+1)*sizeof(ghost_gidx));
    rpt_loopless[0] = 0;
    ghost_gidx nnz_loopless = 0;

    // eliminate loops by deleting diagonal entries
    for (i=0; i<SPM_NROWS(mat); i++) {
        for (j=rpt[i]; j<rpt[i+1]; j++) {
            if (col[j] != i) {
                col_loopless[nnz_loopless] = col[j];
                nnz_loopless++;
            }
        }
        rpt_loopless[i+1] = nnz_loopless;
    }

            
    GHOST_INSTR_STOP("scotch_readin")

    GHOST_CALL_GOTO(ghost_malloc((void **)mat->context->row_map->glb_perm,sizeof(ghost_gidx)*mat->context->row_map->ldim[me]),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)mat->context->row_map->glb_perm_inv,sizeof(ghost_gidx)*mat->context->row_map->ldim[me]),err,ret);
    mat->context->col_map->glb_perm = NULL;
    mat->context->col_map->glb_perm_inv = NULL;
    //mat->context->perm_global->method = GHOST_PERMUTATION_SYMMETRIC;

    memset(mat->context->row_map->glb_perm,0,sizeof(ghost_gidx)*mat->context->row_map->ldim[me]);
    memset(mat->context->row_map->glb_perm_inv,0,sizeof(ghost_gidx)*mat->context->row_map->ldim[me]);

#ifdef GHOST_HAVE_MPI
    GHOST_INSTR_START("scotch_createperm")
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
    SCOTCH_CALL_GOTO(SCOTCH_dgraphBuild(dgraph, 0, (ghost_gidx)mat->context->row_map->dim, mat->context->row_map->dim, rpt_loopless, rpt_loopless+1, NULL, NULL, nnz_loopless, nnz_loopless, col_loopless, NULL, NULL),err,ret);
//    SCOTCH_CALL_GOTO(SCOTCH_dgraphCheck(dgraph),err,ret);

    SCOTCH_CALL_GOTO(SCOTCH_stratInit(strat),err,ret);
    
    /* use strategy string from traits */
    SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrder(strat,mat->traits.scotchStrat),err,ret);

    /* or use some default strategy */
    /* \todo: I'm not sure what the 'balrat' value does (last param),
       I am assuming: allow at most 20% load imbalance (balrat=0.2))
     */
//     int flags=SCOTCH_STRATSPEED|
//              SCOTCH_STRATSCALABILITY|
//              SCOTCH_STRATBALANCE;
//    SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrderBuild(strat, flags, 
//        (ghost_gidx)nprocs,0,0.1),err,ret);
        if (me==0)
        {
          INFO_LOG("SCOTCH strategy used:");
          if (GHOST_VERBOSITY)
          {
            SCOTCH_CALL_GOTO(SCOTCH_stratSave(strat,stderr),err,ret);
            fprintf(stderr,"\n");
          }
        }
    
    dorder = SCOTCH_dorderAlloc();
    if (!dorder) {
        ERROR_LOG("Could not alloc SCOTCH order");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderInit(dgraph,dorder),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderCompute(dgraph,dorder,strat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderPerm(dgraph,dorder,mat->context->row_map->glb_perm),err,ret);
    GHOST_INSTR_STOP("scotch_createperm")
    

    GHOST_INSTR_START("scotch_combineperm")
    ghost_global_perm_inv(mat->context->row_map->glb_perm_inv,mat->context->row_map->glb_perm,mat->context);
    GHOST_INSTR_STOP("scotch_combineperm")

#else

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
    SCOTCH_CALL_GOTO(SCOTCH_graphBuild(graph, 0, (ghost_gidx)SPM_NROWS(mat), rpt_loopless, rpt_loopless+1, NULL, NULL, nnz_loopless, col_loopless, NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphCheck(graph),err,ret);
    
    order = SCOTCH_orderAlloc();
    if (!order) {
        ERROR_LOG("Could not alloc SCOTCH order");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderInit(graph,order,mat->context->row_map->glb_perm,NULL,NULL,NULL,NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_stratGraphOrder(strat,mat->traits->scotchStrat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderCompute(graph,order,strat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderCheck(graph,order),err,ret);

    for (i=0; i<SPM_NROWS(mat); i++) {
        mat->context->row_map->glb_perm_inv[mat->context->row_map->glb_perm[i]] = i;
    }

#endif
   
    goto out;
err:
    ERROR_LOG("Deleting permutations");
    free(mat->context->row_map->glb_perm); mat->context->row_map->glb_perm = NULL;
    free(mat->context->row_map->glb_perm_inv); mat->context->row_map->glb_perm_inv = NULL;
    free(mat->context->row_map->glb_perm); mat->context->row_map->glb_perm = NULL;
    free(mat->context->row_map->glb_perm_inv); mat->context->row_map->glb_perm_inv = NULL;

out:
    free(rpt);
    free(rowSort);
    free(col);
    free(rpt_loopless);
    free(col_loopless);
#ifdef GHOST_HAVE_MPI
    if (dgraph != NULL && dorder != NULL) {
        SCOTCH_dgraphOrderExit(dgraph,dorder);
    }
    if (dgraph) {
        SCOTCH_dgraphExit(dgraph);
    }
#else
    if (graph != NULL && order != NULL) {
        SCOTCH_graphOrderExit(graph,order);
    }
    if (graph) {
        SCOTCH_graphExit(graph);
    }
#endif
    if (strat) {
        SCOTCH_stratExit(strat);
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    
    
    return ret;

#endif
}

