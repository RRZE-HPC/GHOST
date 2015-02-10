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

ghost_error_t ghost_sparsemat_perm_scotch(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType)
{
#ifndef GHOST_HAVE_SCOTCH
    UNUSED(mat);
    UNUSED(matrixSource);
    UNUSED(srcType);
    WARNING_LOG("Scotch not available. Will not create matrix permutation!");
    return GHOST_SUCCESS;
#else
    GHOST_INSTR_START("scotch")
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_gidx_t *col = NULL, i, j, k, c;
    ghost_sorting_helper_t *rowSort = NULL;
    ghost_gidx_t *rpt = NULL;
    ghost_gidx_t *col_loopless = NULL;
    ghost_gidx_t *rpt_loopless = NULL;
    ghost_gidx_t nnz = 0;
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

    if (mat->context->permutation) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }

    if (srcType == GHOST_SPARSEMAT_SRC_NONE) {
        ERROR_LOG("A valid matrix source has to be given!");
        ret = GHOST_ERR_INVALID_ARG;
        goto err;
    }

    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(mat->context->lnrows[me]+1) * sizeof(ghost_gidx_t)),err,ret);
    
    GHOST_INSTR_START("scotch_readin")
    if (srcType == GHOST_SPARSEMAT_SRC_FILE) {
        char *matrixPath = (char *)matrixSource;
        GHOST_CALL_GOTO(ghost_bincrs_rpt_read(rpt, matrixPath, mat->context->lfRow[me], mat->context->lnrows[me]+1, NULL),err,ret);
#pragma omp parallel for
        for (i=1;i<mat->context->lnrows[me]+1;i++) {
            rpt[i] -= rpt[0];
        }
        rpt[0] = 0;

        nnz = rpt[mat->context->lnrows[me]];
        GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_gidx_t)),err,ret);
        GHOST_CALL_GOTO(ghost_bincrs_col_read(col, matrixPath, mat->context->lfRow[me], mat->context->lnrows[me], NULL,1),err,ret);

    } else if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
        ghost_sparsemat_src_rowfunc_t *src = (ghost_sparsemat_src_rowfunc_t *)matrixSource;
        char * tmpval = NULL;
        ghost_gidx_t * tmpcol = NULL;

        ghost_lidx_t rowlen;
        rpt[0] = 0;
#pragma omp parallel private (tmpval,tmpcol,i,rowlen) reduction(+:nnz)
        {
            ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
            ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t));
            
#pragma omp for
            for (i=0; i<mat->context->lnrows[me]; i++) {
                src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
                nnz += rowlen;
            }
            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_gidx_t)),err,ret);
        
#pragma omp parallel private (tmpval,tmpcol,i,j,rowlen) reduction(+:nnz)
        {
            ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
            ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t));
#pragma omp for ordered
            for (i=0; i<mat->context->lnrows[me]; i++) {
#pragma omp ordered
                {
                    src->func(mat->context->lfRow[me]+i,&rowlen,&col[rpt[i]],tmpval);
                    /* remove the diagonal entry ("self-edge") */
                    for (j=0;j<rowlen;j++)
                    {
                      if (col[rpt[i]+j]==mat->context->lfRow[me]+i)
                      {
                        for (k=j; k<rowlen-1;k++)
                        {
                          col[rpt[i]+k]=col[rpt[i]+k+1];
                        }
                        rowlen--;
                        break;
                      }
                    }
                    rpt[i+1] = rpt[i] + rowlen;
                }
            }
            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
            
    }
    nnz=rpt[mat->context->lnrows[me]];
    GHOST_INSTR_STOP("scotch_readin")

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation,sizeof(ghost_permutation_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation->perm,sizeof(ghost_gidx_t)*mat->context->gnrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation->invPerm,sizeof(ghost_gidx_t)*mat->context->gnrows),err,ret);
#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->context->permutation->cu_perm,sizeof(ghost_gidx_t)*mat->context->gnrows),err,ret);
#endif
    memset(mat->context->permutation->perm,0,sizeof(ghost_gidx_t)*mat->context->gnrows);
    memset(mat->context->permutation->invPerm,0,sizeof(ghost_gidx_t)*mat->context->gnrows);
    mat->context->permutation->scope = GHOST_PERMUTATION_GLOBAL;
    mat->context->permutation->len = mat->context->gnrows;

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
    SCOTCH_CALL_GOTO(SCOTCH_dgraphBuild(dgraph, 0, (ghost_gidx_t)mat->context->lnrows[me], mat->context->lnrows[me], rpt, rpt+1, NULL, NULL, nnz, nnz, col, NULL, NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphCheck(dgraph),err,ret);

    SCOTCH_CALL_GOTO(SCOTCH_stratInit(strat),err,ret);
    
    /* use strategy string from traits */
    SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrder(strat,mat->traits->scotchStrat),err,ret);

    /* or use some default strategy */
    /* \todo: I'n not sure what the 'balrat' value does (last param),
       I am assuming: allow at most 20% load imbalance (balrat=0.2))
     */
//    SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrderBuild(strat, SCOTCH_STRATSCALABILITY, 
//        (ghost_gidx_t)nprocs,0,0.1),err,ret);

    dorder = SCOTCH_dorderAlloc();
    if (!dorder) {
        ERROR_LOG("Could not alloc SCOTCH order");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderInit(dgraph,dorder),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderCompute(dgraph,dorder,strat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderPerm(dgraph,dorder,mat->context->permutation->perm+mat->context->lfRow[me]),err,ret);
    GHOST_INSTR_STOP("scotch_createperm")
    

    GHOST_INSTR_START("scotch_combineperm")
    // combine permutation vectors
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,mat->context->permutation->perm,mat->context->gnrows,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);

    // assemble inverse permutation
    for (i=0; i<mat->context->gnrows; i++) {
        mat->context->permutation->invPerm[mat->context->permutation->perm[i]] = i;
    }
    GHOST_INSTR_STOP("scotch_combineperm")
    

#else

    ghost_malloc((void **)&col_loopless,nnz*sizeof(ghost_gidx_t));
    ghost_malloc((void **)&rpt_loopless,(mat->nrows+1)*sizeof(ghost_gidx_t));
    rpt_loopless[0] = 0;
    ghost_gidx_t nnz_loopless = 0;
    ghost_gidx_t j;

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
    SCOTCH_CALL_GOTO(SCOTCH_graphBuild(graph, 0, (ghost_gidx_t)mat->nrows, rpt_loopless, rpt_loopless+1, NULL, NULL, nnz_loopless, col_loopless, NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphCheck(graph),err,ret);
    
    order = SCOTCH_orderAlloc();
    if (!order) {
        ERROR_LOG("Could not alloc SCOTCH order");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderInit(graph,order,mat->context->permutation->perm,NULL,NULL,NULL,NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_stratGraphOrder(strat,mat->traits->scotchStrat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderCompute(graph,order,strat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderCheck(graph,order),err,ret);

    for (i=0; i<mat->nrows; i++) {
        mat->context->permutation->invPerm[mat->context->permutation->perm[i]] = i;
    }

#endif
    
    if (mat->traits->sortScope > 1) {
        GHOST_INSTR_START("post-permutation with sorting")
        ghost_gidx_t nrows = mat->context->gnrows;
        ghost_gidx_t scope = mat->traits->sortScope;
        
        GHOST_CALL_GOTO(ghost_malloc((void **)&rowSort,nrows * sizeof(ghost_sorting_helper_t)),err,ret);

        free(rpt);
        GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(nrows+1)*sizeof(ghost_gidx_t)),err,ret);
        memset(rpt,0,(nrows+1)*sizeof(ghost_gidx_t));
        
        if (srcType == GHOST_SPARSEMAT_SRC_FILE) {
            char *matrixPath = (char *)matrixSource;
            GHOST_CALL_GOTO(ghost_bincrs_rpt_read(rpt, matrixPath, 0, nrows+1, NULL),err,ret);
            for (i=0; i<nrows; i++) {
                rowSort[mat->context->permutation->perm[i]].row = i;
                rowSort[mat->context->permutation->perm[i]].nEntsInRow = rpt[i+1]-rpt[i];
            
            }
        
        } else if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
            ghost_sparsemat_src_rowfunc_t *src = (ghost_sparsemat_src_rowfunc_t *)matrixSource;
            ghost_gidx_t *dummycol;
            char *dummyval;
            GHOST_CALL_GOTO(ghost_malloc((void **)&dummycol,src->maxrowlen*sizeof(ghost_gidx_t)),err,ret);
            GHOST_CALL_GOTO(ghost_malloc((void **)&dummyval,src->maxrowlen*mat->elSize),err,ret);

            for (i=0; i<nrows; i++) {
                rowSort[mat->context->permutation->perm[i]].row = i;
                src->func(i,&rowSort[mat->context->permutation->perm[i]].nEntsInRow,dummycol,dummyval);
            }
            
            free(dummyval);
            free(dummycol);
        }
       
        for (c=0; c<nrows/scope; c++) {
            qsort(rowSort+c*scope, scope, sizeof(ghost_sorting_helper_t), ghost_cmp_entsperrow);
        }
        qsort(rowSort+c*scope, nrows-c*scope, sizeof(ghost_sorting_helper_t), ghost_cmp_entsperrow);
        
        for(i=0; i < nrows; ++i) {
            (mat->context->permutation->invPerm)[i] =rowSort[i].row;
            (mat->context->permutation->perm)[rowSort[i].row] = i;
        }
        GHOST_INSTR_STOP("post-permutation with sorting")
    }
#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat->context->permutation->cu_perm,mat->context->permutation->perm,mat->context->permutation->len*sizeof(ghost_gidx_t));
#endif
    goto out;
err:
    ERROR_LOG("Deleting permutations");
    free(mat->context->permutation->perm); mat->context->permutation->perm = NULL;
    free(mat->context->permutation->invPerm); mat->context->permutation->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
    ghost_cu_free(mat->context->permutation->cu_perm); mat->context->permutation->cu_perm = NULL;
#endif
    free(mat->context->permutation); mat->context->permutation = NULL;

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
    GHOST_INSTR_STOP("scotch")
    
    
    return ret;

#endif
}

