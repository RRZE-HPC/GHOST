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

#if defined(GHOST_HAVE_SCOTCH) && defined(GHOST_HAVE_MPI)
typedef struct {
    ghost_gidx_t idx, pidx;
} ghost_permutation_ent_t;
                    
static int perm_ent_cmp(const void *a, const void *b)
{
    return ((ghost_permutation_ent_t *)a)->pidx - ((ghost_permutation_ent_t *)b)->pidx;
}
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
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_PREPROCESS);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_gidx_t *col = NULL, i, j, k;
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

    if (mat->context->perm_global) {
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
#if 0
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

#endif
    if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
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
                src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,NULL);
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
                    src->func(mat->context->lfRow[me]+i,&rowlen,&col[rpt[i]],tmpval,NULL);
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

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_global,sizeof(ghost_permutation_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_global->perm,sizeof(ghost_gidx_t)*mat->context->lnrows[me]),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_global->invPerm,sizeof(ghost_gidx_t)*mat->context->lnrows[me]),err,ret);
#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->context->perm_global->cu_perm,sizeof(ghost_gidx_t)*mat->context->lnrows[me]),err,ret);
#endif
    memset(mat->context->perm_global->perm,0,sizeof(ghost_gidx_t)*mat->context->lnrows[me]);
    memset(mat->context->perm_global->invPerm,0,sizeof(ghost_gidx_t)*mat->context->lnrows[me]);
    mat->context->perm_global->scope = GHOST_PERMUTATION_GLOBAL;
    mat->context->perm_global->len = mat->context->lnrows[me];

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
//    SCOTCH_CALL_GOTO(SCOTCH_dgraphCheck(dgraph),err,ret);

    SCOTCH_CALL_GOTO(SCOTCH_stratInit(strat),err,ret);
    
    /* use strategy string from traits */
    SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrder(strat,mat->traits->scotchStrat),err,ret);

    /* or use some default strategy */
    /* \todo: I'm not sure what the 'balrat' value does (last param),
       I am assuming: allow at most 20% load imbalance (balrat=0.2))
     */
//     int flags=SCOTCH_STRATSPEED|
//              SCOTCH_STRATSCALABILITY|
//              SCOTCH_STRATBALANCE;
//    SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrderBuild(strat, flags, 
//        (ghost_gidx_t)nprocs,0,0.1),err,ret);
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
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderPerm(dgraph,dorder,mat->context->perm_global->perm),err,ret);
    GHOST_INSTR_STOP("scotch_createperm")
    

    GHOST_INSTR_START("scotch_combineperm")

    ghost_mpi_datatype_t ghost_mpi_dt_perm;
    MPI_CALL_RETURN(MPI_Type_contiguous(2,ghost_mpi_dt_gidx,&ghost_mpi_dt_perm));
    MPI_CALL_RETURN(MPI_Type_commit(&ghost_mpi_dt_perm));

    int proc;
    ghost_permutation_ent_t *permclone;
    ghost_malloc((void **)&permclone,sizeof(ghost_permutation_ent_t)*mat->context->lnrows[me]);

#pragma omp parallel for
    for (i=0; i<mat->context->lnrows[me]; i++) {
        permclone[i].idx = mat->context->lfRow[me]+i;
        permclone[i].pidx = mat->context->perm_global->perm[i];
    }
    qsort(permclone,mat->context->lnrows[me],sizeof(ghost_permutation_ent_t),perm_ent_cmp);
    // permclone is now sorted by ascending pidx

    ghost_lidx_t offs = 0;
    for (proc = 0; proc<nprocs; proc++) {
        int displ[nprocs];
        int nel[nprocs];
        int recvdispl[nprocs];
        memset(displ,0,sizeof(displ));
        memset(nel,0,sizeof(nel));

        // find 1st pidx in sorted permclone which lies in process proc
        while((offs < mat->context->lnrows[me]) && (permclone[offs].pidx < mat->context->lfRow[proc])) {
            offs++;
        }
        displ[me] = offs;
        
        // find last pidx in sorted permclone which lies in process proc
        while((offs < mat->context->lnrows[me]) && (permclone[offs].pidx < mat->context->lfRow[proc]+mat->context->lnrows[proc])) {
            offs++;
        }
        nel[me] = offs-displ[me];

        // proc needs to know how many elements to receive from each process
        if (proc == me) { 
            MPI_Reduce(MPI_IN_PLACE,nel,nprocs,MPI_INT,MPI_MAX,proc,mat->context->mpicomm);
        } else {
            MPI_Reduce(nel,NULL,nprocs,MPI_INT,MPI_MAX,proc,mat->context->mpicomm);
        }

        // assemble receive displacements
        if (proc == me) {
            recvdispl[0] = 0;
            for (i=1; i<nprocs; i++) {
                recvdispl[i] = recvdispl[i-1] + nel[i-1];
            }
            
        }

        // prepare receive buffer
        ghost_permutation_ent_t *recvbuf = NULL;
        if (proc == me) {
            ghost_malloc((void **)&recvbuf,mat->context->lnrows[me]*sizeof(ghost_permutation_ent_t));
        }

        // gather local invPerm
        MPI_Gatherv(&permclone[displ[me]],nel[me],ghost_mpi_dt_perm,recvbuf,nel,recvdispl,ghost_mpi_dt_perm,proc,mat->context->mpicomm);
        
        if (proc == me) {
            // sort the indices and put them into the invPerm array
            qsort(recvbuf,mat->context->lnrows[me],sizeof(ghost_permutation_ent_t),perm_ent_cmp);
            for (i=0; i<mat->context->lnrows[me]; i++) {
                mat->context->perm_global->invPerm[i] = recvbuf[i].idx;
            }
        }

        if (proc == me) {
            free(recvbuf);
        }
    }

    free(permclone);
        
    MPI_CALL_RETURN(MPI_Type_free(&ghost_mpi_dt_perm));

    GHOST_INSTR_STOP("scotch_combineperm")
    

#else

    ghost_malloc((void **)&col_loopless,nnz*sizeof(ghost_gidx_t));
    ghost_malloc((void **)&rpt_loopless,(mat->nrows+1)*sizeof(ghost_gidx_t));
    rpt_loopless[0] = 0;
    ghost_gidx_t nnz_loopless = 0;

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
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderInit(graph,order,mat->context->perm_global->perm,NULL,NULL,NULL,NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_stratGraphOrder(strat,mat->traits->scotchStrat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderCompute(graph,order,strat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_graphOrderCheck(graph,order),err,ret);

    for (i=0; i<mat->nrows; i++) {
        mat->context->perm_global->invPerm[mat->context->perm_global->perm[i]] = i;
    }

#endif
   
#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat->context->perm_global->cu_perm,mat->context->perm_global->perm,mat->context->perm_global->len*sizeof(ghost_gidx_t));
#endif
    goto out;
err:
    ERROR_LOG("Deleting permutations");
    free(mat->context->perm_global->perm); mat->context->perm_global->perm = NULL;
    free(mat->context->perm_global->invPerm); mat->context->perm_global->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
    ghost_cu_free(mat->context->perm_global->cu_perm); mat->context->perm_global->cu_perm = NULL;
#endif
    free(mat->context->perm_global); mat->context->perm_global = NULL;

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
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_PREPROCESS);
    
    
    return ret;

#endif
}

