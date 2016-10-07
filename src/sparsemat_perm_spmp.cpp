#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include <iostream>
#ifdef GHOST_HAVE_SPMP
#include "SpMP/CSR.hpp"
#include "SpMP/reordering/BFSBipartite.hpp"
#include "ghost/machine.h"
#include "ghost/constants.h"

// uncomment if RCM with mirrored triangular matrix should be used for non-symmetric matres
// else, bipartite graph BFS is used
// #define NONSYM_RCM_MIRROR

#ifdef NONSYM_RCM_MIRROR
typedef struct
{
    int row;
    int col;
} coo_ent;

static int cmp_coo_ent(const void* a, const void* b) 
{
    return  ((coo_ent *)a)->row - ((coo_ent *)b)->row;
}
#endif

#endif

ghost_error ghost_sparsemat_perm_spmp(ghost_sparsemat *mat_out, ghost_context *ctx, ghost_sparsemat_src_rowfunc *src)
{
#if !defined(GHOST_HAVE_SPMP)
    UNUSED(mat);
    UNUSED(src);
    WARNING_LOG("SpMP not available. Will not create matrix permutation!");
    return GHOST_SUCCESS;
#else

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    ghost_error ret = GHOST_SUCCESS;

    ghost_lidx i,j;
    int *rpt, *localrpt, *localcol;
    ghost_gidx *col;
    double *val;
    ghost_lidx nnz = 0;
    ghost_lidx rowlen;
    char * tmpval = NULL;
    ghost_gidx * tmpcol = NULL;
    int *intperm = NULL, *intinvperm = NULL;
    int *useperm = NULL, *useinvperm = NULL;
    int localnnz = 0;
    SpMP::CSR *csr = NULL, *csrperm = NULL;

#ifdef NONSYM_RCM_MIRROR
    int *symcol = NULL, *symrpt = NULL;
    coo_ent *syments = NULL;
    double *symval = NULL;
    int syment = 0;
    int symnnz = 0;
#else 
    int *intcolperm = NULL, *intcolinvperm = NULL;
    SpMP::CSR *csrT = NULL;
    //SpMP::CSR *csrTT = NULL; //delete after transpose checking
#endif
    int localent = 0;

    ghost_lidx ncols_halo_padded = ctx->row_map->nrows;
    if (ctx->flags & GHOST_PERM_NO_DISTINCTION) {
        ncols_halo_padded = ctx->col_map->nrowspadded;
    }

    ERROR_LOG("ncols_halo_padded = %d",ncols_halo_padded);


    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(ctx->row_map->nrows+1)*sizeof(int)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localrpt,(ctx->row_map->nrows+1)*sizeof(int)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat_out->context->row_map->loc_perm,sizeof(ghost_gidx)*ctx->row_map->nrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat_out->context->row_map->loc_perm_inv,sizeof(ghost_gidx)*ctx->row_map->nrows),err,ret);   
    //mat_out->context->perm_local->method = GHOST_PERMUTATION_SYMMETRIC ;

#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)mat_out->context->perm_local->cu_perm,sizeof(ghost_gidx)*ctx->row_map->nrows),err,ret);
#endif

    rpt[0] = 0;
    localrpt[0] = 0;

#pragma omp parallel private (tmpval,tmpcol,i,rowlen) reduction(+:nnz)
    {
        ghost_malloc((void **)&tmpval,src->maxrowlen*mat_out->elSize);
        ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));

        if (mat_out->context->row_map->glb_perm) {
#pragma omp for
            for (i=0; i<mat_out->context->row_map->nrows; i++) {
                src->func(mat_out->context->row_map->glb_perm_inv[i],&rowlen,tmpcol,tmpval,src->arg);
                nnz += rowlen;
            }
        } else {
#pragma omp for
            for (i=0; i<mat_out->context->row_map->nrows; i++) {
                src->func(mat_out->context->row_map->offs+i,&rowlen,tmpcol,tmpval,src->arg);
                nnz += rowlen;
            }
        }

        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
    GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz*sizeof(ghost_gidx)),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&localcol,nnz*sizeof(int)),err,ret);

    int rank;
    ghost_rank(&rank,MPI_COMM_WORLD);

    ghost_malloc((void **)&tmpval,src->maxrowlen*mat_out->elSize);
    for (i=0; i<mat_out->context->row_map->nrows; i++) {
        if (mat_out->context->flags & GHOST_PERM_NO_DISTINCTION) {
            for (j=rpt[i]; j<rpt[i+1]; j++) {
                localcol[localent] = col[j] ;
                localent++;
            }
            localrpt[i+1] = localent;
        } else {
            if (mat_out->context->row_map->glb_perm) {
                src->func(mat_out->context->row_map->glb_perm_inv[i],&rowlen,&col[rpt[i]],tmpval,src->arg);
            } else {
                src->func(mat_out->context->row_map->offs+i,&rowlen,&col[rpt[i]],tmpval,src->arg);
            } 
            rpt[i+1] = rpt[i] + rowlen;
            for (j=rpt[i]; j<rpt[i+1]; j++) {
                if (col[j] >= mat_out->context->row_map->offs && col[j] < (mat_out->context->row_map->offs+mat_out->context->row_map->nrows)) {
                    localcol[localent] = col[j] - mat_out->context->row_map->offs;
                    localent++;
                }
            }
            localrpt[i+1] = localent;
        }
    }

    free(tmpval); tmpval = NULL;
    localnnz = localent;

    GHOST_CALL_GOTO(ghost_malloc((void **)&val,sizeof(double)*localnnz),err,ret);
    memset(val,0,sizeof(double)*localnnz);

    ERROR_LOG(">> nnz %d %dx%d",localnnz,ctx->row_map->nrows,ncols_halo_padded);

    csr = new SpMP::CSR(ctx->row_map->nrows,ncols_halo_padded,localrpt,localcol,val);

    GHOST_CALL_GOTO(ghost_malloc((void **)&intperm,sizeof(int)*ctx->row_map->nrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&intinvperm,sizeof(int)*ctx->row_map->nrows),err,ret);

    if (csr->isSymmetric(false,false)) { 
        csr->getRCMPermutation(intperm, intinvperm);

        useperm = intperm;
        useinvperm = intinvperm;
    } else {
#ifdef NONSYM_RCM_MIRROR

        WARNING_LOG("The local matrix is not symmetric! RCM will be done based on the mirrored upper triangular matrix!");

        GHOST_CALL_GOTO(ghost_malloc((void **)&symrpt,sizeof(int)*(ctx->row_map->nrows+1)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&syments,sizeof(coo_ent)*(localnnz*2)),err,ret);

        for (i=0; i<localnnz*2; i++) {
            syments[i].row=INT_MAX;
            syments[i].col=INT_MAX;
        }

        symrpt[0] = 0;
        for (i=0; i<ctx->row_map->nrows; i++) {
            symrpt[i+1] = 0;
            for (j=localrpt[i]; j<localrpt[i+1]; j++) {
                if (localcol[j] >= i) {
                    syments[syment].row = i;
                    syments[syment].col = localcol[j];
                    syment++;
                    if (localcol[j] != i) { // non-diagonal: insert sibling
                        syments[syment].row = localcol[j];
                        syments[syment].col = i;
                        syment++;
                    }
                }
            }
        }
        symnnz = syment;

        qsort(syments,symnnz,sizeof(coo_ent),cmp_coo_ent);

        GHOST_CALL_GOTO(ghost_malloc((void **)&symcol,sizeof(int)*(symnnz)),err,ret);

        syment = 0; 
        for (i=0; i<symnnz; i++) {

            symrpt[syments[i].row+1]++;
            symcol[i] = syments[i].col;

        }
        for (i=0; i<ctx->row_map->nrows; i++) {
            symrpt[i+1] += symrpt[i];
        }
        GHOST_CALL_GOTO(ghost_malloc((void **)&symval,sizeof(double)*symnnz),err,ret);
        memset(symval,0,sizeof(double)*symnnz);

        delete csr;
        csr = new SpMP::CSR(ctx->row_map->nrows,ctx->row_map->nrows,symrpt,symcol,symval);
        csr->getRCMPermutation(intperm, intinvperm);

        useperm = intperm;
        useinvperm = intinvperm;

#else

        INFO_LOG("Doing BFS Bipartite instead of RCM as the matrix is not symmetric.");         

        csrT = csr->transpose();
        /*      csrTT = csrT->transpose();

                INFO_LOG("Checking TRANSPOSE");

                for(int i=0; i<ctx->row_map->nrows; ++i) {
                if(csr->rowptr[i] != csrTT->rowptr[i]) {
                ERROR_LOG("FAILED at %d row , csr_rowptr =%d and csrTT_rowptr =%d",i,csr->rowptr[i], csrTT->rowptr[i]);
                }
                for(int j=csr->rowptr[i]; j<csr->rowptr[i+1]; ++j) {
                if(csr->colidx[j] != csrTT->colidx[j]) {
                ERROR_LOG("FAILED at inner: column csr_colidx =%d and csrTT_colidx=%d",i,csr->colidx[j],csrTT->colidx[j]);
                }
                if(csr->values[j] != csrTT->values[j]) {
                ERROR_LOG("FAILED at inner: value csr_values =%f and csrTT_values=%f",i,csr->values[j],csrTT->values[j]);
                }
                }
                }


                INFO_LOG("TRANSPOSE check finished");         
                */
        //int m = ctx->row_map->nrows;
        //int n = ncols_halo_padded;


        /*        for(int i=0; i<n; ++i) {
                  csrT->rowptr[i] = csr      

                  bfs_matrix = new SpMP::CSR(ctx->row_map->nrows+SPM_NCOLS(mat),ctx->row_map->nrows+SPM_NCOLS(mat),localrpt,localcol,val);
                  */
        GHOST_CALL_GOTO(ghost_malloc((void **)&intcolperm,sizeof(int)*ncols_halo_padded),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&intcolinvperm,sizeof(int)*ncols_halo_padded),err,ret);

        bfsBipartite(*csr, *csrT, intperm, intinvperm, intcolperm, intcolinvperm);

        /*	if(me==0)
            csr->storeMatrixMarket("proc0_before_RCM");
            if(me==1)
            csr->storeMatrixMarket("proc1_before_RCM");
            */
        useperm = intperm;
        useinvperm = intinvperm; 

        //mat_out->context->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;
        ERROR_LOG("alloc col_perm with %d entries",ncols_halo_padded); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat_out->context->col_map->loc_perm,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat_out->context->col_map->loc_perm_inv,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);


        /*    printf("Row perm\n");
              for(int i=0; i<ctx->row_map->nrows;++i) {
              printf("%d\n",intperm[i]);
              }
              printf("Row inv perm\n");
              for(int i=0; i<ctx->row_map->nrows;++i) {
              printf("%d\n",intinvperm[i]);
              }
              printf("Col perm\n");
              for(int i=0; i<SPM_NCOLS(mat);++i) {
              printf("%d\n",intcolperm[i]);
              }
              printf("Col inv perm\n");
              for(int i=0; i<SPM_NCOLS(mat);++i) {
              printf("%d\n",intcolinvperm[i]);
              }
              */          

#pragma omp parallel for
        for (i=0; i<ncols_halo_padded; i++) {
            mat_out->context->col_map->loc_perm[i] = intcolperm[i];
            mat_out->context->col_map->loc_perm_inv[i] = intcolinvperm[i];
        }

#endif

    }

    INFO_LOG("Original bandwidth, avg. width: %d, %g",csr->getBandwidth(),csr->getAverageWidth());

    if(mat_out->context->col_map->loc_perm == NULL) {
        csrperm = csr->permute(useperm,useinvperm);
    }
    else {
        csrperm = csr->permute(intcolperm ,useinvperm);
        /* 	if(me==0)
            csrperm->storeMatrixMarket("proc0_after_RCM");
            if(me==1)
            csrperm->storeMatrixMarket("proc1_after_RCM");
            */
    }

    INFO_LOG("Permuted bandwidth, avg. width: %d, %g",csrperm->getBandwidth(),csrperm->getAverageWidth());


#pragma omp parallel for
    for (i=0; i<ctx->row_map->nrows; i++) {
        mat_out->context->row_map->loc_perm[i] = useperm[i];
        mat_out->context->row_map->loc_perm_inv[i] = useinvperm[i];
    }
    if (!mat_out->context->col_map->loc_perm) { //symmetric permutation, col perm is still NULL
        mat_out->context->col_map->loc_perm = mat_out->context->row_map->loc_perm;
        mat_out->context->col_map->loc_perm_inv = mat_out->context->row_map->loc_perm_inv;
    }

#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat_out->context->row_map->cu_loc_perm,mat_out->context->row_map->loc_perm,ctx->row_map->nrows*sizeof(ghost_gidx));
#endif


    goto out;

err:
    ERROR_LOG("Deleting permutations");
    free(mat_out->context->row_map->loc_perm); mat_out->context->row_map->loc_perm = NULL;
    free(mat_out->context->row_map->loc_perm_inv); mat_out->context->row_map->loc_perm_inv = NULL;
    free(mat_out->context->col_map->loc_perm); mat_out->context->col_map->loc_perm = NULL;
    free(mat_out->context->col_map->loc_perm_inv); mat_out->context->col_map->loc_perm_inv = NULL;

#ifdef GHOST_HAVE_CUDA
    ghost_cu_free(mat_out->context->row_map->cu_loc_perm); mat_out->context->row_map->cu_loc_perm = NULL;
#endif

out:
    free(rpt);
    free(localrpt);
    free(col);
    free(localcol);
    free(val);
    free(intperm);
    free(intinvperm);
#ifdef NONSYM_RCM_MIRROR
    free(syments);
    free(symrpt);
    free(symcol);
    free(symval);
#else
    free(intcolperm);  
    free(intcolinvperm); 
    delete csrT;
#endif
    delete csr;
    delete csrperm;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);

    return ret;
#endif
    }
