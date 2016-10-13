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
//#define NONSYM_RCM_MIRROR

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

extern "C" ghost_error ghost_sparsemat_perm_spmp(ghost_context *ctx, ghost_sparsemat *mat)
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
    int *rptlocal, *collocal;
    double *val;
    ghost_lidx nnz = SPM_NNZ(mat);
    int *intperm = NULL, *intinvperm = NULL;
    int *useperm = NULL, *useinvperm = NULL;
    int nnzlocal = 0;
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

    ghost_lidx ncols_halo_padded = mat->context->row_map->dim;
    if (mat->traits.flags & GHOST_PERM_NO_DISTINCTION) {
        ncols_halo_padded = mat->context->col_map->dim;
    }
    ERROR_LOG("ncolshalopadded = %d row_map->dim %d halo %d",ncols_halo_padded,mat->context->row_map->dim,mat->context->halo_elements);

#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)ctx->perm_local->cu_perm,sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);
#endif

    GHOST_CALL_GOTO(ghost_malloc((void **)&rptlocal,(ctx->row_map->dim+1)*sizeof(int)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&collocal,nnz * sizeof(ghost_lidx)),err,ret);
    
    rptlocal[0] = 0;
    
    for (i=0; i<ctx->row_map->dim; i++) {
        rptlocal[i+1] = rptlocal[i];
        
        ghost_lidx orig_row = i;
        if (ctx->row_map->loc_perm) {
            orig_row = ctx->row_map->loc_perm_inv[i];
        }
        ghost_lidx * col = &mat->col[mat->chunkStart[orig_row]];
        ghost_lidx orig_row_len = mat->chunkStart[orig_row+1]-mat->chunkStart[orig_row];

        for(j=0; j<orig_row_len; ++j) {
            if ((ctx->flags & GHOST_PERM_NO_DISTINCTION) || (col[j] < mat->context->row_map->dim)) {
                collocal[nnzlocal] = col[j];
                nnzlocal++;
                rptlocal[i+1]++;
            }
        }
    }

    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->row_map->loc_perm,sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->row_map->loc_perm_inv,sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);   

    GHOST_CALL_GOTO(ghost_malloc((void **)&val,sizeof(double)*nnzlocal),err,ret);
    memset(val,0,sizeof(double)*nnzlocal);

    csr = new SpMP::CSR(ctx->row_map->dim,ncols_halo_padded,rptlocal,collocal,val);

    GHOST_CALL_GOTO(ghost_malloc((void **)&intperm,sizeof(int)*ctx->row_map->dim),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&intinvperm,sizeof(int)*ctx->row_map->dim),err,ret);

    if (csr->isSymmetric(false,false)) {
       INFO_LOG("Doing RCM"); 
        csr->getRCMPermutation(intperm, intinvperm);

        useperm = intperm;
        useinvperm = intinvperm;
    } else {
#ifdef NONSYM_RCM_MIRROR

        WARNING_LOG("The local matrix is not symmetric! RCM will be done based on the mirrored upper triangular matrix!");

        GHOST_CALL_GOTO(ghost_malloc((void **)&symrpt,sizeof(int)*(ctx->row_map->dim+1)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&syments,sizeof(coo_ent)*(nnzlocal*2)),err,ret);

        for (i=0; i<nnzlocal*2; i++) {
            syments[i].row=INT_MAX;
            syments[i].col=INT_MAX;
        }

        symrpt[0] = 0;
        for (i=0; i<ctx->row_map->dim; i++) {
            symrpt[i+1] = 0;
            for (j=rptlocal[i]; j<rptlocal[i+1]; j++) {
                if (collocal[j] >= i) {
                    syments[syment].row = i;
                    syments[syment].col = collocal[j];
                    syment++;
                    if (collocal[j] != i) { // non-diagonal: insert sibling
                        syments[syment].row = collocal[j];
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
        for (i=0; i<ctx->row_map->dim; i++) {
            symrpt[i+1] += symrpt[i];
        }
        GHOST_CALL_GOTO(ghost_malloc((void **)&symval,sizeof(double)*symnnz),err,ret);
        memset(symval,0,sizeof(double)*symnnz);

        delete csr;
        csr = new SpMP::CSR(ctx->row_map->dim,ctx->row_map->dim,symrpt,symcol,symval);
        csr->getRCMPermutation(intperm, intinvperm);

        useperm = intperm;
        useinvperm = intinvperm;

#else

        INFO_LOG("Doing BFS Bipartite instead of RCM as the matrix is not symmetric.");         


        int me;
        ghost_rank(&me,MPI_COMM_WORLD);
        	if(me==0)
            csr->storeMatrixMarket("proc0_before_RCM");
            if(me==1)
            csr->storeMatrixMarket("proc1_before_RCM");

        csrT = csr->transpose();
        /*      csrTT = csrT->transpose();

                INFO_LOG("Checking TRANSPOSE");

                for(int i=0; i<ctx->row_map->dim; ++i) {
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
        //int m = ctx->row_map->dim;
        //int n = ncols_halo_padded;


        /*        for(int i=0; i<n; ++i) {
                  csrT->rowptr[i] = csr      

                  bfs_matrix = new SpMP::CSR(ctx->row_map->dim+SPM_NCOLS(mat),ctx->row_map->dim+SPM_NCOLS(mat),rptlocal,collocal,val);
                  */
        GHOST_CALL_GOTO(ghost_malloc((void **)&intcolperm,sizeof(int)*ncols_halo_padded),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&intcolinvperm,sizeof(int)*ncols_halo_padded),err,ret);
        
        bfsBipartite(*csr, *csrT, intperm, intinvperm, intcolperm, intcolinvperm);

        useperm = intperm;
        useinvperm = intinvperm; 

        //ctx->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->col_map->loc_perm,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->col_map->loc_perm_inv,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);


        /*    printf("Row perm\n");
              for(int i=0; i<ctx->row_map->dim;++i) {
              printf("%d\n",intperm[i]);
              }
              printf("Row inv perm\n");
              for(int i=0; i<ctx->row_map->dim;++i) {
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
            ctx->col_map->loc_perm[i] = intcolperm[i];
            ctx->col_map->loc_perm_inv[i] = intcolinvperm[i];
        }

#endif

    }


    if(ctx->col_map->loc_perm == NULL) {
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

    INFO_LOG("BW reduction %d->%d, Avg. width reduction %g->%g",csr->getBandwidth(),csrperm->getBandwidth(),csr->getAverageWidth(),csrperm->getAverageWidth());

#pragma omp parallel for
    for (i=0; i<ctx->row_map->dim; i++) {
        ctx->row_map->loc_perm[i] = useperm[i];
        ctx->row_map->loc_perm_inv[i] = useinvperm[i];
    }
    if (!ctx->col_map->loc_perm) { //symmetric permutation, col perm is still NULL
        ctx->col_map->loc_perm = ctx->row_map->loc_perm;
        ctx->col_map->loc_perm_inv = ctx->row_map->loc_perm_inv;
    }

#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(ctx->row_map->cu_loc_perm,ctx->row_map->loc_perm,ctx->row_map->dim*sizeof(ghost_gidx));
#endif


    goto out;

err:
    ERROR_LOG("Deleting permutations");
    free(ctx->row_map->loc_perm); ctx->row_map->loc_perm = NULL;
    free(ctx->row_map->loc_perm_inv); ctx->row_map->loc_perm_inv = NULL;
    free(ctx->col_map->loc_perm); ctx->col_map->loc_perm = NULL;
    free(ctx->col_map->loc_perm_inv); ctx->col_map->loc_perm_inv = NULL;

#ifdef GHOST_HAVE_CUDA
    ghost_cu_free(ctx->row_map->cu_loc_perm); ctx->row_map->cu_loc_perm = NULL;
#endif

out:
    free(rptlocal);
    free(collocal);
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
