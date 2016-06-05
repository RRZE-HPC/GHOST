#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
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

ghost_error ghost_sparsemat_perm_spmp(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType)
{
#if !defined(GHOST_HAVE_SPMP)
    UNUSED(mat);
    UNUSED(matrixSource);
    UNUSED(srcType);
    WARNING_LOG("SpMP not available. Will not create matrix permutation!");
    return GHOST_SUCCESS;
#else

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    ghost_error ret = GHOST_SUCCESS;

    ghost_lidx i,j;
    int me;
    ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
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
    ghost_lidx ncols_halo_padded = mat->ncols;    

#ifdef NONSYM_RCM_MIRROR
    int *symcol = NULL, *symrpt = NULL;
    coo_ent *syments = NULL;
    double *symval = NULL;
    int syment = 0;
    int symnnz = 0;
#else 
    int *intcolperm = NULL, *intcolinvperm = NULL;
    SpMP::CSR *csrT = NULL;
    SpMP::CSR *csrTT = NULL; //delete after transpose checking
#endif
    int localent = 0;
    
    if (srcType != GHOST_SPARSEMAT_SRC_FUNC) {
        ERROR_LOG("Only function sparse matrix source allowed!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(mat->nrows+1)*sizeof(int)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localrpt,(mat->nrows+1)*sizeof(int)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local,sizeof(ghost_permutation)),err,ret);
    mat->context->perm_local->scope = GHOST_PERMUTATION_LOCAL;
    mat->context->perm_local->len = mat->nrows;
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->perm,sizeof(ghost_gidx)*mat->nrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->invPerm,sizeof(ghost_gidx)*mat->nrows),err,ret);
    mat->context->perm_local->colPerm = NULL;
    mat->context->perm_local->colInvPerm = NULL; 
    mat->context->perm_local->method = GHOST_PERMUTATION_SYMMETRIC ;

#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->context->perm_local->cu_perm,sizeof(ghost_gidx)*mat->nrows),err,ret);
#endif

    rpt[0] = 0;
    localrpt[0] = 0;

   
#pragma omp parallel private (tmpval,tmpcol,i,rowlen) reduction(+:nnz)
    {
        ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
        ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));

        if (mat->context->perm_global) {
#pragma omp for
            for (i=0; i<mat->context->lnrows[me]; i++) {
                src->func(mat->context->perm_global->invPerm[i],&rowlen,tmpcol,tmpval,src->arg);
                nnz += rowlen;
            }
        } else {
           for (i=0; i<mat->context->lnrows[me]; i++) {
                src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,src->arg);
                nnz += rowlen;
            }
        }


        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
   
 
    GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz*sizeof(ghost_gidx)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&localcol,nnz*sizeof(int)),err,ret);

    ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
    for (i=0; i<mat->context->lnrows[me]; i++) {
        if (mat->context->perm_global) {
            src->func(mat->context->perm_global->invPerm[i],&rowlen,&col[rpt[i]],tmpval,src->arg);
        } else {
            src->func(mat->context->lfRow[me]+i,&rowlen,&col[rpt[i]],tmpval,src->arg);
        } 
        rpt[i+1] = rpt[i] + rowlen;
       
	if (mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
        	for (j=rpt[i]; j<rpt[i+1]; j++) {
                		localcol[localent] = col[j] ;
                                localent++;
        	}
        	localrpt[i+1] = localent;
    	} else {
               for (j=rpt[i]; j<rpt[i+1]; j++) {
                        if (col[j] >= mat->context->lfRow[me] && col[j] < (mat->context->lfRow[me]+mat->context->lnrows[me])) {
                                localcol[localent] = col[j] - mat->context->lfRow[me];
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

    if (mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
     	ncols_halo_padded = mat->context->nrowspadded + mat->context->halo_elements+1;
    }

    csr = new SpMP::CSR(mat->nrows,ncols_halo_padded,localrpt,localcol,val);
      
    GHOST_CALL_GOTO(ghost_malloc((void **)&intperm,sizeof(int)*mat->nrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&intinvperm,sizeof(int)*mat->nrows),err,ret);
 
    if (csr->isSymmetric(false,false)) { 
        csr->getRCMPermutation(intperm, intinvperm);
        
        useperm = intperm;
        useinvperm = intinvperm;
    } else {
#ifdef NONSYM_RCM_MIRROR

        WARNING_LOG("The local matrix is not symmetric! RCM will be done based on the mirrored upper triangular matrix!");
        
        GHOST_CALL_GOTO(ghost_malloc((void **)&symrpt,sizeof(int)*(mat->nrows+1)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&syments,sizeof(coo_ent)*(localnnz*2)),err,ret);

        for (i=0; i<localnnz*2; i++) {
            syments[i].row=INT_MAX;
            syments[i].col=INT_MAX;
        }
        
        symrpt[0] = 0;
        for (i=0; i<mat->nrows; i++) {
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
        for (i=0; i<mat->nrows; i++) {
            symrpt[i+1] += symrpt[i];
        }
        GHOST_CALL_GOTO(ghost_malloc((void **)&symval,sizeof(double)*symnnz),err,ret);
        memset(symval,0,sizeof(double)*symnnz);

        delete csr;
        csr = new SpMP::CSR(mat->nrows,mat->nrows,symrpt,symcol,symval);
        csr->getRCMPermutation(intperm, intinvperm);
        
        useperm = intperm;
        useinvperm = intinvperm;

#else


       INFO_LOG("Doing BFS Bipartite instead of RCM as the matrix is not symmetric.");    
       
      csrT = csr->transpose();
  /*      csrTT = csrT->transpose();

        INFO_LOG("Checking TRANSPOSE");

        for(int i=0; i<mat->nrows; ++i) {
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
        int m = mat->nrows;
        int n = ncols_halo_padded;//mat->ncols;

 
/*        for(int i=0; i<n; ++i) {
 		csrT->rowptr[i] = csr      
 
        bfs_matrix = new SpMP::CSR(mat->nrows+mat->ncols,mat->nrows+mat->ncols,localrpt,localcol,val);
*/
        GHOST_CALL_GOTO(ghost_malloc((void **)&intcolperm,sizeof(int)*ncols_halo_padded),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&intcolinvperm,sizeof(int)*ncols_halo_padded),err,ret);

        bfsBipartite(*csr, *csrT, intperm, intinvperm, intcolperm, intcolinvperm);


/*	if(me==0)
		csr->storeMatrixMarket("after_bfs_0");
        if(me==1)
		csr->storeMatrixMarket("after_bfs_1");
*/
        useperm = intperm;
        useinvperm = intinvperm; 

        mat->context->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC; 
	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colPerm,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colInvPerm,sizeof(ghost_gidx)*ncols_halo_padded),err,ret);

 
    /*    printf("Row perm\n");
	 for(int i=0; i<mat->nrows;++i) {
 		printf("%d\n",intperm[i]);
         }
       printf("Row inv perm\n");
	 for(int i=0; i<mat->nrows;++i) {
 		printf("%d\n",intinvperm[i]);
 	}
        printf("Col perm\n");
	 for(int i=0; i<mat->ncols;++i) {
 		printf("%d\n",intcolperm[i]);
         }
       printf("Col inv perm\n");
	 for(int i=0; i<mat->ncols;++i) {
 		printf("%d\n",intcolinvperm[i]);
 	}
      */          

	#pragma omp parallel for
    	for (i=0; i<ncols_halo_padded; i++) {
        	mat->context->perm_local->colPerm[i] = intcolperm[i];
        	mat->context->perm_local->colInvPerm[i] = intcolinvperm[i];
    	}

#endif

    }
   
    INFO_LOG("Original bandwidth, avg. width: %d, %g",csr->getBandwidth(),csr->getAverageWidth());

    if(mat->context->perm_local->colPerm == NULL) {
    	csrperm = csr->permute(useperm,useinvperm);
    }
    else {
        csrperm = csr->permute(intcolperm ,useinvperm);
    }

    INFO_LOG("Permuted bandwidth, avg. width: %d, %g",csrperm->getBandwidth(),csrperm->getAverageWidth());
  
MPI_Barrier(mat->context->mpicomm);  
#pragma omp parallel for
    for (i=0; i<mat->nrows; i++) {
        mat->context->perm_local->perm[i] = useperm[i];
        mat->context->perm_local->invPerm[i] = useinvperm[i];
    }

#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat->context->perm_local->cu_perm,mat->context->perm_local->perm,mat->context->perm_local->len*sizeof(ghost_gidx));
#endif


goto out;

err:
    ERROR_LOG("Deleting permutations");
    free(mat->context->perm_local->perm); mat->context->perm_local->perm = NULL;
    free(mat->context->perm_local->invPerm); mat->context->perm_local->invPerm = NULL;

    if( mat->context->perm_local->perm != NULL) {
    	free(mat->context->perm_local->colPerm); mat->context->perm_local->colPerm = NULL;

	free(mat->context->perm_local->colInvPerm); mat->context->perm_local->colInvPerm = NULL;
    }

#ifdef GHOST_HAVE_CUDA
    ghost_cu_free(mat->context->perm_local->cu_perm); mat->context->perm_local->cu_perm = NULL;
#endif
    free(mat->context->perm_local); mat->context->perm_local = NULL;

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
