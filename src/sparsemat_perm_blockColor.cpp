#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#include "ghost/omp.h"
#include <omp.h>

extern "C" ghost_error ghost_sparsemat_blockColor(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType) 
{
   INFO_LOG("Create partition and permute (Block coloring)");
    ghost_error ret = GHOST_SUCCESS;
    ghost_lidx *curcol = NULL; 

    int *nthread = new int[1];  
#ifdef GHOST_HAVE_OPENMP
 #pragma omp parallel
   {
     #pragma omp master
     nthread[0] = ghost_omp_nthread();
   }
#else
    nthread[0] = 1;
#endif
   
   int n_zones = nthread[0];
 
   int me;

printf("1\n");

   GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);

   ghost_lidx nrows = mat->context->lnrows[me];

  //  ghost_lidx *row_ptr = mat->sell->chunkStart;
  //  ghost_lidx *col_ptr = mat->sell->col;
  //  ghost_lidx nrows = mat->nrows;
   
 //   ghost_lidx n_t_zones = n_zones-1; 
    //approximate
    ghost_lidx local_size = static_cast<int>(nrows / n_zones);

    int *rhs_split;
    rhs_split = new int[n_zones+1];
   
    for(int i=0; i<n_zones; ++i) {
      rhs_split[i] = i*local_size;
    }
 
    rhs_split[n_zones] = nrows ;

    ghost_lidx *zone = new ghost_lidx[nrows]; 
 
    for(int i=0; i<nrows; ++i) {
        zone[i] = -(n_zones+1) ; //an invalid number
    }

printf("2\n");
   
   if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
       ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
       ghost_gidx * tmpcol = NULL;
       char * tmpval = NULL;

       ghost_lidx rowlen;

#pragma omp parallel private(tmpval,tmpcol,rowlen) 
      {
       ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
       ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
    
  #pragma omp for
       for (int i=0; i<mat->context->lnrows[me]; i++) {
		src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,NULL); 
          	for(int k=0; k<n_zones; ++k) {
                   //pure zone
                   if(tmpcol[0] >= rhs_split[k] && tmpcol[rowlen-1] <= rhs_split[k+1]) {
                      zone[i] = 2*k;
                    }
                   //transition zone
                   else if(k>0 && zone[i]<0 && (tmpcol[0] >= rhs_split[k-1] && tmpcol[rowlen-1] <= rhs_split[k+1]) ) {
                      zone[i] = 2*k-1;
                   }
                }
        }
      free(tmpcol);
      free(tmpval);
     }
   }

printf("3\n");             
/*       for(int i=0; i<nrows; ++i){
    	    for(int k=0; k<n_zones; ++k){
        	    //pure zone
            	if(col_ptr[row_ptr[i]] >= rhs_split[k] && col_ptr[row_ptr[i+1]-1] <= rhs_split[k+1]) {
            	    zone[i] = 2*k;
           	 }
            	//transition zone
            	else if(k>0 && zone[i]<0 && (col_ptr[row_ptr[i]] >= rhs_split[k-1] && col_ptr[row_ptr[i+1]-1] <= rhs_split[k+1]) ){
                    zone[i] = 2*k-1;                  
            	}
       	    }
    	}
*/

if (!mat->context->perm_local) {
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local,sizeof(ghost_permutation)), err, ret);
    mat->context->perm_local->scope = GHOST_PERMUTATION_LOCAL;
    mat->context->perm_local->len = mat->nrows;
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->perm,sizeof(ghost_gidx)*mat->nrows), err, ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->invPerm,sizeof(ghost_gidx)*mat->nrows), err, ret);
}
 
    mat->nzones = 2*n_zones-1 ;
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->zone_ptr,(mat->nzones+1)*sizeof(ghost_lidx)), err, ret);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(mat->nzones+1)*sizeof(ghost_lidx)), err, ret);
    memset(curcol,0,(mat->nzones+1)*sizeof(ghost_lidx));


printf("4\n");

    for (int i=0;i<mat->nzones+1;i++) {
        mat->zone_ptr[i] = 0;
    }
 
    for (int i=0;i<mat->nrows;i++) {
        mat->zone_ptr[zone[i]+1]++;
    }

    for (int i=1;i<mat->nzones+1;i++) {
        mat->zone_ptr[i] += mat->zone_ptr[i-1];
    }

if(mat->context->perm_local)
    for (int i=0;i<mat->nrows;i++) {
        mat->context->perm_local->perm[i] = mat->context->perm_local->invPerm[zone[i]] + mat->zone_ptr[zone[i]];
        curcol[zone[i]]++;
    }

else   
    for (int i=0;i<mat->nrows;i++) {
        mat->context->perm_local->perm[i] = curcol[zone[i]] + mat->zone_ptr[zone[i]];
        curcol[zone[i]]++;
    }

  
    for (int i=0;i<mat->nrows;i++) {
        mat->context->perm_local->invPerm[mat->context->perm_local->perm[i]] = i;
    }
printf("5\n");  
   goto out;

err:

out:
 free(curcol);
 return ret;
}


