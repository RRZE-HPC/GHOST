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
   mat->kacz_setting.active_threads = n_zones;

   printf("...NZONES... = %d\n", n_zones);
   int me;


   GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);

   ghost_lidx nrows = mat->context->lnrows[me];


  //  ghost_lidx *row_ptr = mat->sell->chunkStart;
  //  ghost_lidx *col_ptr = mat->sell->col;
  //  ghost_lidx nrows = mat->nrows;
   

  //  ghost_lidx n_t_zones = n_zones-1; 
  //
    int lower_bw = 0;
    int upper_bw = 0;
    int max_col_idx = 0;

    if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
       ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
       ghost_gidx * tmpcol = NULL;
       char * tmpval = NULL;

       ghost_lidx rowlen;

      
#pragma omp parallel private(tmpval,tmpcol,rowlen) 
  {
       ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
       ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize); 
int ctr = 0;
  #pragma omp parallel for reduction(max:lower_bw) reduction(max:upper_bw) reduction(+:ctr)
       for (int i=0; i<mat->context->lnrows[me]; i++) {
        	if (mat->context->perm_global && mat->context->perm_local) {
                 	src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,tmpcol,tmpval,NULL);
              	} else if (mat->context->perm_global) {
                	src->func(mat->context->perm_global->invPerm[i],&rowlen,tmpcol,tmpval,NULL);
            	} else if (mat->context->perm_local) {
                 	src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[i],&rowlen,tmpcol,tmpval,NULL);
                } else {
                	src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,NULL);
                }

                int start_col = mat->ncols + mat->nrows;
                int end_col   = 0;

               if(mat->context->perm_local->colPerm == NULL) {
                        for(int j=0; j<rowlen; ++j) {
                                start_col = MIN(start_col, mat->context->perm_local->perm[tmpcol[j]]);
                                end_col   = MAX(end_col, mat->context->perm_local->perm[tmpcol[j]]);
                       }
                } else {
                        for(int j=0; j<rowlen; ++j) {
                                start_col = MIN(start_col, mat->context->perm_local->colPerm[tmpcol[j]]);
                                end_col   = MAX(end_col, mat->context->perm_local->colPerm[tmpcol[j]]);
			}
                }

                lower_bw = MAX(lower_bw, i-start_col);
                upper_bw = MAX(upper_bw , end_col - i);
                max_col_idx = MAX(max_col_idx, end_col);
        }

    }
  }
  //std::cout<<"nrows ="<<mat->nrows<<std::endl;
  //std::cout<<"check"<<row_ptr[mat->nrows-1]<<std::endl;

    int total_bw = lower_bw + upper_bw;
    printf("bw lower =%d, upper=%d, total=%d\n",lower_bw,upper_bw,total_bw);

    //approximate
    ghost_lidx local_size = static_cast<int>( ((max_col_idx+1)-0.2*total_bw) / n_zones); //will have to find a method to balance the load for the last thread


    int *rhs_split;
    rhs_split = new int[n_zones+2];
   
    for(int i=0; i<n_zones+1; ++i) {
      rhs_split[i] = i*local_size;
      printf("%d\n",(int)rhs_split[i]);
    }
 
    rhs_split[n_zones+1] = max_col_idx+1;
    printf("%d\n",(int)rhs_split[n_zones+1]);

     

    ghost_lidx *zone = new ghost_lidx[nrows]; 
 
    for(int i=0; i<nrows; ++i) {
        zone[i] = -(n_zones+1) ; //an invalid number
    }

 
   //a counter for number of rows to be multicolored
   ghost_lidx ctr_MC = 0;
   
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
        	if (mat->context->perm_global && mat->context->perm_local) {
                 	src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,tmpcol,tmpval,NULL);
              	} else if (mat->context->perm_global) {
                	src->func(mat->context->perm_global->invPerm[i],&rowlen,tmpcol,tmpval,NULL);
            	} else if (mat->context->perm_local) {
                 	src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[i],&rowlen,tmpcol,tmpval,NULL);
                } else {
                	src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,NULL);
                }

                int start_col = mat->ncols + mat->nrows;
                int end_col   = 0;
    
                if(mat->context->perm_local->colPerm == NULL) {
                	for(int j=0; j<rowlen; ++j) {
                		start_col = MIN(start_col, mat->context->perm_local->perm[tmpcol[j]]);
                        	end_col   = MAX(end_col, mat->context->perm_local->perm[tmpcol[j]]);
                	} 
 		} else {
                        for(int j=0; j<rowlen; ++j) {
                                start_col = MIN(start_col, mat->context->perm_local->colPerm[tmpcol[j]]);
                                end_col   = MAX(end_col, mat->context->perm_local->colPerm[tmpcol[j]]);
                        }
		}


               for(int k=0; k<n_zones; ++k) {
                   //pure zone
                   if(start_col >= rhs_split[k] && end_col < rhs_split[k+1]) {
                      zone[i] = 2*k;
                    }
                   //transition zone
                   else if(zone[i]<0 && (start_col >= rhs_split[k] && end_col < rhs_split[k+2]) ) {
                      zone[i] = 2*k+1;
                   }
                   //else one can also add the rest of them for Multicoloring
                   else if(k==n_zones-1 && zone[i]<0) {
                  //   printf("check %d and %d on row %d\n", tmpcol[0],tmpcol[rowlen-1],i);
                     zone[i] = 2*n_zones;      //put Multicolor as last segment, but for time being we do this zone using single thread          
                     ctr_MC+=1;
//TODO  - increase zone by 1                   zone[i] = 2*n_zones + 1;
                   }
               }
        }

printf("zones to color = %d",ctr_MC);

      free(tmpcol);
      free(tmpval);
     }
   }
            
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
   	//this branch if no local permutations are carried out before
    	WARNING_LOG("The matrix has not been RCM permuted, BLOCK coloring works better for matrix with small bandwidths\n");
    	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local,sizeof(ghost_permutation)), err, ret);
    	mat->context->perm_local->scope = GHOST_PERMUTATION_LOCAL;
    	mat->context->perm_local->len = mat->nrows;
    	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->perm,sizeof(ghost_gidx)*mat->nrows), err, ret);
    	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->invPerm,sizeof(ghost_gidx)*mat->nrows), err, ret);
        mat->context->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;

        //anyway separate both permutations 
   	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colPerm,sizeof(ghost_gidx)*mat->ncols), err, ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colInvPerm,sizeof(ghost_gidx)*mat->ncols), err, ret);

        	#pragma omp parallel for         
        	for(int i=0; i<mat->ncols; ++i) {
        		mat->context->perm_local->colPerm[i] = i;
			mat->context->perm_local->colInvPerm[i] = i;
         	} 
     } else if(mat->context->perm_local->method == GHOST_PERMUTATION_SYMMETRIC) {
        	//this branch if no unsymmetric permutations have been carried out before
                   
        	if(mat->nrows != mat->ncols) {
			ERROR_LOG("Trying to do symmetric permutation on non-squared matrix\n");
        	}

        //now make it unsymmetric
        mat->context->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;

        //anyway separate both permutations 
   	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colPerm,sizeof(ghost_gidx)*mat->ncols), err, ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colInvPerm,sizeof(ghost_gidx)*mat->ncols), err, ret);

        #pragma omp parallel for         
        for(int i=0; i<mat->ncols; ++i) {
        	mat->context->perm_local->colPerm[i] =  mat->context->perm_local->perm[i];
		mat->context->perm_local->colInvPerm[i] =  mat->context->perm_local->invPerm[i];
         } 
     }

    mat->nzones = 2*n_zones ;
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->zone_ptr,(mat->nzones+2)*sizeof(ghost_lidx)), err, ret);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(mat->nzones+1)*sizeof(ghost_lidx)), err, ret);
    memset(curcol,0,(mat->nzones+1)*sizeof(ghost_lidx));

    for (int i=0;i<mat->nzones+2;i++) {
        mat->zone_ptr[i] = 0;
    }

   for (int i=0;i<mat->nrows;i++) {
        mat->zone_ptr[zone[i]+1]++;
    }

   for (int i=1;i<mat->nzones+2;i++) {
        mat->zone_ptr[i] += mat->zone_ptr[i-1];
    }

    mat->zone_ptr[mat->nzones+1] = nrows;

if(mat->context->perm_local)
    for (int i=0;i<mat->nrows;i++) {
        mat->context->perm_local->perm[mat->context->perm_local->invPerm[i]] = curcol[zone[i]] + mat->zone_ptr[zone[i]];
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

  //TODO previous permutation only to rows, of A and b
 
  //now copy multicoloring, TODO : multicolor this but now  doing as single thread
  mat->ncolors = 1;//mat->zone_ptr[mat->nzones+1] - mat->zone_ptr[mat->nzones];
  ghost_malloc((void **)&mat->color_ptr,(mat->ncolors+1)*sizeof(ghost_lidx)); 
  
  for(int i=0; i<mat->ncolors+1; ++i) {
  	mat->color_ptr[i] = mat->zone_ptr[mat->nzones+i];
  }
 
  double MC_percent;
  MC_percent = ((double)ctr_MC/mat->context->lnrows[me])*100.;
  if( MC_percent > 5  ) {
          WARNING_LOG("%3.2f % rows would be serialized, try reducing number of threads\n", MC_percent);
  }

  if( MC_percent > 40 ) {
          WARNING_LOG("Using Multicoloring KACZ would be best\n");  
  }
	
  goto out;

err:

out:
 free(curcol);
 return ret;
}



