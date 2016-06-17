#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#include "ghost/omp.h"
#include <omp.h>
#ifdef GHOST_HAVE_COLPACK
#include "ColPack/ColPackHeaders.h"
#endif
#include <vector>

extern "C" ghost_error ghost_sparsemat_blockColor(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType) 
{

    INFO_LOG("Create partition and permute (Block coloring)");
    ghost_error ret = GHOST_SUCCESS;
    ghost_lidx *curcol = NULL; 
    bool old_perm = true;
    //a counter for number of rows to be multicolored
    ghost_lidx ctr_nrows_MC = 0;
    ghost_lidx ctr_nnz_MC = 0;
    int total_bw =0;
    int max_col_idx = 0;
    ghost_lidx local_size;
    ghost_lidx *rhs_split = NULL;
    ghost_lidx *zone;
    ghost_lidx nrows =0;
    ghost_lidx nnz = 0;
    ghost_gidx *rpt = NULL;
    ghost_gidx *col = NULL;
    std::vector<int>  colvec;   
    
#ifdef GHOST_HAVE_COLPACK
    ghost_lidx offs = 0;
    ghost_lidx pos = 0;
    uint32_t** adolc; 
    uint32_t *adolc_data = NULL;
#endif
 
    ghost_gidx ncols_halo_padded = mat->ncols;

    if (mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
     	ncols_halo_padded = mat->context->nrowspadded + mat->context->halo_elements+1;
   }

    #ifdef GHOST_HAVE_COLPACK
     ColPack::BipartiteGraphPartialColoringInterface *GC;
     adolc = new uint32_t*[mat->nrows];
    #endif
   
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

   //printf("...NZONES... = %d\n", n_zones);
   int me;


   GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);

   nrows = mat->context->lnrows[me];


  //  ghost_lidx *row_ptr = mat->sell->chunkStart;
  //  ghost_lidx *col_ptr = mat->sell->col;
  //  ghost_lidx nrows = mat->nrows;
   

  //  ghost_lidx n_t_zones = n_zones-1; 
  //
/*    if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
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
                 	src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,tmpcol,tmpval,src->arg);
              	} else if (mat->context->perm_global) {
                	src->func(mat->context->perm_global->invPerm[i],&rowlen,tmpcol,tmpval,src->arg);
            	} else if (mat->context->perm_local) {
                 	src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[i],&rowlen,tmpcol,tmpval,src->arg);
                } else {
                	src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,src->arg);
                }

                ghost_gidx start_col = mat->nrows + mat->context->nrowspadded;
                ghost_gidx end_col   = 0;

                if(mat->context->perm_local){
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
		} else {
		                for(int j=0; j<rowlen; ++j) {
                                	start_col = MIN(start_col, tmpcol[j]);
                                	end_col   = MAX(end_col, tmpcol[j]);
                       		}
        	}
                lower_bw = MAX(lower_bw, i-start_col);
                upper_bw = MAX(upper_bw , end_col - i);
                max_col_idx = MAX(max_col_idx, end_col);
        }
 	
	free(tmpcol);
	free(tmpval);
    }
  }*/
  //std::cout<<"nrows ="<<mat->nrows<<std::endl;
  //std::cout<<"check"<<row_ptr[mat->nrows-1]<<std::endl;

    total_bw = mat->bandwidth;
    max_col_idx = mat->maxColRange;

    //approximate, currently last thread is set to not have anything is transitional sweeps;
    //since large bandwidths might create problems (TODO optimise this value)
    local_size = static_cast<int>( ((max_col_idx+1)-0*total_bw) / n_zones); //will have to find a method to balance the load for the last thread, 	
				                 			      // even if the last thread gets a bit less load its fine since the excess load 
									      // is evenly spread among rest of the threads

    rhs_split = new int[n_zones+2];
   
    for(int i=0; i<n_zones+1; ++i) {
      rhs_split[i] = i*local_size; 
    }
 
    rhs_split[n_zones+1] = max_col_idx+1;
    //printf("%d\n",(int)rhs_split[n_zones+1]);

     

    zone = new ghost_lidx[nrows]; 
 
    for(int i=0; i<nrows; ++i) {
        zone[i] = -(n_zones+1) ; //an invalid number
    }

 if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
       ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
       ghost_gidx * tmpcol = NULL;
       char * tmpval = NULL;
       ghost_lidx rowlen;

  #pragma omp parallel private (tmpval,tmpcol,rowlen) reduction(+:nnz) 
    {
        ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
        ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
        
#pragma omp for
        for (int i=0; i<mat->context->lnrows[me]; i++) {
            if (mat->context->perm_global && mat->context->perm_local) {
                src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,tmpcol,tmpval,src->arg);
            } else if (mat->context->perm_global) {
                src->func(mat->context->perm_global->invPerm[i],&rowlen,tmpcol,tmpval,src->arg);
            } else if (mat->context->perm_local) {
                src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[i],&rowlen,tmpcol,tmpval,src->arg);
            } else {
                src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,src->arg);
            }
            nnz += rowlen;
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
}
 
   if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
       ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
//       ghost_gidx * tmpcol = NULL;
       char * tmpval = NULL;

       ghost_lidx rowlen;

       GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(mat->context->lnrows[me]+1) * sizeof(ghost_gidx)),err,ret);

       rpt[0] = 0;
 
       GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_gidx)),err,ret);
 
//#pragma omp parallel private(tmpval,tmpcol,rowlen) 
#pragma omp parallel private(tmpval,rowlen)       
{
//       ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
       ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
    
#pragma omp for ordered reduction(+:ctr_nrows_MC) 
        for (int i=0; i<mat->context->lnrows[me]; i++) {
#pragma omp ordered
	{
        	if (mat->context->perm_global && mat->context->perm_local) {
//                 	src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,tmpcol,tmpval,src->arg);
                 	src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,&col[rpt[i]],tmpval,src->arg);
              	} else if (mat->context->perm_global) {
                	src->func(mat->context->perm_global->invPerm[i],&rowlen,&col[rpt[i]],tmpval,src->arg);
            	} else if (mat->context->perm_local) {
                 	src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[i],&rowlen,&col[rpt[i]],tmpval,src->arg);
                } else {
                	src->func(mat->context->lfRow[me]+i,&rowlen,&col[rpt[i]],tmpval,src->arg);
                }


//		nnz+=rowlen;
		rpt[i+1] = rpt[i] + rowlen;
	}

                int start_col =  mat->nrows + ncols_halo_padded;
                int end_col   = 0;
    
		if(mat->context->perm_local) {
                	if(mat->context->perm_local->colPerm == NULL) {
                		for(int j=rpt[i]; j<rpt[i+1]; ++j) {
                			start_col = MIN(start_col, mat->context->perm_local->perm[col[j]]);
                        		end_col   = MAX(end_col, mat->context->perm_local->perm[col[j]]);
                		} 
 			} else {
                        	for(int j=rpt[i]; j<rpt[i+1]; ++j) {
                                	start_col = MIN(start_col, mat->context->perm_local->colPerm[col[j]]);
                                	end_col   = MAX(end_col, mat->context->perm_local->colPerm[col[j]]);
                        	}
			}
		} else {
	                for(int j=rpt[i]; j<rpt[i+1]; ++j) {
                               	start_col = MIN(start_col, col[j]);
                               	end_col   = MAX(end_col, col[j]);
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
                     zone[i] = 2*n_zones;      //put Multicolor as last segment, but for time being we do this zone using single thread          
                     ctr_nrows_MC+=1;
                     ctr_nnz_MC += rowlen;
		    //TODO  - increase zone by 1        zone[i] = 2*n_zones + 1;
                   } 
               }
        }

      #pragma omp single
	{
		INFO_LOG("NO. of Rows Multicolored = %d\n",ctr_nrows_MC);
	}
	
     // free(tmpcol);
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
	old_perm = false;
    	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->perm,sizeof(ghost_gidx)*mat->nrows), err, ret);
    	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->invPerm,sizeof(ghost_gidx)*mat->nrows), err, ret);
  
        mat->context->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;

        //anyway separate both permutations 
   	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colPerm,sizeof(ghost_gidx)*ncols_halo_padded), err, ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colInvPerm,sizeof(ghost_gidx)*ncols_halo_padded), err, ret);

      	#pragma omp parallel for         
       	for(int i=0; i<ncols_halo_padded; ++i) { 
       		mat->context->perm_local->colPerm[i] = i;
		mat->context->perm_local->colInvPerm[i] = i;
       	} 

     } else if(mat->context->perm_local->method == GHOST_PERMUTATION_SYMMETRIC) {
       	//this branch if no unsymmetric permutations have been carried out before
                 
       	if(mat->nrows != mat->context->nrowspadded) {
		ERROR_LOG("Trying to do symmetric permutation on non-squared matrix\n");
       	}

        //now make it unsymmetric
        mat->context->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;

        //anyway separate both permutations 
   	GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colPerm,sizeof(ghost_gidx)*mat->ncols), err, ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colInvPerm,sizeof(ghost_gidx)*mat->ncols), err, ret);

        #pragma omp parallel for         
        for(int i=0; i<ncols_halo_padded; ++i) {
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

    if(old_perm){
    	for (int i=0;i<mat->nrows;i++) {
        	mat->context->perm_local->perm[mat->context->perm_local->invPerm[i]] = curcol[zone[i]] + mat->zone_ptr[zone[i]];
       		curcol[zone[i]]++;
    	}
    } else {  
    	for (int i=0;i<mat->nrows;i++) {
        	mat->context->perm_local->perm[i] = curcol[zone[i]] + mat->zone_ptr[zone[i]];
        	curcol[zone[i]]++;
    	}
    }

    for (int i=0;i<mat->nrows;i++) {
        mat->context->perm_local->invPerm[mat->context->perm_local->perm[i]] = i;
    }

#ifdef GHOST_HAVE_COLPACK         
    INFO_LOG("Create permutation from coloring");
    free(curcol);
    //now build adolc for multicoloring
    //int nrows_to_mc = mat->zone_ptr[mat->nzones+1]-mat->zone_ptr[mat->nzones];
    offs = mat->zone_ptr[mat->nzones];		
    adolc_data = new uint32_t[ctr_nnz_MC + ctr_nrows_MC];
    pos = 0;

//here if a previous permutation like RCM was carried out we either need the original unpermuted matrix or a totally permuted matrix
if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
    ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
    ghost_gidx * tmpcol = NULL;
    char * tmpval = NULL;
    ghost_lidx rowlen;
 
    ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
    ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
 
    for(int i=0; i< ctr_nrows_MC; ++i) {
	if (mat->context->perm_global && mat->context->perm_local) {
                 src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[offs+i]],&rowlen,tmpcol,tmpval,src->arg);
        } else if (mat->context->perm_global) {
                src->func(mat->context->perm_global->invPerm[offs+i],&rowlen,tmpcol,tmpval,src->arg);
        } else if (mat->context->perm_local) {
               	src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[offs+i],&rowlen,tmpcol,tmpval,src->arg);
        }
	
	adolc[i] = &(adolc_data[pos]);
    	adolc_data[pos++] = rowlen;
	for(int j=0; j<rowlen; ++j) {
		adolc_data[pos++] = tmpcol[j];
	}
    }
free(tmpcol);
free(tmpval);
}

   //build Bipartite Graph
    GC =  new ColPack::BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC,adolc,ctr_nrows_MC,ncols_halo_padded); 
    COLPACK_CALL_GOTO(GC->PartialDistanceTwoColoring("NATURAL","ROW_PARTIAL_DISTANCE_TWO"),err,ret);

    if(!GC->CheckPartialDistanceTwoRowColoring()) {
        ERROR_LOG("Error in coloring!");
        ret = GHOST_ERR_COLPACK;
        goto err;
    }

    mat->ncolors = GC->GetVertexColorCount();
    GC->GetVertexPartialColors(colvec);
    INFO_LOG("No. of Colors = %d",mat->ncolors);
  
    ghost_malloc((void **)&mat->color_ptr,(mat->ncolors+1)*sizeof(ghost_lidx)); 
    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(mat->ncolors)*sizeof(ghost_lidx)),err,ret);
    memset(curcol,0,mat->ncolors*sizeof(ghost_lidx));
 
    for (int i=0;i<mat->ncolors+1;i++) {
        mat->color_ptr[i] = 0; 
    }

    for (int i=0;i<ctr_nrows_MC;i++) {
        mat->color_ptr[colvec[i]+1]++;
    }

    for (int i=1;i<mat->ncolors+1;i++) {
        mat->color_ptr[i] += mat->color_ptr[i-1];
    }

    for(int i=0; i<mat->ncolors+1;++i) {
	mat->color_ptr[i] = mat->color_ptr[i] + offs; //add offset to it
    }
     
    for (int i=0;i<ctr_nrows_MC;i++) {
        int idx = mat->context->perm_local->invPerm[i+offs];
        mat->context->perm_local->perm[idx]  = curcol[colvec[i]] + mat->color_ptr[colvec[i]];
        curcol[colvec[i]]++;
    }

    for (int i=0;i<mat->nrows;i++) {
        mat->context->perm_local->invPerm[mat->context->perm_local->perm[i]] = i;
    }

#else
    mat->ncolors = 1;
    ghost_malloc((void **)&mat->color_ptr,(mat->ncolors+1)*sizeof(ghost_lidx)); 
  
    for(int i=0; i<mat->ncolors+1; ++i) {
  	mat->color_ptr[i] = mat->zone_ptr[mat->nzones+i];
    }

#endif 

  double MC_percent;
  MC_percent = ((double)ctr_nrows_MC/mat->context->lnrows[me])*100.;
//TODO : quantify this and give a break point
  if( MC_percent > 5  ) {
          WARNING_LOG("%3.2f %% rows would be Multicolored, try reducing number of threads", MC_percent);
  }


  goto out;

err:

out:
 free(curcol);
 return ret;
}



