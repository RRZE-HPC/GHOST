#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#include "ghost/omp.h"
#include <omp.h>

typedef enum {
 MIN_LOWER = 0,
 MAX_LOWER = 1,
 MIN_UPPER = 2,
 MAX_UPPER = 3
}zone_extrema;

ghost_error find_zone_extrema(ghost_sparsemat *mat, int **extrema, ghost_lidx a, ghost_lidx b) 
{
 //for SELL-1-1
 ghost_lidx *row_ptr = mat->sell->chunkStart;
 ghost_lidx *col_ptr = mat->sell->col;
 ghost_error ret = GHOST_SUCCESS;

 GHOST_CALL_GOTO(ghost_malloc((void **)extrema,sizeof(int)*4),err,ret);
 
 int max_lower = 0;
 int max_upper = 0;
 int min_lower = mat->ncols;
 int min_upper = mat->ncols;
 
 for(int i=a-1; i<=b-1; ++i) {
 	min_lower = MIN( min_lower, col_ptr[row_ptr[i]] );
        max_lower = MAX( max_lower, col_ptr[row_ptr[i]] );
        min_upper = MIN( min_upper, col_ptr[row_ptr[i+1]-1] );
        max_upper = MAX( max_upper, col_ptr[row_ptr[i+1]-1] );
 }

 (*extrema)[MIN_LOWER] = min_lower;
 (*extrema)[MAX_LOWER] = max_lower;
 (*extrema)[MIN_UPPER] = min_upper;
 (*extrema)[MAX_UPPER] = max_upper;
 
 goto out;

 err:

 out:
 return ret;
}

ghost_error split_transition(ghost_sparsemat *mat) 
{
     ghost_error ret = GHOST_SUCCESS;

     //for SELL-1-1
     ghost_lidx *row_ptr = mat->sell->chunkStart;
     ghost_lidx *col_ptr = mat->sell->col;
           
      int nthread[1];
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
   ghost_lidx *new_zone_ptr = NULL;
 
   mat->nzones = mat->nzones + 2*(n_zones-1);//add the new zones
  
   GHOST_CALL_GOTO(ghost_malloc((void **)&new_zone_ptr,sizeof(ghost_lidx)*(4*n_zones-2)),err,ret);

  for (int i=0; i<n_zones; ++i) {
        new_zone_ptr[4*i] = mat->zone_ptr[2*i];
        new_zone_ptr[4*i+1] = mat->zone_ptr[2*i+1];
  }

 

  for (int i=0; i<n_zones-1; ++i) {
  	ghost_lidx red_start       = new_zone_ptr[4*i+1];
        ghost_lidx red_end         = -1;
        ghost_lidx black_start     = -1;
        ghost_lidx black_end       = new_zone_ptr[4*i+4];

        int *extrema_pre, *extrema_post;

        find_zone_extrema(mat, &extrema_pre,  new_zone_ptr[4*i],   new_zone_ptr[4*i+1]);
        find_zone_extrema(mat, &extrema_post, new_zone_ptr[4*i+4], new_zone_ptr[4*i+5]); 

        ghost_lidx red_end_col     = extrema_pre[MIN_UPPER];//col_ptr[row_ptr[new_zone_ptr[4*i]+1]-1];
        ghost_lidx black_start_col = extrema_post[MAX_LOWER];//col_ptr[row_ptr[new_zone_ptr[4*i+5]]];

        bool flag_red   = 0;
        bool flag_black = 0;

       //now find corresponding rows of red_end and black_start
        for (int j=red_start; j<black_end; ++j) {
        	if(flag_red == 0 || flag_black == 0) {
 			if(flag_red == 0 && col_ptr[row_ptr[j]] > red_end_col) {
                        	red_end  = j;
                                flag_red = 1;
                        }
                        if(flag_black == 0 && col_ptr[row_ptr[j+1]-1] >= black_start_col) {
                                black_start  = j;
                                flag_black = 1;
                        }
                 }
        }

       free(extrema_pre);
       free(extrema_post);
  
       //now check the cases 
       if(flag_red == 0 && flag_black == 0) {
  		new_zone_ptr[4*i+2] = black_end;
                new_zone_ptr[4*i+3] = black_end;
       }
       else if(flag_red == 0 && flag_black == 1) {
                new_zone_ptr[4*i+2] = black_start;
                new_zone_ptr[4*i+3] = black_start;
       }
       else if(flag_red == 1 && flag_black == 0) {
                new_zone_ptr[4*i+2] = red_end;
                new_zone_ptr[4*i+3] = red_end;
       }
       //this is the only case where we have transition zones
       else if(black_start < red_end) {
                new_zone_ptr[4*i+2] = black_start;
                new_zone_ptr[4*i+3] = red_end;
       }
       else if(black_start >= red_end) {
                int median = (int)( (black_start+red_end)/2.0 );
                new_zone_ptr[4*i+2] = median;
                new_zone_ptr[4*i+3] = median;
       }
       else {
	       EROOR_LOG("GHOST UNKNOWN ERROR in hybrid splitting \n");
               ret = GHOST_ERR_UNKNOWN;
               goto err;
      }
    
   }   

      mat->zone_ptr = new_zone_ptr;
      goto out;
 
err:

out:
 return ret;
      
}
       
           
