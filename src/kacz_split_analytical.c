#include "ghost/kacz_split_analytical.h"

typedef enum {
 MIN_LOWER = 0,
 MAX_LOWER = 1,
 MIN_UPPER = 2,
 MAX_UPPER = 3
}zone_extrema;

//returns the virtual column index; ie takes into account the permutation of halo elements also
#define virtual_col(col_idx)\
   (mat->context->flags & GHOST_PERM_NO_DISTINCTION)?( (col_ptr[col_idx]<mat->context->nrowspadded)?col_ptr[col_idx]:mat->context->perm_local->colPerm[col_ptr[col_idx]] ):col_ptr[col_idx]\


/*
ghost_error find_zone_extrema(ghost_sparsemat *mat, int **extrema, ghost_lidx a, ghost_lidx b) 
{
 //for SELL-1-1
 ghost_lidx *row_ptr = mat->sell->chunkStart;
 ghost_lidx *col_ptr = mat->sell->col;//virtual_col would be used
 ghost_error ret = GHOST_SUCCESS;

 GHOST_CALL_GOTO(ghost_malloc((void **)extrema,sizeof(int)*4),err,ret);
 
 int max_lower = 0;
 int max_upper = 0;
 int min_lower = mat->ncols;
 int min_upper = mat->ncols;

 //TODO work on virtual columns 
 for(int i=a; i< b; ++i) {
 	min_lower = MIN( min_lower, virtual_col(row_ptr[i]) );
        max_lower = MAX( max_lower, virtual_col(row_ptr[i]) );
        min_upper = MIN( min_upper, virtual_col(row_ptr[i+1]-1) );
int me;
ghost_rank(&me,mat->context->mpicomm);
if(col_ptr[row_ptr[i+1]-1] > mat->context->nrowspadded){
	printf("<%d> col changed from %d to %d\n", me,col_ptr[row_ptr[i+1]-1], virtual_col(row_ptr[i+1]-1));
}
        max_upper = MAX( max_upper, virtual_col(row_ptr[i+1]-1) );
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
*/
//This check is not necessary since this should not fail, if implemented correctly
/*ghost_error checker(ghost_sparsemat *mat)
{
     ghost_error ret = GHOST_SUCCESS;
     ghost_lidx *zones = mat->zone_ptr;

     //for SELL-1-1
     ghost_lidx *row_ptr = mat->sell->chunkStart;

     //TODO give virtual columns
     ghost_lidx *col_ptr = mat->sell->col;
  
     int *extrema_pure, *extrema_red, *extrema_black, *extrema_trans;
     int pure_min, pure_max, red_min, red_max, black_min, black_max, trans_min, trans_max;

     find_zone_extrema(mat, &extrema_pure, zones[0], zones[1]);
     find_zone_extrema(mat, &extrema_red,  zones[1], zones[2]);
     find_zone_extrema(mat, &extrema_trans,  zones[2], zones[3]);
     find_zone_extrema(mat, &extrema_black,  zones[3], zones[4]);


     for(int i=1; i<mat->kacz_setting.active_threads; ++i) {
      
        pure_max = extrema_pure[MAX_UPPER]; 
        free(extrema_pure);
        find_zone_extrema(mat, &extrema_pure, zones[4*i], zones[4*i+1]);
        pure_min = extrema_pure[MIN_LOWER];

        //check pure zones
	if( (zones[4*i] != zones[4*i+1]) && ((zones[4*(i-1)] != zones[4*(i-1)+1])) && pure_min <= pure_max ) {
           ret = GHOST_ERR_BLOCKCOLOR;
           printf("ERR 1\n");
           printf("error occured at pure_min=%d, pure_max=%d",pure_min,pure_max); 
//	   break;
	}
       

        red_max = extrema_red[MAX_UPPER];
        free(extrema_red);
        find_zone_extrema(mat, &extrema_red, zones[4*i+1], zones[4*i+2]);
        red_min = extrema_red[MIN_LOWER];

        //check transition zones
        //check red color in transition zones
  	if( (zones[4*i+1] != zones[4*i+2]) && (zones[4*(i-1)+1] != zones[4*(i-1)+2]) && red_min <= red_max) {//col_ptr[row_ptr[zones[4*i+1]]] <= col_ptr[row_ptr[zones[4*i-2]]-1] ) {
            ret = GHOST_ERR_BLOCKCOLOR;
	    printf("ERR 2\n");
//	    break;
	}

        black_max = extrema_black[MAX_UPPER];
        free(extrema_black);
        find_zone_extrema(mat, &extrema_black, zones[4*i+3], zones[4*i+4]);
        black_min = extrema_black[MIN_LOWER];

		//check black color in transition zones
        if( (zones[4*i+3] != zones[4*i+4]) &&(zones[4*(i-1)+3] != zones[4*(i-1)+4]) && black_min <= black_max) {//col_ptr[row_ptr[zones[4*i+3]]] <= col_ptr[row_ptr[zones[4*i]]-1] ) {
printf("check  lower = %d, upper = %d\n",virtual_col(row_ptr[zones[4*i+3]]),virtual_col(row_ptr[zones[4*i]]-1) );
            ret = GHOST_ERR_BLOCKCOLOR;
	    printf("ERR 3\n");
//	    break;
	}
		//check transition in transition zones, if we are using one sweep method, 
	if(mat->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_one_sweep) {
	        trans_max = extrema_trans[MAX_UPPER];
        	free(extrema_trans);
        	find_zone_extrema(mat, &extrema_trans, zones[4*i+2], zones[4*i+3]);
        	trans_min = extrema_trans[MIN_LOWER];

               	if( (zones[4*i+2] != zones[4*i+3]) &&(zones[4*(i-1)+2] != zones[4*(i-1)+3]) && trans_min <= trans_max) {//col_ptr[row_ptr[zones[4*i+2]]] <= col_ptr[row_ptr[zones[4*i-1]]-1] ) {
                        ret = GHOST_ERR_BLOCKCOLOR;
	        	printf("ERR 4\n");

 		//	break;
		}
	}
     }

 if(extrema_pure != NULL)
     free(extrema_pure);
 if(extrema_red != NULL)
     free(extrema_red);
 if(extrema_black != NULL)
     free(extrema_black);
 if(extrema_trans != NULL)
    free(extrema_trans);

 if(ret == GHOST_ERR_BLOCKCOLOR)
  ERROR_LOG("ERROR in BLOCK COLORING, Check hybrid splitting \n");

  
 return ret;            	
}
*/

//finds lower and upper bandwidth of the matrix
/*ghost_error mat_bandwidth(ghost_sparsemat *mat, int *lower_bw, int *upper_bw)
{
  int lower = 0;
  int upper = 0;
  ghost_lidx* row_ptr = mat->sell->chunkStart; 
  ghost_lidx* col_ptr = mat->sell->col;//TODO give virtual colums
  
  //std::cout<<"nrows ="<<mat->nrows<<std::endl;
  //std::cout<<"check"<<row_ptr[mat->nrows-1]<<std::endl;

//TODO replace redundant bw calculations
   for(int i=0; i<mat->nrows; ++i){
            lower = MAX(lower,i - virtual_col(row_ptr[i]));
            upper = MAX(upper, virtual_col(row_ptr[i+1]-1) - i);
   }

  *lower_bw = lower;
  *upper_bw = upper;

  return GHOST_SUCCESS;
}
*/ 
ghost_error split_analytical(ghost_sparsemat *mat) 
{
     //for KACZ_ANALYZE
#ifdef GHOST_KACZ_ANALYZE 
     ghost_lidx line_size, n_lines, rem_lines;
     int start=0 , end=0;
     ghost_lidx *rows;
     ghost_lidx *nnz;
#endif

     int height = mat->nrows;
     int width  = mat->maxColRange+1;
     double diagonal_slope = (double)(height)/width;
     //int separation = (int)ceil(diagonal_slope*mat->bandwidth);
     int possible_threads = (int) ((double)height/mat->bandwidth); //height/separation

     ghost_error ret = GHOST_SUCCESS;
     int *nthread ;
     GHOST_CALL_GOTO(ghost_malloc((void **)&nthread, 1*sizeof(int)),err,ret);

      #ifdef GHOST_HAVE_OPENMP
	 #pragma omp parallel
   	 {
     	   #pragma omp master
     	   nthread[0] = ghost_omp_nthread();
   	 }
	#else
    	   nthread[0] = 1;
      #endif
   
      int current_threads = nthread[0];
      free(nthread);
 
     if( current_threads > possible_threads) {
	WARNING_LOG("Specified number of threads cannot be used for the specified KACZ kernel, setting from %d to %d",current_threads,possible_threads);
        current_threads = possible_threads;
        // disable dynamic thread adjustments 
        ghost_omp_set_dynamic(0);
        ghost_omp_nthread_set(current_threads);
     } 

     mat->kacz_setting.active_threads = current_threads;


     ghost_lidx *chunk_ptr = mat->sell->chunkStart;
     ghost_lidx *col_ptr = mat->sell->col;
     ghost_lidx chunkheight = mat->traits.C;

     mat->nzones = 4*current_threads;
     ghost_malloc((void **)&mat->zone_ptr,(mat->nzones+2)*sizeof(ghost_lidx)); //one extra zone added for convenience
     ghost_lidx *zone_ptr = mat->zone_ptr;

     int pure_gap = (int)( ((double)height/current_threads));
     int pure_thickness = (int)( ( ((double)height/current_threads)-mat->bandwidth*diagonal_slope)) + 1;   
     
     int red_ctr = 0; ;
     int black_ctr = 0; ;
     int black_start = 0;
     int red_end = 0;
     int median = 0;

     for(int i=0; i<current_threads; ++i) {
	zone_ptr[4*i] 		= i*pure_gap;
	zone_ptr[4*i+1]		= zone_ptr[4*i] + pure_thickness;
     }

     zone_ptr[4*current_threads] = height;
     zone_ptr[4*current_threads+1] = height+pure_thickness;//dummy

     for(int i=0; i<current_threads; ++i) {
	black_start = zone_ptr[4*i] + (int) ceil(mat->bandwidth*diagonal_slope);
	red_end     = zone_ptr[4*(i+1)+1] - (int) ceil(mat->bandwidth*diagonal_slope);
	if(black_start<zone_ptr[4*i+1]) {
		black_start = zone_ptr[4*i+1];
		black_ctr  += 1;
	}

	if(red_end>zone_ptr[4*(i+1)]) {
		red_end = zone_ptr[4*(i+1)];
		red_ctr += 1;
	}


	if(black_start <= red_end) {
		median = (int) ((black_start+red_end)/2.0);
		zone_ptr[4*i+2] = median;
		zone_ptr[4*i+3] = median;
	} else {
		zone_ptr[4*i+2] = red_end;
		zone_ptr[4*i+3] = black_start;
	}
     }

   if( (red_ctr == current_threads) && (black_ctr == current_threads) ) {
	INFO_LOG("USING RED BLACK SWEEP WITHOUT TRANSITION");
   }

    //multicoloring is also dummy initialise pointers so we can use the same kacz kernel
    mat->ncolors = 1;//mat->zone_ptr[mat->nzones+1] - mat->zone_ptr[mat->nzones];
    ghost_malloc((void **)&mat->color_ptr,(mat->ncolors+1)*sizeof(ghost_lidx)); 
  
    for(int i=0; i<mat->ncolors+1; ++i) {
  	mat->color_ptr[i] = mat->zone_ptr[mat->nzones];
    }
 
   mat->kacz_setting.kacz_method = GHOST_KACZ_METHOD_BMC_one_sweep;
 
   for(int i=1; i<current_threads; ++i) {	 
        ghost_lidx chunk_lower      = zone_ptr[4*i+2]/chunkheight;
        ghost_lidx rowinchunk_lower = zone_ptr[4*i+2]%chunkheight;
        ghost_lidx chunk_upper 	    = (zone_ptr[4*i-1]-1)/chunkheight;     
        ghost_lidx rowinchunk_upper = (zone_ptr[4*i-1]-1)%chunkheight;

	ghost_lidx lower = virtual_col(chunk_ptr[chunk_lower]+rowinchunk_lower); 		
	ghost_lidx upper = virtual_col(chunk_ptr[chunk_upper]+rowinchunk_upper+chunkheight*(mat->sell->chunkLen[chunk_upper]-1));

       	if(lower <= upper) {
           //printf("check lower = %d and upper =%d\n",virtual_col(row_ptr[zone_ptr[4*i+2]]) , virtual_col(row_ptr[zone_ptr[4*i-1]]-1));
           mat->kacz_setting.kacz_method = GHOST_KACZ_METHOD_BMC_two_sweep;	
           WARNING_LOG("ONLY half the available threads would be used for transitional sweep\n");
           break;
       }
  }
#ifdef GHOST_KACZ_ANALYZE 
 kacz_analyze_print(mat);
#endif

 goto out;
 
 err:

 out:

 return ret;
}
             
