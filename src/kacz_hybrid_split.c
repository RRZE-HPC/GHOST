#include "ghost/kacz_hybrid_split.h"
#include <limits.h>

typedef enum {
    MIN_LOWER = 0,
    MAX_LOWER = 1,
    MIN_UPPER = 2,
    MAX_UPPER = 3
}zone_extrema;

//returns the virtual column index; ie takes into account the permutation of halo elements also
#define virtual_col(col_idx)\
(mat->context->flags & GHOST_PERM_NO_DISTINCTION)?( (col_ptr[col_idx]<mat->context->col_map->nrowspadded)?col_ptr[col_idx]:mat->context->col_map->loc_perm[col_ptr[col_idx]] ):col_ptr[col_idx]\



ghost_error find_zone_extrema(ghost_sparsemat *mat, int **extrema, ghost_lidx a, ghost_lidx b) 
{
    //for SELL-1-1
    ghost_lidx *chunk_ptr = mat->chunkStart;
    ghost_lidx *col_ptr = mat->col;//virtual_col would be used
    ghost_error ret = GHOST_SUCCESS;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)extrema,sizeof(int)*4),err,ret);
    
    ghost_lidx max_lower = 0;
    ghost_lidx max_upper = 0;
    ghost_lidx min_lower = INT_MAX;
    ghost_lidx min_upper = INT_MAX;
    ghost_lidx chunk = 0;
    ghost_lidx rowinchunk = 0;
    ghost_lidx chunkheight = mat->traits.C;
    
    //TODO work on virtual columns 
    for(int i=a; i< b; ++i) {
        chunk = i/chunkheight;			
        rowinchunk = i%chunkheight;
        
        min_lower = MIN( min_lower, virtual_col(chunk_ptr[chunk])+rowinchunk);
        max_lower = MAX( max_lower, virtual_col(chunk_ptr[chunk])+rowinchunk);
        min_upper = MIN( min_upper, virtual_col(chunk_ptr[chunk]+rowinchunk+chunkheight*(mat->chunkLen[chunk]-1)) );
        max_upper = MAX( max_upper, virtual_col(chunk_ptr[chunk]+rowinchunk+chunkheight*(mat->chunkLen[chunk]-1)) );
        
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

//This check is not necessary since this should not fail, if implemented correctly
//Right now this works only for 1 processor since if multiple processor the remote entries gets moved to the end
ghost_error checker(ghost_sparsemat *mat)
{
    ghost_error ret = GHOST_SUCCESS;
    ghost_lidx *zones = mat->zone_ptr;
    
    //for SELL-1-1
    ghost_lidx *row_ptr = mat->chunkStart;
    
    //TODO give virtual columns
    ghost_lidx *col_ptr = mat->col;
    
    int *extrema_pure, *extrema_red, *extrema_black, *extrema_trans, *extrema_trans_1, *extrema_trans_2;
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
            ERROR_LOG("ERR 1");
            ERROR_LOG("pure_min = %d, pure_max=%d, btw [%d-%d] and [%d-%d]",pure_min,pure_max, zones[4*(i-1)], zones[4*(i-1)+1],zones[4*i],zones[4*i+1]);
            // break;
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
            printf("check  lower = %"PRLIDX", upper = %"PRLIDX"\n",virtual_col(row_ptr[zones[4*i+3]]),virtual_col(row_ptr[zones[4*i]]-1) );
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
        } else if(i <mat->kacz_setting.active_threads-1) {
            find_zone_extrema(mat, &extrema_trans_1, zones[4*(i-1)+2], zones[4*(i-1)+3]);
            trans_max = extrema_trans_1[MAX_UPPER];
            free(extrema_trans_1);
            find_zone_extrema(mat, &extrema_trans_2, zones[4*i+6], zones[4*i+7]);
            trans_min = extrema_trans_2[MIN_LOWER];
            free(extrema_trans_2);
            
            if( (zones[4*i+6] != zones[4*i+7]) &&(zones[4*(i-1)+2] != zones[4*(i-1)+3]) && trans_min <= trans_max) {//col_ptr[row_ptr[zones[4*i+2]]] <= col_ptr[row_ptr[zones[4*i-1]]-1] ) {
                printf("check between %d-%d and %d-%d zoneptr\n",4*i-2,4*i-1,4*i+6,4*i+7);	
                ret = GHOST_ERR_BLOCKCOLOR;
                printf("ERR 5\n");
                
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

//finds lower and upper bandwidth of the matrix
ghost_error mat_bandwidth(ghost_sparsemat *mat, int *lower_bw, int *upper_bw, int a, int b)
{
    int lower = 0;
    int upper = 0;
    ghost_lidx* chunk_ptr = mat->chunkStart; 
    ghost_lidx* col_ptr = mat->col;//TODO give virtual colums
    int start_col, end_col;
    ghost_lidx chunk;
    ghost_lidx rowinchunk;
    ghost_lidx chunkheight = mat->traits.C;
    ghost_lidx idx = 0;
    
    /*   for(int i=a; i<b; ++i){
     *	   start_col = SPM_NROWS(mat) + mat->context->col_map->nrowspadded;
     *	   end_col   = 0;
     *	for(int j=chunk_ptr[i]; j<chunk_ptr[i+1]; ++j) {
     *           start_col = MIN(start_col, virtual_col(j));
     *           end_col   = MAX(end_col, virtual_col(j));
}

lower = MAX(lower,i-start_col);
upper = MAX(upper,end_col-i);
}
printf("Bandwidth from calculation crs = %d\n", lower+upper);
lower = 0;
upper = 0;
*/
    for(int i=a; i<b; ++i){
        chunk = i/chunkheight;   //can avoid this by doing reminder loops
        rowinchunk = i%chunkheight;
        start_col = SPM_NROWS(mat) + mat->context->col_map->nrowspadded;
        end_col   = 0;
        idx = chunk_ptr[chunk]+rowinchunk;
        
        for(int j=0; j<mat->chunkLen[chunk]; ++j) {
            if(j==0 || virtual_col(idx)!=0) { //TODO somehow fix it, since filling dummy columns with 0
                start_col = MIN(start_col, virtual_col(idx));
                end_col   = MAX(end_col,   virtual_col(idx));
            }
            idx+=chunkheight;
        }	
        
        lower = MAX(lower,i-start_col);
        upper = MAX(upper,end_col-i);
    }
    
    *lower_bw = lower;
    *upper_bw = upper;
    return GHOST_SUCCESS;
}

ghost_error split_transition(ghost_sparsemat *mat) 
{
    ghost_error ret = GHOST_SUCCESS;
    
    //ghost_lidx *row_ptr = mat->chunkStart;
    //ghost_lidx *col_ptr = mat->col;
    
    
    //height might vary from nrows if we have multicoloring
    ghost_lidx height = mat->zone_ptr[mat->nzones];
    //width might vary from ncols  if we consider remote permutations also
    //ghost_lidx width  = mat->maxColRange+1;
    
    
    int n_zones = mat->kacz_setting.active_threads;//nthread[0];
    ghost_lidx *new_zone_ptr = NULL;
    
    mat->nzones = mat->nzones + 2*(n_zones);//add the new zones
    
    
    //GHOST_CALL_GOTO(ghost_malloc((void **)&new_zone_ptr,sizeof(ghost_lidx)*(4*n_zones+2)),err,ret);
    new_zone_ptr = (ghost_lidx*) malloc(sizeof(ghost_lidx)*(4*n_zones+2));
    
    ghost_lidx lower_bw = 0;
    ghost_lidx upper_bw = 0;
    
    //the bandwidth might have changed due to previous permutations (it can also increase)
    //Further bandwidth only from 0 to height has to be calculated
    mat_bandwidth(mat, &lower_bw, &upper_bw, 0, height);
    
    ghost_lidx total_bw = lower_bw + upper_bw;//lower_bw + upper_bw;
    //printf("New lower b/w =%d, upper b/w=%d, total =%d",lower_bw,upper_bw,total_bw);
    //printf("HEIGHT = %d, WIDTH = %d\n",height,width); 
    //double diagonal_slope = (double)(height)/width;
    //ghost_lidx separation  = (int)(ceil((diagonal_slope*total_bw))); 
    
    for (int i=0; i<n_zones; ++i) {
        new_zone_ptr[4*i] = mat->zone_ptr[2*i];
        new_zone_ptr[4*i+1] = mat->zone_ptr[2*i+1];
    }
    
    new_zone_ptr[4*(n_zones)]  = mat->zone_ptr[2*(n_zones)] ;
    new_zone_ptr[4*(n_zones)+1] = mat->zone_ptr[2*(n_zones)] ; //simply for ease of calculation, not accessible by user
    
    for (int i=0; i<n_zones; ++i) {
        ghost_lidx black_start = new_zone_ptr[4*i] + total_bw;
        ghost_lidx red_end     = new_zone_ptr[4*i+5] - total_bw;
        
        if(i==n_zones-1)
            red_end = new_zone_ptr[4*i+5];
        
        if(black_start < new_zone_ptr[4*i+1])
            black_start = new_zone_ptr[4*i+1];
        
        if(black_start > new_zone_ptr[4*i+4])
            black_start = new_zone_ptr[4*i+4];
        
        if(red_end > new_zone_ptr[4*i+4])
            red_end = new_zone_ptr[4*i+4];
        
        if(red_end < new_zone_ptr[4*i+1])
            red_end = new_zone_ptr[4*i+1];
        
        //now check the cases
        if(black_start <= red_end) {
            int median =(int)( (black_start+red_end)/2.0);
            new_zone_ptr[4*i+2] = median; //else can leave as it is , but idea is to reduce this transition zones, will have to check load balancing
            new_zone_ptr[4*i+3] = median;
        } else {
            new_zone_ptr[4*i+2] = red_end;
            new_zone_ptr[4*i+3] = black_start;
        }
    }
    
    
    //now check whether the transition in transition is overlapping- if one region overlaps we use 2 sweep method (with threads/2) , else one sweep method 
    mat->kacz_setting.kacz_method = GHOST_KACZ_METHOD_BMC_one_sweep;
    
    for(int i=1; i<n_zones; ++i) {
        //	ghost_gidx lower = virtual_col(row_ptr[new_zone_ptr[4*i+2]]); //This might not work if the matrix is not RCM permuted
        //	ghost_gidx upper = virtual_col(row_ptr[new_zone_ptr[4*i-1]]-1);
        int *extrema_lower_trans = NULL, *extrema_upper_trans = NULL;
        
        find_zone_extrema(mat, &extrema_lower_trans, new_zone_ptr[4*i+2], new_zone_ptr[4*i+3]);    
        ghost_lidx lower = extrema_lower_trans[MIN_LOWER]; 
        
        find_zone_extrema(mat, &extrema_upper_trans, new_zone_ptr[4*i-2], new_zone_ptr[4*i-1]);
        ghost_lidx upper = extrema_lower_trans[MAX_UPPER]; 
        
        
        if(lower <= upper) {
            //printf("check lower = %d and upper =%d\n",virtual_col(row_ptr[new_zone_ptr[4*i+2]]) , virtual_col(row_ptr[new_zone_ptr[4*i-1]]-1));
            mat->kacz_setting.kacz_method = GHOST_KACZ_METHOD_BMC_two_sweep;	
            WARNING_LOG("ONLY half the available threads would be used for transitional sweep\n");
            break;
        }
        
        if(extrema_lower_trans != NULL)
            free(extrema_lower_trans);
        
        if(extrema_upper_trans != NULL)
            free(extrema_upper_trans);
    }
    
    
    mat->zone_ptr = new_zone_ptr;
    
    #ifdef GHOST_KACZ_ANALYZE 
    kacz_analyze_print(mat);
    #endif
    
    //currently be done only if CHUNKHEIGHT==1, and NO_DISTINCTION is on, since if no distinction is not on 
    //further permutation occurs after ghost_sparsemat_fromfunc_common , which permutes remote entries
    //this causes problem for checking although the result is correct
    if(mat->traits.C == 1 && mat->context->flags & GHOST_PERM_NO_DISTINCTION) 
    {
        INFO_LOG("CHECKING BLOCK COLORING")
        checker(mat);
        INFO_LOG("CHECKING FINISHED")
    }
    
    return ret;
    
}

//To be used only when the requirement is satisfied
ghost_error split_analytical(ghost_sparsemat *mat) 
{
    //for KACZ_ANALYZE
    #ifdef GHOST_KACZ_ANALYZE 
    //ghost_lidx line_size, n_lines, rem_lines;
    //int start=0 , end=0;
    //ghost_lidx *rows;
    //ghost_lidx *nnz;
    #endif
    
    int height = SPM_NROWS(mat);
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
    
    
    //ghost_lidx *chunk_ptr = mat->chunkStart;
    //ghost_lidx *col_ptr = mat->col;
    //ghost_lidx chunkheight = mat->traits.C;
    
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
        
        //ghost_lidx chunk_lower      = zone_ptr[4*i+2]/chunkheight;
        //ghost_lidx rowinchunk_lower = zone_ptr[4*i+2]%chunkheight;
        //ghost_lidx chunk_upper 	    = (zone_ptr[4*i-1]-1)/chunkheight;     
        //ghost_lidx rowinchunk_upper = (zone_ptr[4*i-1]-1)%chunkheight;
        
        int *extrema_lower_trans = NULL, *extrema_upper_trans = NULL;
        
        find_zone_extrema(mat, &extrema_lower_trans, zone_ptr[4*i+2], zone_ptr[4*i+3]);
        ghost_lidx lower = extrema_lower_trans[MIN_LOWER]; 
        
        find_zone_extrema(mat, &extrema_upper_trans, zone_ptr[4*i-2], zone_ptr[4*i-1]);
        ghost_lidx upper = extrema_lower_trans[MAX_UPPER]; 
        
        
        /*ghost_lidx lower = virtual_col(chunk_ptr[chunk_lower]+rowinchunk_lower); 	// TODO : A scanning might be required	
         * ghost_lidx upper = virtual_col(chunk_ptr[chunk_upper]+rowinchunk_upper+chunkheight*(mat->chunkLen[chunk_upper]-1));
         */
        if(lower <= upper) {
            //printf("check lower = %d and upper =%d\n",virtual_col(row_ptr[zone_ptr[4*i+2]]) , virtual_col(row_ptr[zone_ptr[4*i-1]]-1));
            mat->kacz_setting.kacz_method = GHOST_KACZ_METHOD_BMC_two_sweep;	
            WARNING_LOG("ONLY half the available threads would be used for transitional sweep\n");
            break;
        }
        
        
        if(extrema_lower_trans != NULL)
            free(extrema_lower_trans);
        
        if(extrema_upper_trans != NULL)
            free(extrema_upper_trans);
        
    }
    #ifdef GHOST_KACZ_ANALYZE 
    kacz_analyze_print(mat);
    #endif
    
    goto out;
    
    err:
    
    out:
    
    return ret;
}

//not used - expensive, eventhough it refines in detail, it might then lead to load balancing
/*ghost_error split_transition(ghost_sparsemat *mat) 
 * {
 *     ghost_error ret = GHOST_SUCCESS;
 * 
 *     //for SELL-1-1
 *     ghost_lidx *row_ptr = mat->chunkStart;
 *     ghost_lidx *col_ptr = mat->col;
 *           
 *      int nthread[1];
 * #ifdef GHOST_HAVE_OPENMP
 * #pragma omp parallel
 *   {
 *     #pragma omp master
 *     nthread[0] = ghost_omp_nthread();
 *   }
 * #else
 *    nthread[0] = 1;
 * #endif
 * 
 * 
 *   int n_zones = nthread[0];
 *   ghost_lidx *new_zone_ptr = NULL;
 * 
 *   mat->kacz_setting.active_threads = nthread[0];//TODO add this to sparsemat
 *  
 *   mat->nzones = mat->nzones + 2*(n_zones-1);//add the new zones
 *  
 *   GHOST_CALL_GOTO(ghost_malloc((void **)&new_zone_ptr,sizeof(ghost_lidx)*(4*n_zones-2)),err,ret);
 * 
 *  for (int i=0; i<n_zones; ++i) {
 *        new_zone_ptr[4*i] = mat->zone_ptr[2*i];
 *        new_zone_ptr[4*i+1] = mat->zone_ptr[2*i+1];
 *  }
 * 
 * 
 * 
 *  for (int i=0; i<n_zones-1; ++i) {
 *  	ghost_lidx red_start       = new_zone_ptr[4*i+1];
 *        ghost_lidx red_end         = -1;
 *        ghost_lidx black_start     = -1;
 *        ghost_lidx black_end       = new_zone_ptr[4*i+4];
 * 
 *        int *extrema_pre, *extrema_post;
 * 
 *        find_zone_extrema(mat, &extrema_pre,  new_zone_ptr[4*i],   new_zone_ptr[4*i+1]);
 *        find_zone_extrema(mat, &extrema_post, new_zone_ptr[4*i+4], new_zone_ptr[4*i+5]); 
 * 
 *        ghost_lidx red_end_col     = extrema_pre[MIN_UPPER];//col_ptr[row_ptr[new_zone_ptr[4*i]+1]-1];
 *        ghost_lidx black_start_col = extrema_post[MAX_LOWER];//col_ptr[row_ptr[new_zone_ptr[4*i+5]]];
 * 
 *        bool flag_red   = 0;
 *        bool flag_black = 0;
 * 
 *       //now find corresponding rows of red_end and black_start
 *        for (int j=red_start; j<black_end; ++j) {
 *        	if(flag_red == 0 || flag_black == 0) {
 * 			if(flag_red == 0 && col_ptr[row_ptr[j]] >= red_end_col) {
 *                        	red_end  = j;
 *                                flag_red = 1;
 *                        }
 *                        if(flag_black == 0 && col_ptr[row_ptr[j+1]-1] >= black_start_col) {
 *                                black_start  = j;
 *                                flag_black = 1;
 *                        }
 *                 }
 *        }
 * 
 *       free(extrema_pre);
 *       free(extrema_post);
 *  
 *       //now check the cases 
 *       if(flag_red == 0 && flag_black == 0) {
 *  		new_zone_ptr[4*i+2] = black_end;
 *                new_zone_ptr[4*i+3] = black_end;
 *       }
 *       else if(flag_red == 0 && flag_black == 1) {
 *                new_zone_ptr[4*i+2] = black_start;
 *                new_zone_ptr[4*i+3] = black_start;
 *       }
 *       else if(flag_red == 1 && flag_black == 0) {
 *                new_zone_ptr[4*i+2] = red_end;
 *                new_zone_ptr[4*i+3] = red_end;
 *       }
 *       //this is the only case where we have transition zones
 *       else if(black_start < red_end) {
 *                new_zone_ptr[4*i+2] = black_start;
 *                new_zone_ptr[4*i+3] = red_end;
 *       }
 *       else if(black_start >= red_end) {
 *                int median = (int)( (black_start+red_end)/2.0 );
 *                new_zone_ptr[4*i+2] = median;
 *                new_zone_ptr[4*i+3] = median;
 *       }
 *       else {
 *	       EROOR_LOG("GHOST UNKNOWN ERROR in hybrid splitting \n");
 *               ret = GHOST_ERR_UNKNOWN;
 *               goto err;
 *      }
 *    
 *   }  
 * 
 *   //now check whether the transition in transition is overlapping- if one region overlaps we use 2 sweep method (with threads/2) , else one sweep method 
 *   for(int i=1; i<n_zones-1; ++i) {
 *	mat->kacz_setting.kacz_method = GHOST_KACZ_METHOD_BMC_one_sweep;
 * 
 *       if(col_ptr[row_ptr[new_zone_ptr[4*i+2]]] <= col_ptr[row_ptr[new_zone_ptr[[4*i-1]]]-1]) {
 *           mat->kacz_setting.kacz_method = GHOST_KACZ_METHOD_BMC_two_sweep;	
 *           break;
 *       }
 *  }
 * 
 * 
 *      mat->zone_ptr = new_zone_ptr;
 *      goto out;
 * 
 * err:
 * 
 * out:
 * return ret;
 *      
 * }*/               
