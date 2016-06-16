#include "ghost/kacz_hybrid_split.h"

typedef enum {
 MIN_LOWER = 0,
 MAX_LOWER = 1,
 MIN_UPPER = 2,
 MAX_UPPER = 3
}zone_extrema;

//returns the virtual column index; ie takes into account the permutation of halo elements also
#define virtual_col(col_idx)\
   (mat->context->flags & GHOST_PERM_NO_DISTINCTION)?( (col_ptr[col_idx]<mat->context->nrowspadded)?col_ptr[col_idx]:mat->context->perm_local->colPerm[col_ptr[col_idx]] ):col_ptr[col_idx]\



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

//This check is not necessary since this should not fail, if implemented correctly
ghost_error checker(ghost_sparsemat *mat)
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
	    printf("check  lower = %d, upper = %d\n",virtual_col(row_ptr[zones[4*i+3]]),virtual_col(row_ptr[zones[4*i]]-1) );
            ret = GHOST_ERR_BLOCKCOLOR;
	    printf("ERR 3\n");
//	    break;
	}
		//check transition in transition zones, if we are using one sweep method, 
	if(mat->kacz_setting.kacz_method == BMC_one_sweep) {
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

//finds lower and upper bandwidth of the matrix
ghost_error mat_bandwidth(ghost_sparsemat *mat, int *lower_bw, int *upper_bw, int a, int b)
{
  int lower = 0;
  int upper = 0;
  ghost_lidx* row_ptr = mat->sell->chunkStart; 
  ghost_lidx* col_ptr = mat->sell->col;//TODO give virtual colums
  int start_col, end_col;

  //std::cout<<"nrows ="<<mat->nrows<<std::endl;
  //std::cout<<"check"<<row_ptr[mat->nrows-1]<<std::endl;

   for(int i=a; i<b; ++i){
	   start_col = mat->nrows + mat->context->nrowspadded;
	   end_col   = 0;
	for(int j=row_ptr[i]; j<row_ptr[i+1]; ++j) {
           start_col = MIN(start_col, virtual_col(j));
           end_col = MAX(end_col, virtual_col(j));
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

    ghost_lidx *row_ptr = mat->sell->chunkStart;
    ghost_lidx *col_ptr = mat->sell->col;


   //height might vary from nrows if we have multicoloring
   ghost_lidx height = mat->zone_ptr[mat->nzones];
   //width might vary from ncols  if we consider remote permutations also
   ghost_lidx width  = mat->maxColRange+1;


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
   double diagonal_slope = (double)(height)/width;
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

   //now for last zone we have only black-start
   /*ghost_lidx black_start = new_zone_ptr[4*(n_zones-1)] + total_bw;
   ghost_lidx red_end     = mat->nrows+1 - total_bw;

   new_zone_ptr[4*(n_zones-1)+2] = black_start;  //no transition
   new_zone_ptr[4*(n_zones-1)+3] = black_start;  //no transition
   */
 
   //now check whether the transition in transition is overlapping- if one region overlaps we use 2 sweep method (with threads/2) , else one sweep method 
   mat->kacz_setting.kacz_method = BMC_one_sweep;
 
   for(int i=1; i<n_zones; ++i) {
	ghost_gidx lower = virtual_col(row_ptr[new_zone_ptr[4*i+2]]);
	ghost_gidx upper = virtual_col(row_ptr[new_zone_ptr[4*i-1]]-1);

       if(lower <= upper) {
           //printf("check lower = %d and upper =%d\n",virtual_col(row_ptr[new_zone_ptr[4*i+2]]) , virtual_col(row_ptr[new_zone_ptr[4*i-1]]-1));
           mat->kacz_setting.kacz_method = BMC_two_sweep;	
           WARNING_LOG("ONLY half the available threads would be used for transitional sweep\n");
           break;
       }
  }


#ifdef GHOST_KACZ_ANALYZE 

 ghost_lidx line_size = 12;
 ghost_lidx n_lines = mat->kacz_setting.active_threads / line_size;
 ghost_lidx rem_lines =  mat->kacz_setting.active_threads % line_size;
 int start=0 , end=0;
 
 printf("%10s:","THREADS");
                          
 for(int line=0; line<n_lines; ++line) {
 	start = line*line_size;
        end   = (line+1)*line_size;

	for(int i=start ; i<end; ++i){
  		printf("|%10d",i+1);	
   	}
        printf("\n");
        printf("%10s:","");
  }
 
  start = mat->kacz_setting.active_threads - rem_lines;
  end   = mat->kacz_setting.active_threads;

  for(int i=start ; i<end; ++i){
         printf("|%10d",i+1);
  }

 printf("|%10s","TOTAL");

 const char *zone_name[4];
 zone_name[0] = "PURE ZONE";
 zone_name[1] = "RED TRANS ZONE";
 zone_name[2] = "TRANS IN TRANS ZONE";
 zone_name[3] = "BLACK TRANS ZONE";

 ghost_lidx rows[mat->kacz_setting.active_threads];
 ghost_lidx nnz[mat->kacz_setting.active_threads];

 #ifdef GHOST_HAVE_OPENMP
	#pragma omp parallel shared(line_size)
	 {
 #endif
         ghost_lidx tid = ghost_omp_threadnum();

         for(ghost_lidx zone=0; zone<4; ++zone) {
         	rows[tid] = new_zone_ptr[4*tid+zone+1] - new_zone_ptr[4*tid+zone];
                nnz[tid]  = 0;
        
                if(rows[tid]!=0) {
			for(int j=new_zone_ptr[4*tid+zone]; j<new_zone_ptr[4*tid+zone+1]; ++j) {
                		nnz[tid] += row_ptr[j+1] - row_ptr[j] ;   
        		}
                }

          	#pragma omp barrier
      
          	#pragma omp single
          	{
                	 printf("\n\n%s\n",zone_name[zone]);
			 printf("%10s:","ROWS");
  			 ghost_lidx ctr = 0;
                         ghost_lidx n_lines = mat->kacz_setting.active_threads / line_size;
                         ghost_lidx rem_lines =  mat->kacz_setting.active_threads % line_size;
                         int start=0 , end=0;
                         
                         for(int line=0; line<n_lines; ++line) {
                                start = line*line_size;
                                end   = (line+1)*line_size;

				for(int i=start ; i<end; ++i){
  					printf("|%10d",rows[i]);
   					ctr += rows[i];
   				}
                         printf("\n");
                         printf("%10s:","");
                        }
 
			start = mat->kacz_setting.active_threads - rem_lines;
  			end   = mat->kacz_setting.active_threads;

                          for(int i=start ; i<end; ++i){
                                  printf("|%10d",rows[i]);
                                  ctr += rows[i];
                         }

  			printf("|%10d",ctr);
  			printf("\n%10s:","%");
 	
			if(ctr!=0) {

	                         for(int line=0; line<n_lines; ++line) {
        	                        start = line*line_size;
                	                end   = (line+1)*line_size;

                        	        for(int i=start ; i<end; ++i){
                                	        printf("|%10d",(int)(((double)rows[i]/ctr)*100));
                                	}
                         	 printf("\n");
                         	 printf("%10s:","");
                        	}
			
				start = mat->kacz_setting.active_threads - rem_lines;
  				end   = mat->kacz_setting.active_threads;


		                for(int i=start ; i<end; ++i){
                                	printf("|%10d",(int)(((double)rows[i]/ctr)*100));          
                                } 
 				printf("|%10d",100);
  			}

 		
			printf("\n%10s:","NNZ");
 			ctr = 0;
                       
			for(int line=0; line<n_lines; ++line) {
                                start = line*line_size;
                                end   = (line+1)*line_size;

				for(int i=start ; i<end; ++i){
  					printf("|%10d",nnz[i]);
   					ctr += nnz[i];
   				}
                         printf("\n");
                         printf("%10s:","");

                         }
 	         	 
		  	 start = mat->kacz_setting.active_threads - rem_lines;
  			 end   = mat->kacz_setting.active_threads;

                        for(int i=start ; i<end; ++i){
                                  printf("|%10d",nnz[i]);
                                  ctr += nnz[i];
                         }

        		printf("|%10d",ctr);
			printf("\n%10s:","%");

			if(ctr!=0) {
 
	                         for(int line=0; line<n_lines; ++line) {
        	                        start = line*line_size;
                	                end   = (line+1)*line_size;

                        	        for(int i=start ; i<end; ++i){
                                	        printf("|%10d",(int)(((double)nnz[i]/ctr)*100));
                                	}
                         	 printf("\n");
                         	 printf("%10s:","");
                        	}

				start = mat->kacz_setting.active_threads - rem_lines;
  				end   = mat->kacz_setting.active_threads;

                        	for(int i=start ; i<end; ++i){
                                	printf("|%10d",(int)(((double)nnz[i]/ctr)*100));          
                                } 
				printf("|%10d",100);
			 }


          	}
          }
  #ifdef GHOST_HAVE_OPENMP
   	  }
  #endif
  printf("\n\n");
 #endif
 
 mat->zone_ptr = new_zone_ptr;
 goto out;
 
 err:

 out :

 return ret;
}

//not used - expensive, eventhough it refines in detail, it might then lead to load balancing
/*ghost_error split_transition(ghost_sparsemat *mat) 
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

   mat->kacz_setting.active_threads = nthread[0];//TODO add this to sparsemat
  
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
 			if(flag_red == 0 && col_ptr[row_ptr[j]] >= red_end_col) {
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

   //now check whether the transition in transition is overlapping- if one region overlaps we use 2 sweep method (with threads/2) , else one sweep method 
   for(int i=1; i<n_zones-1; ++i) {
	mat->kacz_setting.kacz_method = BMC_one_sweep;

       if(col_ptr[row_ptr[new_zone_ptr[4*i+2]]] <= col_ptr[row_ptr[new_zone_ptr[[4*i-1]]]-1]) {
           mat->kacz_setting.kacz_method = BMC_two_sweep;	
           break;
       }
  }


      mat->zone_ptr = new_zone_ptr;
      goto out;
 
err:

out:
 return ret;
      
}*/               
