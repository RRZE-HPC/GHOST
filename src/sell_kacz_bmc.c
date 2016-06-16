#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/omp.h"
#include "ghost/sell_kacz_rb.h"
#include <omp.h>
#include <math.h>

//this is necessary since #pragma omp for doesn't understand !=
#define FORWARD_LOOP(start,end)                                         \
  for (ghost_lidx row=start; row<end; ++row){                   \
         double rownorm = 0.;                                          \
         double scal = 0;                                              \
	 ghost_lidx  idx = sellmat->chunkStart[row];                   \
                                                                       \
         if(bval != NULL)                                              \
          scal  = -bval[row];                                          \
        for (ghost_lidx j=0; j<sellmat->rowLen[row]; ++j) {            \
                 scal += (double)mval[idx] * xval[sellmat->col[idx]];  \
                if(opts.normalize==no)                                 \
                 rownorm += mval[idx]*mval[idx];                       \
                 idx += 1;                                             \
          }                                                            \
        if(opts.normalize==no){                                        \
         scal /= (double)rownorm;                                      \
        }                                                              \
        scal *= omega;                                                 \
	idx -= sellmat->rowLen[row];                                   \
                                                                       \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<sellmat->rowLen[row]; j++) {           \
		xval[sellmat->col[idx]] = xval[sellmat->col[idx]] - scal * (double)mval[idx];\
                idx += 1;                                              \
          }                                                            \
      }                                                                \

#define BACKWARD_LOOP(start,end)                                         \
  for (ghost_lidx row=start; row>end; --row){                   \
         double rownorm = 0.;                                          \
         double scal = 0;                                              \
	 ghost_lidx  idx = sellmat->chunkStart[row];                   \
                                                                       \
         if(bval != NULL)                                              \
          scal  = -bval[row];                                          \
        for (ghost_lidx j=0; j<sellmat->rowLen[row]; ++j) {            \
                 scal += (double)mval[idx] * xval[sellmat->col[idx]];  \
                if(opts.normalize==no)                                 \
                 rownorm += mval[idx]*mval[idx];                       \
                 idx += 1;                                             \
          }                                                            \
        if(opts.normalize==no){                                        \
         scal /= (double)rownorm;                                      \
        }                                                              \
        scal *= omega;                                                 \
	idx -= sellmat->rowLen[row];                                   \
                                                                       \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<sellmat->rowLen[row]; j++) {           \
		xval[sellmat->col[idx]] = xval[sellmat->col[idx]] - scal * (double)mval[idx];\
                idx += 1;                                              \
          }                                                            \
      }                                                                \


#define LOOP(start,end,stride)                                         \
  for (ghost_lidx row=start; row!=end; row+=stride){                   \
         double rownorm = 0.;                                          \
         double scal = 0;                                              \
	 ghost_lidx  idx = sellmat->chunkStart[row];                   \
                                                                       \
         if(bval != NULL)                                              \
          scal  = -bval[row];                                          \
        for (ghost_lidx j=0; j<sellmat->rowLen[row]; ++j) {            \
                 scal += (double)mval[idx] * xval[sellmat->col[idx]];  \
                if(opts.normalize==no)                                 \
                 rownorm += mval[idx]*mval[idx];                       \
                 idx += 1;                                             \
          }                                                            \
        if(opts.normalize==no){                                        \
         scal /= (double)rownorm;                                      \
        }                                                              \
        scal *= omega;                                                 \
	idx -= sellmat->rowLen[row];                                   \
                                                                       \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<sellmat->rowLen[row]; j++) {           \
		xval[sellmat->col[idx]] = xval[sellmat->col[idx]] - scal * (double)mval[idx];\
                idx += 1;                                              \
          }                                                            \
      }                                                                \

#define LOCK_NEIGHBOUR(tid)						       \
	if(tid == 0)						       \
        	flag[0] = zone+1;        			       \
        if(tid == nthreads-1)					       \
        	flag[nthreads+1] = zone+1;			       \
        							       \
   	flag[tid+1] = zone+1;   				       \
 	_Pragma("omp flush")       				       \
								       \
    	if(opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {	       \
		while(flag[tid+2]<zone+1){			       \
       			_Pragma("omp flush")       		       \
        	}						       \
     	} else {						       \
        	 while(flag[tid]<zone+1 ){			       \
                        _Pragma("omp flush")     		       \
      	       	 } 						       \
     	}    


ghost_error ghost_initialize_kacz_bmc(ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
{
    ghost_sell *sellmat = SELL(mat);
    double *mval = (double *)sellmat->val;
    double *bval = (double *)(b->val);
    double rownorm = 0;
    ghost_lidx idx;

  //normalize if necessary
   if(opts.normalize == yes) {
       for(int row=0; row < mat->nrows; ++row) {
           rownorm = 0;
           idx =  sellmat->chunkStart[row];
           for (int j=0; j<sellmat->rowLen[row]; ++j) {
             rownorm += mval[idx]*mval[idx];
	     idx += 1;
           }
           
           bval[row] = (double)(bval[row])/rownorm;

          idx =  sellmat->chunkStart[row];
          for (int j=0; j<sellmat->rowLen[row]; ++j) {
             mval[idx] = (double)(mval[idx])/sqrt(rownorm);
	     idx += 1;
           }
        }
    }     
  return GHOST_SUCCESS; 
}

ghost_error ghost_kacz_bmc(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    //int flag_err = 0;
    //currently only implementation for SELL-1-1
    //const int CHUNKHEIGHT = 1;  
    const int NVECS = 1;

    //TODO check for RCM and give a Warning
    if (mat->nzones == 0 || mat->zone_ptr == NULL){
        ERROR_LOG("Splitting of matrix by Block Multicoloring  has not be done!");
    }
  
   if (NVECS > 1) {
        ERROR_LOG("Multi-vec not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
   
    ghost_sell *sellmat = SELL(mat); 
    double *bval = NULL;
  
   if(b!= NULL)
     bval = (double *)(b->val);

    double *xval = (double *)(x->val);
    double *mval = (double *)sellmat->val;
    double omega = *(double *)opts.omega;
    ghost_lidx *zone_ptr = (ghost_lidx*) mat->zone_ptr;
    //ghost_lidx nzones    = mat->nzones;
    ghost_lidx *color_ptr= (ghost_lidx*) mat->color_ptr;
    //ghost_lidx ncolors   = mat->ncolors;
    ghost_lidx nthreads  = mat->kacz_setting.active_threads;

   // disables dynamic thread adjustments 
    ghost_omp_set_dynamic(0);
    ghost_omp_nthread_set(nthreads);
    //printf("Setting number of threads to %d for KACZ sweep\n",nthreads);

    ghost_lidx *flag;
    flag = (ghost_lidx*) malloc((nthreads+2)*sizeof(ghost_lidx));

    for (int i=0; i<nthreads+2; i++) {
        flag[i] = 0;
    }

    //always execute first and last blocks
    flag[0] = 1;         
    flag[nthreads+1] = 1;

   //Do multicolored rows, if in backward direction
    int mc_start, mc_end;

    ghost_lidx stride = 0;

    if (opts.direction == GHOST_KACZ_DIRECTION_BACKWARD) {
            //for time being single thread
	for(int i=mat->ncolors; i>0; --i) {
            mc_start  = color_ptr[i]-1;
            mc_end    = color_ptr[i-1]-1;
            stride   = -1;

#ifdef GHOST_HAVE_OPENMP  
	#pragma omp parallel for 
#endif
            BACKWARD_LOOP(mc_start,mc_end)
	}
    } 
   
#ifdef GHOST_HAVE_OPENMP  
#pragma omp parallel private(stride)
{
#endif
    ghost_lidx start[4];
    ghost_lidx end[4];


    ghost_lidx tid = ghost_omp_threadnum();

 //   ghost_lidx stride     = 0;

    if (opts.direction == GHOST_KACZ_DIRECTION_BACKWARD) {
            start[0]  = zone_ptr[4*tid+4]-1; 
	    end[0]    = zone_ptr[4*tid+3]-1;
            start[1]  = zone_ptr[4*tid+3]-1;
    	    end[1]    = zone_ptr[4*tid+2]-1;
            start[2]  = zone_ptr[4*tid+2]-1;
            end[2]    = zone_ptr[4*tid+1]-1;
            start[3]  = zone_ptr[4*tid+1]-1;
            end[3]    = zone_ptr[4*tid]  -1;
            stride    = -1;
    } else {
            start[0]  = zone_ptr[4*tid];
            end[0]    = zone_ptr[4*tid+1];
            start[1]  = zone_ptr[4*tid+1];
            end[1]    = zone_ptr[4*tid+2];
	    start[2]  = zone_ptr[4*tid+2];
            end[2]    = zone_ptr[4*tid+3];
            start[3]  = zone_ptr[4*tid+3];
            end[3]    = zone_ptr[4*tid+4];
            stride    = 1;
    }

    //double rownorm = 0.;

 if(mat->kacz_setting.kacz_method == BMC_one_sweep) { 
    for(ghost_lidx zone = 0; zone<4; ++zone) { 

            LOOP(start[zone],end[zone],stride)   
                   
   	      #pragma omp barrier                           
 
	      //TODO A more efficient way of locking is necessary, normal locks makes problem if the matrix size is small
	      //but this explicit flush method is expensive than barrier
	
/*              if(tid == 0)
                  flag[0] = zone+1;
         
               if(tid == nthreads-1)
                  flag[nthreads+1] = zone+1;
                 
            
    		flag[tid+1] = zone+1; 
    		#pragma omp flush      

    		if(opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
			 while(flag[tid+2]<zone+1){
                        // printf("Forward: thread %d spinning, my flag is %d, neighbour flag is %d and zone = %d\n",tid, flag[tid+1], flag[tid+2],zone+1);
                         
        	         #pragma omp flush
         	         }
     		} else {
        		 while(flag[tid]<zone+1 ){
                         //printf("Backward: thread %d spinning, flag is %d\n",tid, flag[tid]);

          	 	 #pragma omp flush
      	       	         } 
     		}    
  
*/
  	//	LOCK_NEIGHBOUR(tid)
   }
 } else if (mat->kacz_setting.kacz_method == BMC_two_sweep) {
//TODO remove barriers its for testing 
     LOOP(start[0],end[0],stride)
      #pragma omp barrier 
      
      if(opts.direction == GHOST_KACZ_DIRECTION_BACKWARD) {
      	if(tid%2 != 0) {
            LOOP(start[1],end[1],stride)
        } 
       #pragma omp barrier
       if(tid%2 == 0) {
            LOOP(start[1],end[1],stride)
        }
      } else {
      LOOP(start[1],end[1],stride) 
      }
      #pragma omp barrier
     
      if(opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
      	if(tid%2 == 0) {
            LOOP(start[2],end[2],stride)
        } 
       #pragma omp barrier 
       if(tid%2 != 0) {
            LOOP(start[2],end[2],stride)
        }
      } else {
      LOOP(start[2],end[2],stride)
      }

     #pragma omp barrier 
     LOOP(start[3],end[3],stride)               
  }      
#ifdef GHOST_HAVE_OPENMP
  }
#endif

//do multicoloring if in FORWARD direction
   if (opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {

	for(int i=0; i<mat->ncolors; ++i) {
           //for time being single thread
            mc_start  = color_ptr[i];
            mc_end    = color_ptr[i+1];
            stride    = 1;

#ifdef GHOST_HAVE_OPENMP
	#pragma omp parallel for
#endif
            FORWARD_LOOP(mc_start,mc_end)
	}
    }

    free(flag);	    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

//dont use
ghost_error ghost_kacz_bmc_with_shift(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, double *shift_r,  ghost_kacz_opts opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
   
    //currently only implementation for SELL-1-1
    //const int CHUNKHEIGHT = 1;  
    const int NVECS = 1;

    //TODO check for RCM and give a Warning
    if (mat->nzones == 0 || mat->zone_ptr == NULL){
        ERROR_LOG("Splitting of matrix to Red and Black ( odd and even) have not be done!");
    }
  
   if (NVECS > 1) {
        ERROR_LOG("Multi-vec not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

   
    ghost_sell *sellmat = SELL(mat); 
    double *bval = NULL;
  
   if(b!= NULL)
     bval = (double *)(b->val);

    double *xval = (double *)(x->val);
    double *mval = (double *)sellmat->val;
    double omega = *(double *)opts.omega;
    ghost_lidx *zone_ptr = mat->zone_ptr;
    ghost_lidx nzones    = mat->nzones;
    ghost_lidx nthreads  = nzones/2;

   // disables dynamic thread adjustments 
    ghost_omp_set_dynamic(0);
    ghost_omp_nthread_set(nthreads);
    //printf("Setting number of threads to %d for KACZ sweep\n",nthreads);

    omp_lock_t* execute;
    execute = (omp_lock_t*) malloc(nthreads*sizeof(omp_lock_t));
    
/*    for (int i=0; i<nthreads; i++) {
        omp_init_lock(&execute[i]);
    }
*/
    
#ifdef GHOST_HAVE_OPENMP  
#pragma omp parallel
{
#endif
    ghost_lidx tid = ghost_omp_threadnum();
    ghost_lidx even_start = zone_ptr[2*tid];
    ghost_lidx even_end   = zone_ptr[2*tid+1];
    ghost_lidx odd_start  = zone_ptr[2*tid+1];
    ghost_lidx odd_end    = zone_ptr[2*tid+2];
    ghost_lidx stride     = 0;

    if (opts.direction == GHOST_KACZ_DIRECTION_BACKWARD) {
            even_start   = zone_ptr[2*tid+2]-1; 
	    even_end     = zone_ptr[2*tid+1]-1;
            odd_start    = zone_ptr[2*tid+1]-1;
    	    odd_end      = zone_ptr[2*tid]-1;
            stride       = -1;
     } else {
            even_start = zone_ptr[2*tid];
            even_end   = zone_ptr[2*tid+1];
            odd_start  = zone_ptr[2*tid+1];
            odd_end    = zone_ptr[2*tid+2];
            stride     = 1;
     }
  
     //only for SELL-1-1 currently
     //even sweep
 //    omp_set_lock(&execute[tid]);
     double rownorm = 0.;
    
      for (ghost_lidx row=even_start; row!=even_end; row+=stride){
  	 //printf("projecting to row ........ %d\n",row); 
         rownorm = 0.;
	 ghost_lidx  idx = sellmat->chunkStart[row];

	 double scal = 0;

         if(bval != NULL)
          scal  = -bval[row];
        
        for (ghost_lidx j=0; j<sellmat->rowLen[row]; ++j) {
                 scal += (double)mval[idx] * xval[sellmat->col[idx]];
                if(opts.normalize==no)
                 rownorm += mval[idx]*mval[idx]; 
                 idx += 1;
          }
        scal -= (*shift_r) * xval[row];

        if(opts.normalize==no) {
          rownorm += (*shift_r) * (*shift_r);
          scal /= (double)rownorm;
         }

        scal *= omega;

	idx -= sellmat->rowLen[row];

 	#pragma simd vectorlength(4)
         for (ghost_lidx j=0; j<sellmat->rowLen[row]; j++) {
		xval[sellmat->col[idx]] = xval[sellmat->col[idx]] - scal * (double)mval[idx];
                idx += 1;
          }
        xval[row] = xval[row] + scal * (*shift_r);
       
      }
       
//    omp_unset_lock(&execute[tid]);
 #pragma omp barrier 
 //odd sweep
 /*   if(tid+1 < nthreads) {
         omp_set_lock(&execute[tid+1]);
    }*/

     for (ghost_lidx row=odd_start; row!=odd_end; row+=stride){
         //printf("projecting to row ........ %d\n",row);
         rownorm = 0.; 
         ghost_lidx idx = sellmat->chunkStart[row];
         double scal = 0;
  
         if(bval != NULL)
          scal  = -bval[row];
 
         for (ghost_lidx j=0; j<sellmat->rowLen[row]; ++j) {
                 scal += (double)mval[idx] * xval[sellmat->col[idx]];
                if(opts.normalize==no)
                 rownorm += mval[idx]*mval[idx];
                 idx += 1;
          }
        scal -= (*shift_r) * xval[row];  

        if(opts.normalize==no) {
         rownorm += (*shift_r) * (*shift_r);
         scal /= (double)rownorm;
         }

        scal *= omega;
	
 	idx -= sellmat->rowLen[row];

     #pragma simd vectorlength(4)
         for (ghost_lidx j=0; j<sellmat->rowLen[row]; j++) {
                xval[sellmat->col[idx]] = xval[sellmat->col[idx]]  - scal * (double)mval[idx];
                idx += 1;
          }      
         xval[row] = xval[row] + scal * (*shift_r);
      }

 /*   if(tid+1 < nthreads) {
         omp_unset_lock(&execute[tid+1]);
    }*/
  #pragma omp barrier    
#ifdef GHOST_HAVE_OPENMP
  }
#endif
    for (int i=0; i<nthreads; i++) {
        omp_destroy_lock(&execute[i]);
    }

    free(execute);
   	    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}


