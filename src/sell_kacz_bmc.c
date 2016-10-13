#include "ghost/sell_kacz_bmc.h"

#define NVECS 1
#define CHUNKHEIGHT 1

#if (NVECS==1 && CHUNKHEIGHT==1)
//this is necessary since #pragma omp for doesn't understand !=
#define FORWARD_LOOP(start,end)                                        \
  for (ghost_lidx row=start; row<end; ++row){                          \
         double rownorm = 0.;                                          \
         double scal = 0;                                              \
	 ghost_lidx  idx = mat->chunkStart[row];                   \
                                                                       \
         if(bval != NULL)                                              \
          scal  = -bval[row];                                          \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            \
                 scal += (double)mval[idx] * xval[mat->col[idx]];  \
                if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                                 \
                 rownorm += mval[idx]*mval[idx];                       \
                 idx += 1;                                             \
          }                                                            \
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){                                        \
         scal /= (double)rownorm;                                      \
        }                                                              \
        scal *= omega;                                                 \
	idx -= mat->rowLen[row];                                   \
                                                                       \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           \
		xval[mat->col[idx]] = xval[mat->col[idx]] - scal * (double)mval[idx];\
                idx += 1;                                              \
          }                                                            \
      }                                                                \

#define BACKWARD_LOOP(start,end)                                       \
  for (ghost_lidx row=start; row>end; --row){                          \
         double rownorm = 0.;                                          \
         double scal = 0;                                              \
	 ghost_lidx  idx = mat->chunkStart[row];                   \
                                                                       \
         if(bval != NULL)                                              \
          scal  = -bval[row];                                          \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            \
                 scal += (double)mval[idx] * xval[mat->col[idx]];  \
                if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                                 \
                 rownorm += mval[idx]*mval[idx];                       \
                 idx += 1;                                             \
          }                                                            \
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){                                        \
         scal /= (double)rownorm;                                      \
        }                                                              \
        scal *= omega;                                                 \
	idx -= mat->rowLen[row];                                   \
                                                                       \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           \
		xval[mat->col[idx]] = xval[mat->col[idx]] - scal * (double)mval[idx];\
                idx += 1;                                              \
          }                                                            \
      }                                                                \


#define LOOP(start,end,stride)                                         \
  for (ghost_lidx row=start; row!=end; row+=stride){                   \
         double rownorm = 0.;                                          \
         double scal = 0;                                              \
	 ghost_lidx  idx = mat->chunkStart[row];                   \
                                                                       \
         if(bval != NULL)                                              \
          scal  = -bval[row];                                          \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            \
                 scal += (double)mval[idx] * xval[mat->col[idx]];  \
                if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                                 \
                 rownorm += mval[idx]*mval[idx];                       \
                 idx += 1;                                             \
          }                                                            \
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){                                        \
         scal /= (double)rownorm;                                      \
        }                                                              \
        scal *= omega;                                                 \
	idx -= mat->rowLen[row];                                   \
                                                                       \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           \
		xval[mat->col[idx]] = xval[mat->col[idx]] - scal * (double)mval[idx];\
                idx += 1;                                              \
          }                                                            \
      }                                                                \

#else 

#define FORWARD_LOOP(start,end)               	                     						\
 start_rem   = start%CHUNKHEIGHT;										\
 start_chunk = start/CHUNKHEIGHT+1;										\
 end_chunk   = end/CHUNKHEIGHT;											\
 end_rem     = end%CHUNKHEIGHT;											\
 chunk       = 0;												\
 rowinchunk  = 0; 												\
 idx=0, row=0;   												\
  for(rowinchunk=start_rem; rowinchunk<CHUNKHEIGHT; ++rowinchunk) {						\
     	double rownorm = 0.;                                          						\
       	double scal[NVECS] = {0};                                     						\
 	idx = mat->chunkStart[start_chunk-1] + rowinchunk;                 					\
	row = rowinchunk + (start_chunk-1)*CHUNKHEIGHT;								\
          	                                                            					\
       	if(bval != NULL) {                                            						\
       		for(int block=0; block<NVECS; ++block) {               						\
        		scal[block]  = -bval[NVECS*row+block];         						\
		}						       						\
	}							       						\
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            						\
		for(int block=0; block<NVECS; ++block) {	       						\
               		scal[block] += (double)mval[idx] * xval[NVECS*mat->col[idx]+block];  		\
		}						       						\
               	if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                         	       						\
      			rownorm += mval[idx]*mval[idx];        	       						\
		idx+=CHUNKHEIGHT;							       			\
       	}                                                               					\
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){ 				       						\
		for(int block=0; block<NVECS; ++block){                       					\
         		scal[block] /= (double)rownorm;                        					\
                	scal[block] *= omega;                                  					\
	 	}						               					\
        }                                                              						\
	idx -= CHUNKHEIGHT*mat->rowLen[row];                                   				\
                                                                       						\
 	_Pragma("simd vectorlength(4)")                                						\
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           						\
		for(int block=0; block<NVECS; ++block) {	       						\
        		xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * (double)mval[idx];\
        	}						       						\
	      idx += CHUNKHEIGHT;                                                				\
         }                                                            						\
  }														\
														\
   for (chunk=start_chunk; chunk<end_chunk; ++chunk){        							\
	for(rowinchunk=0; rowinchunk<CHUNKHEIGHT; ++rowinchunk) { 						\
         	double rownorm = 0.;                                          					\
         	double scal[NVECS] = {0};                                     					\
	 	idx = mat->chunkStart[chunk] + rowinchunk;                 					\
		row = rowinchunk + chunk*CHUNKHEIGHT;								\
           	                                                            					\
         	if(bval != NULL) {                                            					\
          		for(int block=0; block<NVECS; ++block) {               					\
          			scal[block]  = -bval[NVECS*row+block];         					\
			}						       					\
		}							       					\
        	for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            					\
			for(int block=0; block<NVECS; ++block) {	       					\
                 		scal[block] += (double)mval[idx] * xval[NVECS*mat->col[idx]+block];  	\
			}						       					\
                	if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                         	       					\
               			rownorm += mval[idx]*mval[idx];        	       					\
			idx+=CHUNKHEIGHT;							       		\
       		}                                                               				\
        	if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){ 				       					\
	 		for(int block=0; block<NVECS; ++block){                       				\
         			scal[block] /= (double)rownorm;                        				\
                		scal[block] *= omega;                                  				\
	 		}						               				\
        	}                                                              					\
		idx -= CHUNKHEIGHT*mat->rowLen[row];                                   			\
                                                                       						\
 		_Pragma("simd vectorlength(4)")                                					\
         	for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           					\
			for(int block=0; block<NVECS; ++block) {	       					\
        			xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * (double)mval[idx];\
        		}							       				\
	      	idx += CHUNKHEIGHT;                                                				\
          	}                                                            					\
      	}													\
  }                                                                						\
  for(rowinchunk=0; rowinchunk<end_rem; ++rowinchunk) {								\
     	double rownorm = 0.;                                          						\
       	double scal[NVECS] = {0};                                     						\
 	idx = mat->chunkStart[end_chunk] + rowinchunk;                 					\
	row = rowinchunk + (end_chunk)*CHUNKHEIGHT;								\
          	                                                            					\
       	if(bval != NULL) {                                            						\
       		for(int block=0; block<NVECS; ++block) {               						\
        		scal[block]  = -bval[NVECS*row+block];         						\
		}						       						\
	}							       						\
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            						\
		for(int block=0; block<NVECS; ++block) {	       						\
               		scal[block] += (double)mval[idx] * xval[NVECS*mat->col[idx]+block];  		\
		}						       						\
               	if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                         	       						\
      			rownorm += mval[idx]*mval[idx];        	       						\
		idx+=CHUNKHEIGHT;							       			\
       	}                                                               					\
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){ 				       						\
		for(int block=0; block<NVECS; ++block){                       					\
         		scal[block] /= (double)rownorm;                        					\
                	scal[block] *= omega;                                  					\
	 	}						               					\
        }                                                              						\
	idx -= CHUNKHEIGHT*mat->rowLen[row];                                   				\
                                                                       						\
 	_Pragma("simd vectorlength(4)")                                						\
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           						\
		for(int block=0; block<NVECS; ++block) {	       						\
        		xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * (double)mval[idx];\
        	}						       						\
	      idx += CHUNKHEIGHT;                                                				\
         }                                                            						\
  }														\
	
#define BACKWARD_LOOP(start,end)    					                        		\
 start_rem   = start%CHUNKHEIGHT;										\
 start_chunk = start/CHUNKHEIGHT-1;										\
 end_chunk   = end/CHUNKHEIGHT;											\
 end_rem     = end%CHUNKHEIGHT;											\
 chunk       = 0;												\
 rowinchunk  = 0; 												\
 idx=0, row=0;   												\
  for(rowinchunk=start_rem; rowinchunk>=0; --rowinchunk) {							\
     	double rownorm = 0.;                                          						\
       	double scal[NVECS] = {0};                                     						\
 	idx = mat->chunkStart[start_chunk+1] + rowinchunk;                 					\
	row = rowinchunk + (start_chunk+1)*CHUNKHEIGHT;								\
          	                                                            					\
       	if(bval != NULL) {                                            						\
       		for(int block=0; block<NVECS; ++block) {               						\
        		scal[block]  = -bval[NVECS*row+block];         						\
		}						       						\
	}							       						\
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            						\
		for(int block=0; block<NVECS; ++block) {	       						\
               		scal[block] += (double)mval[idx] * xval[NVECS*mat->col[idx]+block];  		\
		}						       						\
               	if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                         	       						\
      			rownorm += mval[idx]*mval[idx];        	       						\
		idx+=CHUNKHEIGHT;							       			\
       	}                                                               					\
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){ 				       						\
		for(int block=0; block<NVECS; ++block){                       					\
         		scal[block] /= (double)rownorm;                        					\
                	scal[block] *= omega;                                  					\
	 	}						               					\
        }                                                              						\
	idx -= CHUNKHEIGHT*mat->rowLen[row];                                   				\
                                                                       						\
 	_Pragma("simd vectorlength(4)")                                						\
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           						\
		for(int block=0; block<NVECS; ++block) {	       						\
        		xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * (double)mval[idx];\
        	}						       						\
	      idx += CHUNKHEIGHT;                                                				\
         }                                                            						\
  }														\
														\
   for (chunk=start_chunk; chunk>end_chunk; --chunk){        							\
	for(rowinchunk=CHUNKHEIGHT-1; rowinchunk>=0; --rowinchunk) { 						\
         	double rownorm = 0.;                                          					\
         	double scal[NVECS] = {0};                                     					\
	 	idx = mat->chunkStart[chunk] + rowinchunk;                 					\
		row = rowinchunk + chunk*CHUNKHEIGHT;								\
           	                                                            					\
         	if(bval != NULL) {                                            					\
          		for(int block=0; block<NVECS; ++block) {               					\
          			scal[block]  = -bval[NVECS*row+block];         					\
			}						       					\
		}							       					\
        	for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            					\
			for(int block=0; block<NVECS; ++block) {	       					\
                 		scal[block] += (double)mval[idx] * xval[NVECS*mat->col[idx]+block];  	\
			}						       					\
                	if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                         	       					\
               			rownorm += mval[idx]*mval[idx];        	       					\
			idx+=CHUNKHEIGHT;							       		\
       		}                                                               				\
        	if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){ 				       					\
	 		for(int block=0; block<NVECS; ++block){                       				\
         			scal[block] /= (double)rownorm;                        				\
                		scal[block] *= omega;                                  				\
	 		}						               				\
        	}                                                              					\
		idx -= CHUNKHEIGHT*mat->rowLen[row];                                   			\
                                                                       						\
 		_Pragma("simd vectorlength(4)")                                					\
         	for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           					\
			for(int block=0; block<NVECS; ++block) {	       					\
        			xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * (double)mval[idx];\
        		}							       				\
	      	idx += CHUNKHEIGHT;                                                				\
          	}                                                            					\
      	}													\
  }                                                                						\
  for(rowinchunk=CHUNKHEIGHT-1; rowinchunk>end_rem; --rowinchunk) {						\
     	double rownorm = 0.;                                          						\
       	double scal[NVECS] = {0};                                     						\
 	idx = mat->chunkStart[end_chunk] + rowinchunk;                 					\
	row = rowinchunk + (end_chunk)*CHUNKHEIGHT;								\
          	                                                            					\
       	if(bval != NULL) {                                            						\
       		for(int block=0; block<NVECS; ++block) {               						\
        		scal[block]  = -bval[NVECS*row+block];         						\
		}						       						\
	}							       						\
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            						\
		for(int block=0; block<NVECS; ++block) {	       						\
               		scal[block] += (double)mval[idx] * xval[NVECS*mat->col[idx]+block];  		\
		}						       						\
               	if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                         	       						\
      			rownorm += mval[idx]*mval[idx];        	       						\
		idx+=CHUNKHEIGHT;							       			\
       	}                                                               					\
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){ 				       						\
		for(int block=0; block<NVECS; ++block){                       					\
         		scal[block] /= (double)rownorm;                        					\
                	scal[block] *= omega;                                  					\
	 	}						               					\
        }                                                              						\
	idx -= CHUNKHEIGHT*mat->rowLen[row];                                   				\
                                                                       						\
 	_Pragma("simd vectorlength(4)")                                						\
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           						\
		for(int block=0; block<NVECS; ++block) {	       						\
        		xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * (double)mval[idx];\
        	}						       						\
	      idx += CHUNKHEIGHT;                                                				\
         }                                                            						\
  }														\


#endif

#define LOCK_NEIGHBOUR(tid)					       \
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


/*  for (ghost_lidx row=start; row>end; --row){                          \
         double rownorm = 0.;                                          \
         double scal[NVECS] = {0};                                     \
	 ghost_lidx  idx = mat->chunkStart[row];                   \
                                                                       \
         if(bval != NULL) {                                            \
          	for(int block=0; block<NVECS; ++block) {               \
          		scal[block]  = -bval[NVECS*row+block];         \
		}						       \
	}							       \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            \
		for(int block=0; block<NVECS; ++block) {	       \
                 	scal[block] += (double)mval[idx] * xval[NVECS*mat->col[idx]+block];  \
		}						       \
                if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                         	       \
               		rownorm += mval[idx]*mval[idx];        	       \
	idx+=1;							       \
       }                                                               \
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){ 				       \
	 for(int block=0; block<NVECS; ++block){                       \
         	scal[block] /= (double)rownorm;                        \
                scal[block] *= omega;                                  \
	 }						               \
        }                                                              \
	idx -= mat->rowLen[row];                                   \
                                                                       \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           \
		for(int block=0; block<NVECS; ++block) {	       \
        		xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * (double)mval[idx];\
        	}						       \
	      idx += 1;                                                \
          }                                                            \
      }                                                                \

#define LOOP_CHUNKHEIGHT_NVECS(start,end,stride)                                 \
  for (ghost_lidx row=start; row!=end; row+=stride){                   \
         double rownorm = 0.;                                          \
         double scal[NVECS] = {0};                                     \
	 ghost_lidx  idx = mat->chunkStart[row];                   \
                                                                       \
         if(bval != NULL) {                                            \
          	for(int block=0; block<NVECS; ++block) {               \
          		scal[block]  = -bval[NVECS*row+block];         \
		}						       \
	}							       \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            \
		for(int block=0; block<NVECS; ++block) {	       \
                 	scal[block] += (double)mval[idx] * xval[NVECS*mat->col[idx]+block];  \
		}						       \
                if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                         	       \
               		rownorm += mval[idx]*mval[idx];        	       \
	idx+=1;							       \
       }                                                               \
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){ 				       \
	 for(int block=0; block<NVECS; ++block){                       \
         	scal[block] /= (double)rownorm;                        \
                scal[block] *= omega;                                  \
	 }						               \
        }                                                              \
	idx -= mat->rowLen[row];                                   \
                                                                       \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           \
		for(int block=0; block<NVECS; ++block) {	       \
        		xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * (double)mval[idx];\
        	}						       \
	      idx += 1;                                                \
          }                                                            \
      }                                                                \

*/
ghost_error ghost_initialize_kacz_bmc(ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
{
    double *mval = (double *)mat->val;
    double *bval = (double *)(b->val);
    double rownorm = 0;
    ghost_lidx idx;

  //normalize if necessary
   if(opts.normalize == GHOST_KACZ_NORMALIZE_YES) {
       for(int row=0; row < SPM_NROWS(mat); ++row) {
           rownorm = 0;
           idx =  mat->chunkStart[row];
           for (int j=0; j<mat->rowLen[row]; ++j) {
             rownorm += mval[idx]*mval[idx];
	     idx += 1;
           }
           
           bval[row] = (double)(bval[row])/rownorm;

          idx =  mat->chunkStart[row];
          for (int j=0; j<mat->rowLen[row]; ++j) {
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
    //const int NVECS = 1;

#if CHUNKHEIGHT>1
    ghost_lidx start_rem, start_chunk, end_chunk, end_rem;									
    ghost_lidx chunk       = 0;											
    ghost_lidx rowinchunk  = 0; 											
    ghost_lidx idx=0, row=0;   											
#endif

   if (mat->context->nzones == 0 || mat->context->zone_ptr == NULL){
        ERROR_LOG("Splitting of matrix by Block Multicoloring  has not be done!");
    }
  
/*   if (NVECS > 1) {
        ERROR_LOG("Multi-vec not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
*/   
    double *bval = NULL;
  
   if(b!= NULL)
     bval = (double *)(b->val);

    double *xval = (double *)(x->val);
    double *mval = (double *)mat->val;
    double omega = *(double *)opts.omega;
    ghost_lidx *zone_ptr = (ghost_lidx*) mat->context->zone_ptr;
    //ghost_lidx nzones    = mat->context->nzones;
    ghost_lidx *color_ptr= (ghost_lidx*) mat->context->color_ptr;
    //ghost_lidx ncolors   = mat->context->ncolors;
    ghost_lidx nthreads  = mat->context->kacz_setting.active_threads;

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
    ghost_lidx stride     = 0;


    if (opts.direction == GHOST_KACZ_DIRECTION_BACKWARD) {
            //for time being single thread
	for(int i=mat->context->ncolors; i>0; --i) {
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

    stride     = 0;

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

 if(mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_one_sweep) { 
    for(ghost_lidx zone = 0; zone<4; ++zone) { 

	LOOP(start[zone],end[zone],stride);
/*  	if (opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
               FORWARD_LOOP(start[zone],end[zone])   
 	 }
        else{
	       BACKWARD_LOOP(start[zone],end[zone]) 
	}
*/
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
 } else if (mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_two_sweep) {
      LOOP(start[0],end[0],stride)
      #pragma omp barrier  
      if(opts.direction == GHOST_KACZ_DIRECTION_BACKWARD) {
      	if(tid%2 != 0) {
            BACKWARD_LOOP(start[1],end[1])
        } 
       #pragma omp barrier
       if(tid%2 == 0) {
            BACKWARD_LOOP(start[1],end[1])
        }
      } else {
      FORWARD_LOOP(start[1],end[1]) 
      }
      #pragma omp barrier
     
      if(opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
      	if(tid%2 == 0) {
            FORWARD_LOOP(start[2],end[2])
        } 
       #pragma omp barrier 
       if(tid%2 != 0) {
            FORWARD_LOOP(start[2],end[2])
        }
      } else {
      BACKWARD_LOOP(start[2],end[2])
      }
     #pragma omp barrier
     LOOP(start[3],end[3],stride)   
  }      
#ifdef GHOST_HAVE_OPENMP
  }
#endif

//do multicoloring if in FORWARD direction
   if (opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {

	for(int i=0; i<mat->context->ncolors; ++i) {
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

//this is necessary since #pragma omp for doesn't understand !=
#define FORWARD_SHIFT_LOOP(start,end)                                  \
  for (ghost_lidx row=start; row<end; ++row){                          \
	 double inv_rownorm;					       \
         double rownorm = sigma_r*sigma_r + sigma_i*sigma_i;           \
         double scal_r = sigma_r*x_r[row] - sigma_i*x_i[row];          \
	 double scal_i = sigma_r*x_i[row] + sigma_i*x_r[row];          \
	 ghost_lidx  idx = mat->chunkStart[row];                   \
                                                                       \
        if(bval != NULL)                                               \
          scal_r  += bval[row];                                        \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            \
                 scal_r -= (double)mval[idx] * x_r[mat->col[idx]]; \
		 scal_i -= (double)mval[idx] * x_i[mat->col[idx]]; \
                 if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                                \
           	       rownorm += mval[idx]*mval[idx];                 \
                 idx += 1;                                             \
        }                                                              \
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){    				       \
	 inv_rownorm = 1.0/rownorm;                                    \
         scal_r *= inv_rownorm;                                        \
	 scal_i *= inv_rownorm;                                        \
        }                                                              \
        scal_r *= omega;                                               \
	scal_i *= omega;					       \
	idx -= mat->rowLen[row];                                   \
                                                                       \
        x_r[row] = x_r[row] - scal_r*sigma_r + scal_i*sigma_i;         \
	x_i[row] = x_i[row] - scal_r*sigma_i - scal_i*sigma_r;         \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           \
		x_r[mat->col[idx]] = x_r[mat->col[idx]] + scal_r * (double)mval[idx];\
		x_i[mat->col[idx]] = x_i[mat->col[idx]] + scal_i * (double)mval[idx];\
                idx += 1;                                              \
         }                                                             \
      } 

#define BACKWARD_SHIFT_LOOP(start,end)                                 \
  for (ghost_lidx row=start; row>end; --row){                          \
	 double inv_rownorm;					       \
         double rownorm = sigma_r*sigma_r + sigma_i*sigma_i;           \
         double scal_r = sigma_r*x_r[row] - sigma_i*x_i[row];          \
	 double scal_i = sigma_r*x_i[row] + sigma_i*x_r[row];          \
	 ghost_lidx  idx = mat->chunkStart[row];                   \
                                                                       \
        if(bval != NULL)                                               \
          scal_r  += bval[row];                                        \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            \
                 scal_r -= (double)mval[idx] * x_r[mat->col[idx]]; \
		 scal_i -= (double)mval[idx] * x_i[mat->col[idx]]; \
                 if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                                \
           	       rownorm += mval[idx]*mval[idx];                 \
                 idx += 1;                                             \
        }                                                              \
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){    				       \
	 inv_rownorm = 1.0/rownorm;                                    \
         scal_r *= inv_rownorm;                                        \
	 scal_i *= inv_rownorm;                                        \
        }                                                              \
        scal_r *= omega;                                               \
	scal_i *= omega;					       \
	idx -= mat->rowLen[row];                                   \
                                                                       \
        x_r[row] = x_r[row] - scal_r*sigma_r + scal_i*sigma_i;         \
	x_i[row] = x_i[row] - scal_r*sigma_i - scal_i*sigma_r;         \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           \
		x_r[mat->col[idx]] = x_r[mat->col[idx]] + scal_r * (double)mval[idx];\
		x_i[mat->col[idx]] = x_i[mat->col[idx]] + scal_i * (double)mval[idx];\
                idx += 1;                                              \
         }                                                             \
      } 

#define SHIFT_LOOP(start,end,stride)                                   \
  for (ghost_lidx row=start; row!=end; row+=stride){                   \
	 double inv_rownorm;					       \
         double rownorm = sigma_r*sigma_r + sigma_i*sigma_i;           \
         double scal_r = sigma_r*x_r[row] - sigma_i*x_i[row];          \
	 double scal_i = sigma_r*x_i[row] + sigma_i*x_r[row];          \
	 ghost_lidx  idx = mat->chunkStart[row];                   \
                                                                       \
        if(bval != NULL)                                               \
          scal_r  += bval[row];                                        \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) {            \
                 scal_r -= (double)mval[idx] * x_r[mat->col[idx]]; \
		 scal_i -= (double)mval[idx] * x_i[mat->col[idx]]; \
                 if(opts.normalize==GHOST_KACZ_NORMALIZE_NO)                                \
           	       rownorm += mval[idx]*mval[idx];                 \
                 idx += 1;                                             \
        }                                                              \
        if(opts.normalize==GHOST_KACZ_NORMALIZE_NO){    				       \
	 inv_rownorm = 1.0/rownorm;                                    \
         scal_r *= inv_rownorm;                                        \
	 scal_i *= inv_rownorm;                                        \
        }                                                              \
        scal_r *= omega;                                               \
	scal_i *= omega;					       \
	idx -= mat->rowLen[row];                                   \
                                                                       \
        x_r[row] = x_r[row] - scal_r*sigma_r + scal_i*sigma_i;         \
	x_i[row] = x_i[row] - scal_r*sigma_i - scal_i*sigma_r;         \
 	_Pragma("simd vectorlength(4)")                                \
         for (ghost_lidx j=0; j<mat->rowLen[row]; j++) {           \
		x_r[mat->col[idx]] = x_r[mat->col[idx]] + scal_r * (double)mval[idx];\
		x_i[mat->col[idx]] = x_i[mat->col[idx]] + scal_i * (double)mval[idx];\
                idx += 1;                                              \
         }                                                             \
      }                                                                \



ghost_error ghost_kacz_shift_bmc(ghost_densemat *x_real, ghost_densemat *x_imag, ghost_sparsemat *mat, ghost_densemat *b, double sigma_r, double sigma_i, ghost_kacz_opts opts)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    //int flag_err = 0;
    //currently only implementation for SELL-1-1
    //const int CHUNKHEIGHT = 1;  
    // const int NVECS = 1;

   if (mat->context->nzones == 0 || mat->context->zone_ptr == NULL){
        ERROR_LOG("Splitting of matrix by Block Multicoloring  has not be done!");
    }
  
   if (NVECS > 1) {
        ERROR_LOG("Multi-vec not implemented!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
   
    double *bval = NULL;
  
   if(b!= NULL)
     bval = (double *)(b->val);

    double *x_r = (double *)(x_real->val);
    double *x_i = (double *)(x_imag->val);
    double *mval = (double *)mat->val;
    double omega = *(double *)opts.omega;
    ghost_lidx *zone_ptr = (ghost_lidx*) mat->context->zone_ptr;
    //ghost_lidx nzones    = mat->context->nzones;
    ghost_lidx *color_ptr= (ghost_lidx*) mat->context->color_ptr;
    //ghost_lidx ncolors   = mat->context->ncolors;
    ghost_lidx nthreads  = mat->context->kacz_setting.active_threads;

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
	for(int i=mat->context->ncolors; i>0; --i) {
            mc_start  = color_ptr[i]-1;
            mc_end    = color_ptr[i-1]-1;
            stride   = -1;

#ifdef GHOST_HAVE_OPENMP  
	#pragma omp parallel for 
#endif
            BACKWARD_SHIFT_LOOP(mc_start,mc_end)
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

 if(mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_one_sweep) { 
    for(ghost_lidx zone = 0; zone<4; ++zone) { 

            SHIFT_LOOP(start[zone],end[zone],stride)   
                   
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
 } else if (mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_two_sweep) {
//TODO remove barriers its for testing 
     SHIFT_LOOP(start[0],end[0],stride)
      #pragma omp barrier 
      
      if(opts.direction == GHOST_KACZ_DIRECTION_BACKWARD) {
      	if(tid%2 != 0) {
            SHIFT_LOOP(start[1],end[1],stride)
        } 
       #pragma omp barrier
       if(tid%2 == 0) {
            SHIFT_LOOP(start[1],end[1],stride)
        }
      } else {
      	SHIFT_LOOP(start[1],end[1],stride) 
      }
      #pragma omp barrier
     
      if(opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {
      	if(tid%2 == 0) {
            SHIFT_LOOP(start[2],end[2],stride)
        } 
       #pragma omp barrier 
       if(tid%2 != 0) {
            SHIFT_LOOP(start[2],end[2],stride)
        }
      } else {
      	SHIFT_LOOP(start[2],end[2],stride)
      }

     #pragma omp barrier 
     SHIFT_LOOP(start[3],end[3],stride)               
  }      
#ifdef GHOST_HAVE_OPENMP
  }
#endif

//do multicoloring if in FORWARD direction
   if (opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {

	for(int i=0; i<mat->context->ncolors; ++i) {
           //for time being single thread
            mc_start  = color_ptr[i];
            mc_end    = color_ptr[i+1];
            stride    = 1;

#ifdef GHOST_HAVE_OPENMP
	#pragma omp parallel for
#endif
            FORWARD_SHIFT_LOOP(mc_start,mc_end)
	}
    }

    free(flag);	    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}


