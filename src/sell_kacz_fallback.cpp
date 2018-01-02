#include "ghost/sell_kacz_fallback.h"
#include "ghost/cpp11_fixes.h"
#include <complex>

#define LOOP_IN_CHUNK(row) \
VT *scal; \
VT *rownorm; \
scal = (VT*) malloc(sizeof(VT)*NBLOCKS*NSHIFTS); \
rownorm = (VT*) malloc(sizeof(VT)*NSHIFTS); \
ghost_lidx reset_idx = mat->chunkStart[chunk] + rowinchunk; \
idx = reset_idx; \
if(bval != NULL) { \
    for(int shift=0; shift<NSHIFTS; ++shift) { \
       for(int block=0; block<NBLOCKS; ++block) { \
            scal[shift*NBLOCKS+block] = bval[NBLOCKS*row+block]; \
        } \
    } \
} \
else { \
     for(int shift=0; shift<NSHIFTS; ++shift) { \
       for(int block=0; block<NBLOCKS; ++block) { \
            scal[shift*NBLOCKS+block] = 0; \
        } \
    } \
} \
for(int shift=0; shift<NSHIFTS; ++shift) {\
    rownorm[shift] = 0;\
}\
ghost_lidx diag_idx = 0; \
if(mat->context->row_map->loc_perm && mat->context->col_map->loc_perm != mat->context->row_map->loc_perm) {\
    diag_idx = mat->context->col_map->loc_perm[mat->context->row_map->loc_perm_inv[row]]; /*Assumes diag entry if absent is not the border element*/\
} \
else { \
    diag_idx = row; \
} \
VT mval_diag = 0; \
ghost_lidx j; \
for (j=0; j<mat->rowLen[row] && mat->col[idx]!=diag_idx; ++j) { \
    VT mval_idx = mval[idx]; \
    ghost_lidx col_idx = mat->col[idx]; \
    for(int shift=0; shift<NSHIFTS; ++shift) { \
        rownorm[shift] += ghost::norm(mval_idx); \
        for(int block=0; block<NBLOCKS; ++block) { \
            scal[shift*NBLOCKS+block] -= mval_idx * xval[col_idx*NBLOCKS*NSHIFTS+shift*NBLOCKS+block]; \
        } \
    } \
    idx+= CHUNKHEIGHT; \
} \
if(j!=mat->rowLen[row]) { \
    mval_diag = mval[idx];\
    idx += CHUNKHEIGHT;\
}\
for(int shift=0; shift<NSHIFTS; ++shift) { \
    mval_diag = mval_diag - sigma[shift]; \
    rownorm[shift] += ghost::norm(mval_diag); \
    for(int block=0; block<NBLOCKS; ++block) { \
        scal[shift*NBLOCKS+block] -= mval_diag * xval[diag_idx*NBLOCKS*NSHIFTS+shift*NBLOCKS+block]; \
    } \
} \
\
for (ghost_lidx k=j+1; k<mat->rowLen[row]; ++k) { \
    VT mval_idx = mval[idx]; \
    ghost_lidx col_idx = mat->col[idx]; \
    for(int shift=0; shift<NSHIFTS; ++shift) { \
        rownorm[shift] += ghost::norm(mval_idx); \
        for(int block=0; block<NBLOCKS; ++block) { \
            scal[shift*NBLOCKS+block] -= mval_idx * xval[col_idx*NBLOCKS*NSHIFTS+shift*NBLOCKS+block]; \
        } \
    } \
    idx+= CHUNKHEIGHT; \
} \
\
for(int shift=0; shift<NSHIFTS; ++shift) { \
    for(int block=0; block<NBLOCKS; ++block) { \
        scal[shift*NBLOCKS+block] /= (VT)rownorm[shift]; \
        scal[shift*NBLOCKS+block] *= omega; \
    } \
} \
\
idx = reset_idx; /*-= CHUNKHEIGHT*mat->rowLen[row];*/ \
\
for ( j=0; j<mat->rowLen[row]; j++) { \
    VT mval_idx = mval[idx]; \
    ghost_lidx col_idx = mat->col[idx]; \
    for(int shift=0; shift<NSHIFTS; ++shift) { \
        for(int block=0; block<NBLOCKS; ++block) { \
            xval[col_idx*NBLOCKS*NSHIFTS + shift*NBLOCKS + block] += scal[shift*NBLOCKS+block]*ghost::conj_or_nop(mval_idx);\
        } \
    } \
    idx += CHUNKHEIGHT; \
} \
if(opts.shift) { \
    for(int shift=0; shift<NSHIFTS; ++shift) { \
        for(int block=0; block<NBLOCKS; ++block) { \
            xval[diag_idx*NBLOCKS*NSHIFTS + shift*NBLOCKS + block] += scal[shift*NBLOCKS+block] * ghost::conj_or_nop(-sigma[shift]); \
        } \
    } \
} \
\



#define FORWARD_LOOP(start,end,MT,VT) \
{\
    start_rem = start%CHUNKHEIGHT; \
    start_chunk = start/CHUNKHEIGHT; \
    end_chunk = end/CHUNKHEIGHT; \
    end_rem = end%CHUNKHEIGHT; \
    chunk = 0; \
    rowinchunk = 0; \
    idx=0, row=0; \
    for(rowinchunk=start_rem; rowinchunk<MIN(CHUNKHEIGHT,(end_chunk-start_chunk)*CHUNKHEIGHT+end_rem); ++rowinchunk) { \
        row = rowinchunk + (start_chunk)*CHUNKHEIGHT; \
        chunk = start_chunk;\
        LOOP_IN_CHUNK(row); \
    } \
    _Pragma("omp parallel for private(chunk, rowinchunk, idx, row)") \
    for (chunk=start_chunk+1; chunk<end_chunk; ++chunk){ \
        for(rowinchunk=0; rowinchunk<CHUNKHEIGHT; ++rowinchunk) { \
            row = rowinchunk + chunk*CHUNKHEIGHT; \
            LOOP_IN_CHUNK(row); \
        } \
    } \
    if(start_chunk<end_chunk) { \
        for(rowinchunk=0; rowinchunk<end_rem; ++rowinchunk) { \
            row = rowinchunk + (end_chunk)*CHUNKHEIGHT; \
            chunk = end_chunk;\
            LOOP_IN_CHUNK(row); \
        } \
    }\
}\

#define BACKWARD_LOOP(start,end,MT,VT) \
{\
    start_rem = start%CHUNKHEIGHT; \
    start_chunk = start/CHUNKHEIGHT; \
    end_chunk = end/CHUNKHEIGHT; \
    end_rem = end%CHUNKHEIGHT; \
    chunk = 0; \
    rowinchunk = 0; \
    idx=0, row=0; \
    for(rowinchunk=start_rem; rowinchunk>=MAX(0,(end_chunk-start_chunk)*CHUNKHEIGHT+end_rem+1); --rowinchunk) { \
        row = rowinchunk + (start_chunk)*CHUNKHEIGHT; \
        chunk = start_chunk;\
        LOOP_IN_CHUNK(row) \
    } \
    _Pragma("omp parallel for private(chunk, rowinchunk, idx, row)") \
    for (chunk=start_chunk-1; chunk>end_chunk; --chunk){ \
        for(rowinchunk=CHUNKHEIGHT-1; rowinchunk>=0; --rowinchunk) { \
            row = rowinchunk + chunk*CHUNKHEIGHT; \
            LOOP_IN_CHUNK(row) \
        } \
    } \
    if(start_chunk>end_chunk) { \
        for(rowinchunk=CHUNKHEIGHT-1; rowinchunk>end_rem; --rowinchunk) { \
            row = rowinchunk + (end_chunk)*CHUNKHEIGHT; \
            chunk = end_chunk; \
            LOOP_IN_CHUNK(row) \
        } \
    } \
}\

#define LOCK_NEIGHBOUR(tid) \
if(tid == 0) \
    flag[0] = zone+1; \
if(tid == nthreads-1) \
    flag[nthreads+1] = zone+1; \
    flag[tid+1] = zone+1; \
    _Pragma("omp flush") \
    \
    if(opts.direction == GHOST_KACZ_DIRECTION_FORWARD) {\
        while(flag[tid+2]<zone+1){ \
            _Pragma("omp flush") \
        } \
    } else { \
        while(flag[tid]<zone+1 ){ \
            _Pragma("omp flush") \
        } \
    } 
        

 template <typename MT, typename VT>
 static ghost_error ghost_kacz_fallback_tmpl(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
 {
     GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
     
     ghost_lidx start_rem, start_chunk, end_chunk, end_rem; 
     ghost_lidx chunk = 0; 
     ghost_lidx rowinchunk = 0; 
     ghost_lidx idx=0, row=0; 
     
     if (mat->context->nzones == 0 || mat->context->zone_ptr == NULL){
         GHOST_ERROR_LOG("Splitting of matrix by Block Multicoloring has not be done!");
     }
     
     VT* sigma = NULL;
     VT zero = 0;
     int NBLOCKS, NSHIFTS;
     if(opts.shift) 
     {
         if(opts.num_shifts == 0) 
         { 
             NSHIFTS=1;
             sigma = &zero; 
         } else {
             NSHIFTS=opts.num_shifts;
             sigma = (VT*)opts.shift;
         }
     } else {
         NSHIFTS =1;
         sigma = &zero; 
     }
     
     NBLOCKS = x->traits.ncols/NSHIFTS;
     
     
     int CHUNKHEIGHT = mat->traits.C;
     
     MT *bval = NULL; 
     
     if(b!= NULL)
         bval = (MT *)(b->val);
     
     VT *xval = (VT *)(x->val);
     MT *mval = (MT *)mat->val;
     MT omega = *(MT *)opts.omega;
     ghost_lidx *zone_ptr = (ghost_lidx*) mat->context->zone_ptr;
     ghost_lidx *color_ptr= (ghost_lidx*) mat->context->color_ptr;
     ghost_lidx nthreads = mat->context->kacz_setting.active_threads;
     //int prev_nthreads;
     int prev_omp_nested;
     
     #ifdef GHOST_HAVE_OPENMP
     // disables dynamic thread adjustments 
     ghost_omp_set_dynamic(0);
     ghost_omp_nthread_set(nthreads);
     prev_omp_nested = ghost_omp_get_nested();
     ghost_omp_set_nested(0); 
     //printf("Setting number of threads to %d for KACZ sweep\n",nthreads);
     
     #endif
     ghost_lidx *flag;
     flag = (ghost_lidx*) malloc((nthreads+2)*sizeof(ghost_lidx));
     
     for (int i=0; i<nthreads+2; i++) {
         flag[i] = 0;
     } //always execute first and last blocks
     flag[0] = 1; 
     flag[nthreads+1] = 1;
     
     int mc_start, mc_end;
     
     //    ghost_lidx stride = 0;
     if (opts.direction == GHOST_KACZ_DIRECTION_BACKWARD) {
         #ifdef GHOST_HAVE_COLPACK 
         for(int i=mat->context->ncolors; i>0; --i) {
             mc_start  = color_ptr[i]-1;
             mc_end    = color_ptr[i-1]-1;
             BACKWARD_LOOP(mc_start,mc_end,MT,MT)
         }
         #else
         ghost_omp_set_dynamic(0);
         ghost_omp_nthread_set(1);
         
         for(int i=mat->context->ncolors; i>0; --i) {
             mc_start  = color_ptr[i]-1;
             mc_end    = color_ptr[i-1]-1;
             BACKWARD_LOOP(mc_start,mc_end,MT,MT)
         }
         ghost_omp_nthread_set(nthreads);
         
         #endif
         
         #ifdef GHOST_HAVE_OPENMP  
         #pragma omp parallel private (start_rem, start_chunk, end_chunk, end_rem, chunk, rowinchunk, idx, row)
         {
             #endif 
             ghost_lidx tid = ghost_omp_threadnum();
             ghost_lidx start[4];
             ghost_lidx end[4];
             
             start[0]  = zone_ptr[4*tid+4]-1; 
             end[0]    = zone_ptr[4*tid+3]-1;   	
             start[1]  = zone_ptr[4*tid+3]-1;
             end[1]    = zone_ptr[4*tid+2]-1;
             start[2]  = zone_ptr[4*tid+2]-1;
             end[2]    = zone_ptr[4*tid+1]-1;
             start[3]  = zone_ptr[4*tid+1]-1;
             end[3]    = zone_ptr[4*tid]  -1;
             if(mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_one_sweep) { 
                 for(ghost_lidx zone = 0; zone<4; ++zone) {
                    BACKWARD_LOOP(start[zone],end[zone],MT,MT)   
                    #pragma omp barrier                           
                 }
             } else if (mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_two_sweep) {
                 BACKWARD_LOOP(start[0],end[0],MT,MT)
                 #pragma omp barrier 
                 if(tid%2 != 0) {
                     BACKWARD_LOOP(start[1],end[1],MT,MT)
                 }	 
                 #pragma omp barrier
                 if(tid%2 == 0) {
                     BACKWARD_LOOP(start[1],end[1],MT,MT)
                 }
                 #pragma omp barrier
                 BACKWARD_LOOP(start[2],end[2],MT,MT)
                 #pragma omp barrier 
                 BACKWARD_LOOP(start[3],end[3],MT,MT)               
             }      
             
             #ifdef GHOST_HAVE_OPENMP
         }
         #endif
         
     } else {
         
         #ifdef GHOST_HAVE_OPENMP  
         #pragma omp parallel private(start_rem, start_chunk, end_chunk, end_rem, chunk, rowinchunk, idx, row)
         {
             #endif 
             ghost_lidx tid = ghost_omp_threadnum();
             ghost_lidx start[4];
             ghost_lidx end[4];
             start[0]  = zone_ptr[4*tid];
             end[0]    = zone_ptr[4*tid+1];
             start[1]  = zone_ptr[4*tid+1];
             end[1]    = zone_ptr[4*tid+2];
             start[2]  = zone_ptr[4*tid+2];
             end[2]    = zone_ptr[4*tid+3];
             start[3]  = zone_ptr[4*tid+3];
             end[3]    = zone_ptr[4*tid+4];
             
             if(mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_one_sweep) { 
                 for(ghost_lidx zone = 0; zone<4; ++zone) { 
                     FORWARD_LOOP(start[zone],end[zone],MT,MT)        
                     #pragma omp barrier                           
                 }
             } else if (mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_two_sweep) {
                 FORWARD_LOOP(start[0],end[0],MT,MT)
                 #pragma omp barrier 
                 FORWARD_LOOP(start[1],end[1],MT,MT)
                 #pragma omp barrier
                 if(tid%2 == 0) {
                     FORWARD_LOOP(start[2],end[2],MT,MT)
                 }	 
                 #pragma omp barrier
                 if(tid%2 != 0) {
                     FORWARD_LOOP(start[2],end[2],MT,MT)
                 }
                 #pragma omp barrier 
                 FORWARD_LOOP(start[3],end[3],MT,MT)               
             }      
             
             #ifdef GHOST_HAVE_OPENMP
         }
         #endif
         
         #ifdef GHOST_HAVE_COLPACK 
         for(int i=0; i<mat->context->ncolors; ++i) { 
             mc_start  = color_ptr[i];
             mc_end    = color_ptr[i+1];
             FORWARD_LOOP(mc_start,mc_end,MT,MT)
         }
         #else
         ghost_omp_set_dynamic(0);
         ghost_omp_nthread_set(1);
         
         for(int i=0; i<mat->context->ncolors; ++i) { 
             mc_start  = color_ptr[i];
             mc_end    = color_ptr[i+1];
             FORWARD_LOOP(mc_start,mc_end,MT,MT)
         }
         
         ghost_omp_nthread_set(nthreads);
         
         #endif
         
     }
     
     free(flag);	    
     
     //reset values TODO reset n_threads
     #ifdef GHOST_HAVE_OPENMP
     ghost_omp_set_nested(prev_omp_nested);    
     #endif
     
     GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
     return GHOST_SUCCESS;
 }
 
 ghost_error ghost_kacz_fallback(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
 {
     ghost_error ret = GHOST_SUCCESS;
     
     //SELECT_TMPL_1DATATYPE(mat->traits.datatype,std::complex,ret,ghost_kacz_BMC_u_plain_rm_CHUNKHEIGHT_NVECS_tmpl,x,mat,b,opts);
     SELECT_TMPL_2DATATYPES_base_derived(mat->traits.datatype,x->traits.datatype,std::complex,ret,ghost_kacz_fallback_tmpl,x,mat,b,opts);
     return ret;
 }
 
 
