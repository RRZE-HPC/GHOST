#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/omp.h"
#include "ghost/machine.h"
#include "ghost/math.h"
#include "ghost/sparsemat.h"
#include "ghost/densemat.h"
#include "ghost/locality.h"
//#include "ghost/sell_spmtv_bmc_rm_gen.h"
#include "ghost/sell_spmtv_bmc_rm.h"
#include <complex>
#include <complex.h>

//#GHOST_SUBST NVECS ${BLOCKDIM1}
//#GHOST_SUBST CHUNKHEIGHT ${CHUNKHEIGHT}
#define CHUNKHEIGHT 1
#define NVECS 1

#if (NVECS==1 && CHUNKHEIGHT==1)
//this is necessary since #pragma omp for doesn't understand !=
#define LOOP(start,end,MT,VT) \
    _Pragma("omp parallel for") \
for (ghost_lidx row=start; row<end; ++row){ \
    VT x_row = xval[row]; \
    ghost_lidx idx = mat->chunkStart[row]; \
    _Pragma("simd vectorlength(4)") \
    for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
        bval[mat->col[idx+j]] = bval[mat->col[idx+j]] + (MT)mval[idx+j] * x_row;\
    } \
} \

#elif CHUNKHEIGHT == 1

#define LOOP(start,end,MT,VT) \
    _Pragma("omp parallel for") \
for (ghost_lidx row=start; row<end; ++row){ \
    ghost_lidx idx = mat->chunkStart[row]; \
    for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
        MT mval_idx = (MT)mval[idx]; \
        ghost_lidx col_idx = mat->col[idx]; \
        _Pragma("simd") \
        for(int block=0; block<NVECS; ++block) { \
            VT x_row = xval[row]; \
            bval[NVECS*col_idx+block] = bval[NVECS*col_idx+block] + mval_idx * xval[NVECS*row + block];\
        } \
        idx += 1; \
    } \
} \

#else 

#define LOOP(start,end,MT,VT) \
    start_rem = start%CHUNKHEIGHT; \
start_chunk = start/CHUNKHEIGHT; \
end_chunk = end/CHUNKHEIGHT; \
end_rem = end%CHUNKHEIGHT; \
chunk = 0; \
rowinchunk = 0; \
idx=0, row=0; \
for(rowinchunk=start_rem; rowinchunk<MIN(CHUNKHEIGHT,(end_chunk-start_chunk)*CHUNKHEIGHT+end_rem); ++rowinchunk) { \
    MT rownorm = 0.; \
    MT scal[NVECS] = {0}; \
    idx = mat->chunkStart[start_chunk] + rowinchunk; \
    row = rowinchunk + (start_chunk)*CHUNKHEIGHT; \
    if(bval != NULL) { \
        for(int block=0; block<NVECS; ++block) { \
            scal[block] = -bval[NVECS*row+block]; \
        } \
    } \
    \
    for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) { \
        MT mval_idx = mval[idx]; \
        for(int block=0; block<NVECS; ++block) { \
            scal[block] += mval_idx * xval[NVECS*mat->col[idx]+block]; \
        } \
        rownorm += std::norm(mval_idx); \
        idx+=CHUNKHEIGHT; \
    } \
    for(int block=0; block<NVECS; ++block) { \
        scal[block] /= (MT)rownorm; \
        scal[block] *= omega; \
    } \
    \
    idx -= CHUNKHEIGHT*mat->rowLen[row]; \
    \
    _Pragma("simd vectorlength(4)") \
    for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
        for(int block=0; block<NVECS; ++block) { \
            xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * std::conj(mval[idx]);\
        } \
        idx += CHUNKHEIGHT; \
    } \
} \
_Pragma("omp parallel for private(chunk, rowinchunk, idx, row)") \
for (chunk=start_chunk+1; chunk<end_chunk; ++chunk){ \
    for(rowinchunk=0; rowinchunk<CHUNKHEIGHT; ++rowinchunk) { \
        MT rownorm = 0.; \
        MT scal[NVECS] = {0}; \
        idx = mat->chunkStart[chunk] + rowinchunk; \
        row = rowinchunk + chunk*CHUNKHEIGHT; \
        if(bval != NULL) { \
            for(int block=0; block<NVECS; ++block) { \
                scal[block] = -bval[NVECS*row+block]; \
            } \
        } \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) { \
            MT mval_idx = (MT)mval[idx]; \
            for(int block=0; block<NVECS; ++block) { \
                scal[block] += (MT)mval_idx * xval[NVECS*mat->col[idx]+block];\
            } \
            rownorm += std::norm(mval_idx); \
            idx+=CHUNKHEIGHT; \
        } \
        for(int block=0; block<NVECS; ++block){ \
            scal[block] /= (MT)rownorm; \
            scal[block] *= omega; \
        } \
        idx -= CHUNKHEIGHT*mat->rowLen[row]; \
        \
        _Pragma("simd vectorlength(4)") \
        for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
            for(int block=0; block<NVECS; ++block) { \
                xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * std::conj(mval[idx]);\
            } \
            idx += CHUNKHEIGHT; \
        } \
    } \
} \
if(start_chunk<end_chunk) { \
    for(rowinchunk=0; rowinchunk<end_rem; ++rowinchunk) { \
        MT rownorm = 0.; \
        MT scal[NVECS] = {0}; \
        idx = mat->chunkStart[end_chunk] + rowinchunk; \
        row = rowinchunk + (end_chunk)*CHUNKHEIGHT; \
        if(bval != NULL) { \
            for(int block=0; block<NVECS; ++block) { \
                scal[block] = -bval[NVECS*row+block]; \
            } \
        } \
        for (ghost_lidx j=0; j<mat->rowLen[row]; ++j) { \
            MT mval_idx = (MT)mval[idx]; \
            for(int block=0; block<NVECS; ++block) { \
                scal[block] += (MT)mval_idx * xval[NVECS*mat->col[idx]+block];\
            } \
            rownorm += std::norm(mval_idx); \
            idx+=CHUNKHEIGHT; \
        } \
        for(int block=0; block<NVECS; ++block){ \
            scal[block] /= (MT)rownorm; \
            scal[block] *= omega; \
        } \
        idx -= CHUNKHEIGHT*mat->rowLen[row]; \
        \
        _Pragma("simd vectorlength(4)") \
        for (ghost_lidx j=0; j<mat->rowLen[row]; j++) { \
            for(int block=0; block<NVECS; ++block) { \
                xval[NVECS*mat->col[idx]+block] = xval[NVECS*mat->col[idx]+block] - scal[block] * std::conj(mval[idx]);\
            } \
            idx += CHUNKHEIGHT; \
        } \
    } \
}
#endif


//static ghost_error ghost_spmtv_BMC_u_plain_rm_CHUNKHEIGHT_NVECS_tmpl(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)

ghost_error ghost_spmtv_BMC(ghost_densemat *b, ghost_sparsemat *mat, ghost_densemat *x)
{

    typedef double MT;
    typedef double VT;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    //int flag_err = 0;
    //currently only implementation for SELL-1-1
    //const int CHUNKHEIGHT = 1; 
    //const int NVECS = 1;

#if CHUNKHEIGHT>1
    ghost_lidx start_rem, start_chunk, end_chunk, end_rem; 
    ghost_lidx chunk = 0; 
    ghost_lidx rowinchunk = 0; 
    ghost_lidx idx=0, row=0; 
#endif

    bool flag_BMC = true;
    bool flag_MC = true;

    if (mat->context->nzones == 0 || mat->context->zone_ptr == NULL){
        flag_BMC = false;
    }

    if (mat->context->ncolors == 0 || mat->context->color_ptr == NULL){
        flag_MC = false;
    }

    if( (flag_MC == false) && (flag_BMC==false) ){
        ERROR_LOG("Splitting of matrix by (Block) Multicoloring has not be done!");
    }

    MT *bval = (MT *)(b->val);       
    MT *xval = (MT *)(x->val);
    MT *mval = (MT *)mat->val;
    ghost_lidx *zone_ptr = (ghost_lidx*) mat->context->zone_ptr;
    ghost_lidx *color_ptr= (ghost_lidx*) mat->context->color_ptr;
    ghost_lidx nthreads;

    int prev_nthreads;
    int prev_omp_nested;

#ifdef GHOST_HAVE_OPENMP
    // disables dynamic thread adjustments 
    ghost_omp_set_dynamic(0);
#pragma omp parallel shared(prev_nthreads)
    {
#pragma omp single
        {
            prev_nthreads = ghost_omp_nthread();
        }
    }

    if(flag_BMC) {
        nthreads = mat->context->kacz_setting.active_threads;
    }
    else{
        nthreads = prev_nthreads;
    }

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

    if(flag_BMC)
    {
#ifdef GHOST_HAVE_OPENMP 
#if CHUNKHEIGHT > 1
#pragma omp parallel private(start_rem, start_chunk, end_chunk, end_rem, chunk, rowinchunk, idx, row)
#else
#pragma omp parallel
#endif
        {
#endif 
            ghost_lidx tid = ghost_omp_threadnum();
            ghost_lidx start[4];
            ghost_lidx end[4];
            start[0] = zone_ptr[4*tid];
            end[0] = zone_ptr[4*tid+1];
            start[1] = zone_ptr[4*tid+1];
            end[1] = zone_ptr[4*tid+2];
            start[2] = zone_ptr[4*tid+2];
            end[2] = zone_ptr[4*tid+3];
            start[3] = zone_ptr[4*tid+3];
            end[3] = zone_ptr[4*tid+4];

            if(mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_one_sweep) { 
                for(ghost_lidx zone = 0; zone<4; ++zone) {
                    LOOP(start[zone],end[zone],MT,VT) 
#pragma omp barrier 
                }
            } else if (mat->context->kacz_setting.kacz_method == GHOST_KACZ_METHOD_BMC_two_sweep) {
                LOOP(start[0],end[0],MT,VT)
#pragma omp barrier 
                    LOOP(start[1],end[1],MT,VT)
#pragma omp barrier
                    if(tid%2 == 0) {
                        LOOP(start[2],end[2],MT,VT)
                    } 
#pragma omp barrier
                if(tid%2 != 0) {
                    LOOP(start[2],end[2],MT,VT)
                }
#pragma omp barrier 
                LOOP(start[3],end[3],MT,VT) 
            } 

#ifdef GHOST_HAVE_OPENMP
        }
#endif
    }

#ifndef GHOST_HAVE_COLPACK 
    ghost_omp_set_dynamic(0);
    ghost_omp_nthread_set(1);
#endif
    for(int i=0; i<mat->context->ncolors; ++i) {
        mc_start = color_ptr[i];
        mc_end = color_ptr[i+1];

        LOOP(mc_start,mc_end,MT,VT)
    }

#ifndef GHOST_HAVE_COLPACK 
    ghost_omp_nthread_set(nthreads);
#endif        


    free(flag); 

#ifdef GHOST_HAVE_OPENMP
    ghost_omp_nthread_set(prev_nthreads);
    ghost_omp_set_nested(prev_omp_nested); 
#endif

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

/*ghost_error ghost_spmtv__BMC_u_plain_x_x_rm_CHUNKHEIGHT_NVECS(ghost_densemat *x, ghost_sparsemat *mat, ghost_densemat *b, ghost_kacz_opts opts)
  {
  ghost_error ret = GHOST_SUCCESS;

  SELECT_TMPL_1DATATYPE(mat->traits.datatype,std::complex,ret,ghost_spmtv_BMC_u_plain_rm_CHUNKHEIGHT_NVECS_tmpl,x,mat,b,opts);
// TODO mixed datatypes
// SELECT_TMPL_2DATATYPES(mat->traits.datatype,x->traits.datatype,ghost_complex,ret,ghost_kacz_BMC_u_plain_cm_CHUNKHEIGHT_NVECS_tmpl,x,mat,b,opts);    
return ret;    
}*/

