#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "spmvm.h"
#include "matricks.h"

void hybrid_kernel_0   (ghost_vec_t*, ghost_setup_t*, ghost_vec_t*, int);
void kern_glob_CRS_0(ghost_vec_t* res, CR_TYPE* cr, ghost_vec_t* invec, int spmvmOptions);
void kern_glob_CRS_CD_0(ghost_vec_t* res, CR_TYPE* cr, ghost_vec_t* invec, int spmvmOptions);

#ifdef MIC
void mic_kernel_0      (ghost_vec_t*, BJDS_TYPE*, ghost_vec_t*, int);
void mic_kernel_0_unr      (ghost_vec_t*, BJDS_TYPE*, ghost_vec_t*, int);
void mic_kernel_0_intr      (ghost_vec_t*, BJDS_TYPE*, ghost_vec_t*, int);
void mic_kernel_0_intr_16(ghost_vec_t*, BJDS_TYPE*, ghost_vec_t*, int);
void mic_kernel_0_intr_16_rem(ghost_vec_t* res, BJDS_TYPE* bjds, ghost_vec_t* invec, int spmvmOptions);
void mic_kernel_0_intr_overlap      (ghost_vec_t*, BJDS_TYPE*, ghost_vec_t*, int);
#endif

#ifdef AVX
void avx_kernel_0_intr(ghost_vec_t* res, BJDS_TYPE* mv, ghost_vec_t* invec, int spmvmOptions);
void avx_kernel_0_intr_rem(ghost_vec_t* res, BJDS_TYPE* mv, ghost_vec_t* invec, int spmvmOptions);
void avx_kernel_0_intr_rem_if(ghost_vec_t* res, BJDS_TYPE* bjds, ghost_vec_t* invec, int spmvmOptions);
#endif

#ifdef SSE
void sse_kernel_0_intr(ghost_vec_t* res, BJDS_TYPE* bjds, ghost_vec_t* invec, int spmvmOptions);
void sse_kernel_0_intr_rem(ghost_vec_t* res, BJDS_TYPE* bjds, ghost_vec_t* invec, int spmvmOptions);
#endif

#ifdef MPI
void hybrid_kernel_I   (ghost_vec_t*, ghost_setup_t*, ghost_vec_t*, int);
void hybrid_kernel_II  (ghost_vec_t*, ghost_setup_t*, ghost_vec_t*, int);
void hybrid_kernel_III (ghost_vec_t*, ghost_setup_t*, ghost_vec_t*, int);
#endif


#endif
