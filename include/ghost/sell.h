
#ifndef __GHOST_SPMFORMAT_SELL__
#define __GHOST_SPMFORMAT_SELL__

#include "config.h"
#include "types.h"

#define SELL_AUX_S 0
#define SELL_AUX_C 1
#define SELL_AUX_T 2

typedef struct 
{
#ifdef GHOST_HAVE_OPENCL
    cl_mem val;
    cl_mem col;
    cl_mem rowLen;
    cl_mem chunkStart;
    cl_mem chunkLen;
    ghost_cl_midx_t nrows;
    ghost_cl_midx_t nrowsPadded;
#else
    void *empty;
#endif
} 
CL_SELL_TYPE;

typedef struct 
{
#ifdef GHOST_HAVE_CUDA
    char * val;
    ghost_midx_t * col;
    ghost_midx_t * rowLen;
    ghost_midx_t * rowLenPadded;
    ghost_mnnz_t * chunkStart;
    ghost_midx_t * chunkLen;
    ghost_midx_t nrows;
    ghost_midx_t nrowsPadded;
    int T; // number of threads per row (if applicable)
    ghost_midx_t chunkHeight;
#else
    void *empty;
#endif
} 
CU_SELL_TYPE;

typedef struct 
{
    char *val;
    ghost_midx_t *col;
    ghost_mnnz_t *chunkStart;
    ghost_midx_t nrows;
    ghost_midx_t ncols;
    ghost_midx_t nrowsPadded;
    ghost_mnnz_t nnz;
    ghost_mnnz_t nEnts;
    double beta; // chunk occupancy
    double variance; // row length variance
    double deviation; // row lenght standard deviation
    double cv; // row lenght coefficient of variation
    int T; // number of threads per row (if applicable)
    ghost_midx_t *chunkMin; // for version with remainder loop
    ghost_midx_t *chunkLen; // for version with remainder loop
    ghost_midx_t *chunkLenPadded; // for version with remainder loop
    ghost_midx_t *rowLen;   // for version with remainder loop
    ghost_midx_t *rowLenPadded; // for SELL-T 
    ghost_midx_t maxRowLen;
    ghost_midx_t nMaxRows;
    ghost_midx_t chunkHeight;
    ghost_midx_t scope;
    
    CL_SELL_TYPE *clmat;
    CU_SELL_TYPE *cumat;
} 
SELL_TYPE;

typedef struct 
{
    ghost_midx_t row, nEntsInRow;
} 
ghost_sorting_t;
#define SELL(mat) ((SELL_TYPE *)(mat->data))

#define SELL_CUDA_THREADSPERBLOCK 256

enum GHOST_SELL_C {GHOST_SELL_C_1 = 1, GHOST_SELL_C_2 = 2, GHOST_SELL_C_4 = 4, GHOST_SELL_C_256 = 256};
enum GHOST_SELL_T {GHOST_SELL_T_1 = 1, GHOST_SELL_T_2 = 2, GHOST_SELL_T_4 = 4, GHOST_SELL_T_256 = 256};

ghost_mat_t * ghost_SELL_init(ghost_context_t *, ghost_mtraits_t *);
#ifdef __cplusplus
template<typename, typename, int> void SELL_kernel_plain_tmpl(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
template<typename, typename> void SELL_kernel_plain_ELLPACK_tmpl(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int);
template <typename> void SELL_fromCRS(ghost_mat_t *, void *);
extern "C" {
#endif
void dd_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ds_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dc_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dz_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sd_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ss_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sc_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sz_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cd_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cs_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cc_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cz_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zd_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zs_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zc_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zz_SELL_kernel_plain(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);

void dd_SELL_kernel_SSE(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dd_SELL_kernel_AVX(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dd_SELL_kernel_AVX_32(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dd_SELL_kernel_MIC_16(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dd_SELL_kernel_MIC_32(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
#ifdef GHOST_HAVE_CUDA
void dd_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ds_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dc_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void dz_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sd_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void ss_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sc_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void sz_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cd_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cs_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cc_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void cz_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zd_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zs_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zc_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
void zz_SELL_kernel_CU(ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options);
#endif
void d_SELL_fromCRS(ghost_mat_t *mat, void *crs);
void s_SELL_fromCRS(ghost_mat_t *mat, void *crs);
void c_SELL_fromCRS(ghost_mat_t *mat, void *crs);
void z_SELL_fromCRS(ghost_mat_t *mat, void *crs);

const char * d_SELL_stringify(ghost_mat_t *mat, int dense);
const char * s_SELL_stringify(ghost_mat_t *mat, int dense);
const char * c_SELL_stringify(ghost_mat_t *mat, int dense);
const char * z_SELL_stringify(ghost_mat_t *mat, int dense);
#ifdef __cplusplus
}
#endif

#endif
