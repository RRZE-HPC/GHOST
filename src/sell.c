#include <ghost_sell.h>
#include <ghost_crs.h>
#include <ghost_util.h>
#include <ghost_mat.h>
#include <ghost_constants.h>

#include <libgen.h>
#include <string.h>
#include <stdlib.h>

#if GHOST_HAVE_OPENMP
#include <omp.h>
#endif

#ifdef GHOST_HAVE_CUDA
//#include "private/sell_cukernel.h"
#endif

#if defined(SSE) || defined(AVX) || defined(MIC)
#include <immintrin.h>
#endif
#if defined(VSX)
#include <altivec.h>
#endif

void (*SELL_kernels_SSE[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_SSE,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

void (*SELL_kernels_AVX[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_AVX,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

void (*SELL_kernels_AVX_32[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_AVX_32,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

void (*SELL_kernels_MIC_16[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_MIC_16,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

void (*SELL_kernels_MIC_32[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{NULL,NULL,NULL,NULL},
    {NULL,&dd_SELL_kernel_MIC_32,NULL,NULL},
    {NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL}};

void (*SELL_kernels_plain[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{&ss_SELL_kernel_plain,&sd_SELL_kernel_plain,&sc_SELL_kernel_plain,&sz_SELL_kernel_plain},
    {&ds_SELL_kernel_plain,&dd_SELL_kernel_plain,&dc_SELL_kernel_plain,&dz_SELL_kernel_plain},
    {&cs_SELL_kernel_plain,&cd_SELL_kernel_plain,&cc_SELL_kernel_plain,&cz_SELL_kernel_plain},
    {&zs_SELL_kernel_plain,&zd_SELL_kernel_plain,&zc_SELL_kernel_plain,&zz_SELL_kernel_plain}};

#ifdef GHOST_HAVE_CUDA
void (*SELL_kernels_CU[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{&ss_SELL_kernel_CU,&sd_SELL_kernel_CU,&sc_SELL_kernel_CU,&sz_SELL_kernel_CU},
    {&ds_SELL_kernel_CU,&dd_SELL_kernel_CU,&dc_SELL_kernel_CU,&dz_SELL_kernel_CU},
    {&cs_SELL_kernel_CU,&cd_SELL_kernel_CU,&cc_SELL_kernel_CU,&cz_SELL_kernel_CU},
    {&zs_SELL_kernel_CU,&zd_SELL_kernel_CU,&zc_SELL_kernel_CU,&zz_SELL_kernel_CU}};
#endif

void (*SELL_fromCRS_funcs[4]) (ghost_mat_t *, void *) = 
{&s_SELL_fromCRS, &d_SELL_fromCRS, &c_SELL_fromCRS, &z_SELL_fromCRS}; 

static ghost_mnnz_t SELL_nnz(ghost_mat_t *mat);
static ghost_midx_t SELL_nrows(ghost_mat_t *mat);
static ghost_midx_t SELL_ncols(ghost_mat_t *mat);
static void SELL_printInfo(ghost_mat_t *mat);
static char * SELL_formatName(ghost_mat_t *mat);
static ghost_midx_t SELL_rowLen (ghost_mat_t *mat, ghost_midx_t i);
static size_t SELL_byteSize (ghost_mat_t *mat);
static void SELL_fromCRS(ghost_mat_t *mat, void *crs);
static void SELL_upload(ghost_mat_t* mat); 
static void SELL_CUupload(ghost_mat_t *mat);
static void SELL_fromBin(ghost_mat_t *mat, char *);
static void SELL_free(ghost_mat_t *mat);
static void SELL_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#ifdef GHOST_HAVE_OPENCL
static void SELL_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef GHOST_HAVE_CUDA
static void SELL_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options);
#endif

ghost_mat_t * ghost_SELL_init(ghost_mtraits_t * traits)
{
    ghost_mat_t *mat = (ghost_mat_t *)ghost_malloc(sizeof(ghost_mat_t));
    mat->traits = traits;
    DEBUG_LOG(1,"Setting functions for SELL matrix");

    mat->CLupload = &SELL_upload;
    mat->CUupload = &SELL_CUupload;
    mat->fromFile = &SELL_fromBin;
    mat->printInfo = &SELL_printInfo;
    mat->formatName = &SELL_formatName;
    mat->rowLen     = &SELL_rowLen;
    mat->byteSize   = &SELL_byteSize;
    mat->kernel     = &SELL_kernel_plain;
    mat->fromCRS    = &SELL_fromCRS;
#ifdef VSX_INTR
    mat->kernel = &SELL_kernel_VSX;
#endif
#ifdef GHOST_HAVE_OPENCL
    if (!(traits->flags & GHOST_SPM_HOST))
        mat->kernel   = &SELL_kernel_CL;
#endif
#ifdef GHOST_HAVE_CUDA
    if (!(traits->flags & GHOST_SPM_HOST))
        mat->kernel   = &SELL_kernel_CU;
#endif
    mat->nnz      = &SELL_nnz;
    mat->nrows    = &SELL_nrows;
    mat->ncols    = &SELL_ncols;
    mat->destroy  = &SELL_free;

    mat->localPart = NULL;
    mat->remotePart = NULL;

    return mat;
}

static ghost_mnnz_t SELL_nnz(ghost_mat_t *mat)
{
    if (mat->data == NULL)
        return -1;
    return SELL(mat)->nnz;
}
static ghost_midx_t SELL_nrows(ghost_mat_t *mat)
{
    if (mat->data == NULL)
        return -1;
    return SELL(mat)->nrows;
}
static ghost_midx_t SELL_ncols(ghost_mat_t *mat)
{
    UNUSED(mat);
    return 0;
}

static void SELL_printInfo(ghost_mat_t *mat)
{
    ghost_printLine("Max row length (# rows)",NULL,"%d (%d)",SELL(mat)->maxRowLen,SELL(mat)->nMaxRows);
    ghost_printLine("Chunk height (C)",NULL,"%d",SELL(mat)->chunkHeight);
    ghost_printLine("Chunk occupancy (beta)",NULL,"%f",SELL(mat)->beta);
    ghost_printLine("Row length variance",NULL,"%f",SELL(mat)->variance);
    ghost_printLine("Row length standard deviation",NULL,"%f",SELL(mat)->deviation);
    ghost_printLine("Row length coefficient of variation",NULL,"%f",SELL(mat)->deviation*1./(ghost_getMatNnz(mat)*1.0/(double)ghost_getMatNrows(mat)));
    ghost_printLine("Threads per row (T)",NULL,"%d",SELL(mat)->T);
    if (mat->traits->flags & GHOST_SPM_SORTED) {
        ghost_printLine("Sorted",NULL,"yes");
        ghost_printLine("Scope (sigma)",NULL,"%u",*(unsigned int *)(mat->traits->aux));
        ghost_printLine("Permuted columns",NULL,"%s",mat->traits->flags&GHOST_SPM_PERMUTECOLIDX?"yes":"no");
    } else {
        ghost_printLine("Sorted",NULL,"no");
    }
}

static char * SELL_formatName(ghost_mat_t *mat)
{
    UNUSED(mat);
    return "SELL";
}

static ghost_midx_t SELL_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
    if (mat->traits->flags & GHOST_SPM_SORTED)
        i = mat->context->rowPerm[i];

    return SELL(mat)->rowLen[i];
}

/*static ghost_dt SELL_entry (ghost_mat_t *mat, ghost_midx_t i, ghost_midx_t j)
  {
  ghost_midx_t e;

  if (mat->traits->flags & GHOST_SPM_SORTED)
  i = mat->rowPerm[i];
  if (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX)
  j = mat->rowPerm[j];

  for (e=SELL(mat)->chunkStart[i/SELL_LEN]+i%SELL_LEN; 
  e<SELL(mat)->chunkStart[i/SELL_LEN+1]; 
  e+=SELL_LEN) {
  if (SELL(mat)->col[e] == j)
  return SELL(mat)->val[e];
  }
  return 0.;
  }*/

static size_t SELL_byteSize (ghost_mat_t *mat)
{
    return (size_t)((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_mnnz_t) + 
            SELL(mat)->nEnts*(sizeof(ghost_midx_t)+ghost_sizeofDataType(mat->traits->datatype)));
}

static void SELL_fromBin(ghost_mat_t *mat, char *matrixPath)
{
    DEBUG_LOG(1,"Creating SELL matrix from binary file");
    ghost_mtraits_t crsTraits = {.format = GHOST_SPM_FORMAT_CRS,.flags=GHOST_SPM_HOST,.datatype=mat->traits->datatype};
    ghost_mat_t *crsMat = ghost_createMatrix(mat->context,&crsTraits,1);
    crsMat->fromFile(crsMat,matrixPath);
    mat->name = basename(matrixPath);
    

#ifdef GHOST_HAVE_MPI

    DEBUG_LOG(1,"Converting local and remote part to the desired data format");    
    mat->localPart = ghost_createMatrix(mat->context,&mat->traits[0],1); // TODO trats[1]
    mat->localPart->symmetry = crsMat->symmetry;
    mat->localPart->fromCRS(mat->localPart,crsMat->localPart->data);

    mat->remotePart = ghost_createMatrix(mat->context,&mat->traits[0],1); // TODO traits[2]
    mat->remotePart->fromCRS(mat->remotePart,crsMat->remotePart->data);


#ifdef GHOST_HAVE_OPENCL
    if (!(mat->localPart->traits->flags & GHOST_SPM_HOST))
        mat->localPart->CLupload(mat->localPart);
    if (!(mat->remotePart->traits->flags & GHOST_SPM_HOST))
        mat->remotePart->CLupload(mat->remotePart);
#endif
#ifdef GHOST_HAVE_CUDA
    if (!(mat->localPart->traits->flags & GHOST_SPM_HOST))
        mat->localPart->CUupload(mat->localPart);
    if (!(mat->remotePart->traits->flags & GHOST_SPM_HOST))
        mat->remotePart->CUupload(mat->remotePart);
#endif
#endif

    mat->symmetry = crsMat->symmetry;
    if ((CR(crsMat)->nrows != CR(crsMat)->ncols) && (mat->traits->flags & GHOST_SPM_PERMUTECOLIDX)) { // TODO not here???    
        WARNING_LOG("Preventing column re-ordering as the matrix is not square!");
        mat->traits->flags &= ~GHOST_SPM_PERMUTECOLIDX;
    }
    mat->fromCRS(mat,crsMat->data);
    crsMat->destroy(crsMat);

#ifdef GHOST_HAVE_OPENCL
    if (!(mat->traits->flags & GHOST_SPM_HOST))
        mat->CLupload(mat);
#endif
#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPM_HOST))
        mat->CUupload(mat);
#endif

    DEBUG_LOG(1,"SELL matrix successfully created");
}

static void SELL_fromCRS(ghost_mat_t *mat, void *crs)
{
    SELL_fromCRS_funcs[ghost_dataTypeIdx(mat->traits->datatype)](mat,crs);
}

static void SELL_upload(ghost_mat_t* mat) 
{
    DEBUG_LOG(1,"Uploading SELL matrix to device");
#ifdef GHOST_HAVE_OPENCL
    if (!(mat->traits->flags & GHOST_SPM_HOST)) {
        DEBUG_LOG(1,"Creating matrix on OpenCL device");
        SELL(mat)->clmat = (CL_SELL_TYPE *)ghost_malloc(sizeof(CL_SELL_TYPE));
        SELL(mat)->clmat->rowLen = CL_allocDeviceMemory((SELL(mat)->nrows)*sizeof(ghost_cl_midx_t));
        SELL(mat)->clmat->col = CL_allocDeviceMemory((SELL(mat)->nEnts)*sizeof(ghost_cl_midx_t));
        SELL(mat)->clmat->val = CL_allocDeviceMemory((SELL(mat)->nEnts)*ghost_sizeofDataType(mat->traits->datatype));
        SELL(mat)->clmat->chunkStart = CL_allocDeviceMemory((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_cl_mnnz_t));
        SELL(mat)->clmat->chunkLen = CL_allocDeviceMemory((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_cl_midx_t));

        SELL(mat)->clmat->nrows = SELL(mat)->nrows;
        SELL(mat)->clmat->nrowsPadded = SELL(mat)->nrowsPadded;
        CL_copyHostToDevice(SELL(mat)->clmat->rowLen, SELL(mat)->rowLen, SELL(mat)->nrows*sizeof(ghost_cl_midx_t));
        CL_copyHostToDevice(SELL(mat)->clmat->col, SELL(mat)->col, SELL(mat)->nEnts*sizeof(ghost_cl_midx_t));
        CL_copyHostToDevice(SELL(mat)->clmat->val, SELL(mat)->val, SELL(mat)->nEnts*ghost_sizeofDataType(mat->traits->datatype));
        CL_copyHostToDevice(SELL(mat)->clmat->chunkStart, SELL(mat)->chunkStart, (SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_cl_mnnz_t));
        CL_copyHostToDevice(SELL(mat)->clmat->chunkLen, SELL(mat)->chunkLen, (SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_cl_midx_t));

        int nDigits = (int)log10(SELL(mat)->chunkHeight)+1;
        char options[128];
        char sellLenStr[32];
        snprintf(sellLenStr,32,"-DSELL_LEN=%d",SELL(mat)->chunkHeight);
        int sellLenStrlen = 11+nDigits;
        strncpy(options,sellLenStr,sellLenStrlen);


        cl_int err;
        cl_uint numKernels;

        if (mat->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
            if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
                strncpy(options+sellLenStrlen," -DGHOST_MAT_C",14);
            } else {
                strncpy(options+sellLenStrlen," -DGHOST_MAT_Z",14);
            }
        } else {
            if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
                strncpy(options+sellLenStrlen," -DGHOST_MAT_S",14);
            } else {
                strncpy(options+sellLenStrlen," -DGHOST_MAT_D",14);
            }

        }
        strncpy(options+sellLenStrlen+14," -DGHOST_VEC_S\0",15);
        cl_program program = CL_registerProgram("sell_clkernel.cl",options);
        CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
        DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
        mat->clkernel[0] = clCreateKernel(program,"SELL_kernel",&err);
        CL_checkerror(err);

        strncpy(options+sellLenStrlen+14," -DGHOST_VEC_D\0",15);
        program = CL_registerProgram("sell_clkernel.cl",options);
        CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
        DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
        mat->clkernel[1] = clCreateKernel(program,"SELL_kernel",&err);
        CL_checkerror(err);

        strncpy(options+sellLenStrlen+14," -DGHOST_VEC_C\0",15);
        program = CL_registerProgram("sell_clkernel.cl",options);
        CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
        DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
        mat->clkernel[2] = clCreateKernel(program,"SELL_kernel",&err);
        CL_checkerror(err);

        strncpy(options+sellLenStrlen+14," -DGHOST_VEC_Z\0",15);
        program = CL_registerProgram("sell_clkernel.cl",options);
        CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
        DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
        mat->clkernel[3] = clCreateKernel(program,"SELL_kernel",&err);
        CL_checkerror(err);

        int i;
        for (i=0; i<4; i++) {
            CL_safecall(clSetKernelArg(mat->clkernel[i],3,sizeof(ghost_cl_midx_t), &(SELL(mat)->clmat->nrows)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],4,sizeof(ghost_cl_midx_t), &(SELL(mat)->clmat->nrowsPadded)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],5,sizeof(cl_mem), &(SELL(mat)->clmat->rowLen)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],6,sizeof(cl_mem), &(SELL(mat)->clmat->col)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],7,sizeof(cl_mem), &(SELL(mat)->clmat->val)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],8,sizeof(cl_mem), &(SELL(mat)->clmat->chunkStart)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],9,sizeof(cl_mem), &(SELL(mat)->clmat->chunkLen)));
        }
        //    printf("### %lu\n",CL_getLocalSize(mat->clkernel));
        CL_checkerror(err);

    }
#else
    if (mat->traits->flags & GHOST_SPM_DEVICE) {
        ABORT("Device matrix cannot be created without OpenCL");
    }
#endif
}

static void SELL_CUupload(ghost_mat_t* mat) 
{
    DEBUG_LOG(1,"Uploading SELL matrix to CUDA device");
#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPM_HOST)) {
        DEBUG_LOG(1,"Creating matrix on CUDA device");
        SELL(mat)->cumat = (CU_SELL_TYPE *)ghost_malloc(sizeof(CU_SELL_TYPE));
        SELL(mat)->cumat->rowLen = CU_allocDeviceMemory((SELL(mat)->nrows)*sizeof(ghost_midx_t));
        SELL(mat)->cumat->rowLenPadded = CU_allocDeviceMemory((SELL(mat)->nrows)*sizeof(ghost_midx_t));
        SELL(mat)->cumat->col = CU_allocDeviceMemory((SELL(mat)->nEnts)*sizeof(ghost_midx_t));
        SELL(mat)->cumat->val = CU_allocDeviceMemory((SELL(mat)->nEnts)*ghost_sizeofDataType(mat->traits->datatype));
        SELL(mat)->cumat->chunkStart = CU_allocDeviceMemory((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_mnnz_t));
        SELL(mat)->cumat->chunkLen = CU_allocDeviceMemory((SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_midx_t));

        SELL(mat)->cumat->nrows = SELL(mat)->nrows;
        SELL(mat)->cumat->nrowsPadded = SELL(mat)->nrowsPadded;
        CU_copyHostToDevice(SELL(mat)->cumat->rowLen, SELL(mat)->rowLen, SELL(mat)->nrows*sizeof(ghost_midx_t));
        CU_copyHostToDevice(SELL(mat)->cumat->rowLenPadded, SELL(mat)->rowLenPadded, SELL(mat)->nrows*sizeof(ghost_midx_t));
        CU_copyHostToDevice(SELL(mat)->cumat->col, SELL(mat)->col, SELL(mat)->nEnts*sizeof(ghost_midx_t));
        CU_copyHostToDevice(SELL(mat)->cumat->val, SELL(mat)->val, SELL(mat)->nEnts*ghost_sizeofDataType(mat->traits->datatype));
        CU_copyHostToDevice(SELL(mat)->cumat->chunkStart, SELL(mat)->chunkStart, (SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_mnnz_t));
        CU_copyHostToDevice(SELL(mat)->cumat->chunkLen, SELL(mat)->chunkLen, (SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight)*sizeof(ghost_midx_t));
    }
#else
    if (mat->traits->flags & GHOST_SPM_DEVICE) {
        ABORT("Device matrix cannot be created without CUDA");
    }
#endif
}


static void SELL_free(ghost_mat_t *mat)
{
    free(SELL(mat)->val);
    free(SELL(mat)->col);
    free(SELL(mat)->chunkStart);
    free(SELL(mat)->chunkMin);
    free(SELL(mat)->chunkLen);
    free(SELL(mat)->rowLen);

    free(mat->data);

    free(mat);

}

static void SELL_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
    DEBUG_LOG(1,"Calling plain (maybe intrinsics) SELL kernel");
    DEBUG_LOG(2,"lhs vector has %s data and %"PRvecIDX" sub-vectors",ghost_datatypeName(lhs->traits->datatype),lhs->traits->nvecs);

    void (*kernel) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int) = NULL;

#if GHOST_HAVE_OPENMP
/*    static int first = 1;
    if ((mat->byteSize(mat) < ghost_getSizeOfLLC()) || (SELL(mat)->deviation*1./(ghost_getMatNnz(mat)*1.0/(double)ghost_getMatNrows(mat)) < 0.4)) {
        if (first) {
            INFO_LOG("Setting OpenMP scheduling to STATIC for SELL SpMVM kernel");
        }
        omp_set_schedule(omp_sched_static,0);
    } else {
        if (first) {
            INFO_LOG("Setting OpenMP scheduling to GUIDED,4 for SELL SpMVM kernel");
        }
        omp_set_schedule(omp_sched_guided,4);
    }
    first=0;*/
#endif


#if GHOST_HAVE_SSE
#ifndef LONGIDX
    if (!((options & GHOST_SPMVM_APPLY_SCALE) || (options & GHOST_SPMVM_APPLY_SHIFT))) {
    kernel = SELL_kernels_SSE
        [ghost_dataTypeIdx(mat->traits->datatype)]
        [ghost_dataTypeIdx(lhs->traits->datatype)];
}
#endif
#elif defined(AVX_INTR)
#ifndef LONGIDX
    if (!((options & GHOST_SPMVM_APPLY_SCALE) || (options & GHOST_SPMVM_APPLY_SHIFT))) {
    if (SELL(mat)->chunkHeight == 4) {
    kernel = SELL_kernels_AVX
        [ghost_dataTypeIdx(mat->traits->datatype)]
        [ghost_dataTypeIdx(lhs->traits->datatype)];
    } else if (SELL(mat)->chunkHeight == 32) {
    kernel = SELL_kernels_AVX_32
        [ghost_dataTypeIdx(mat->traits->datatype)]
        [ghost_dataTypeIdx(lhs->traits->datatype)];
    }
    }
#endif
#elif defined(MIC_INTR)
#ifndef LONGIDX
    if (!((options & GHOST_SPMVM_APPLY_SCALE) || (options & GHOST_SPMVM_APPLY_SHIFT))) {
    if (SELL(mat)->chunkHeight == 16) {
        kernel = SELL_kernels_MIC_16
            [ghost_dataTypeIdx(mat->traits->datatype)]
            [ghost_dataTypeIdx(lhs->traits->datatype)];
    } else if (SELL(mat)->chunkHeight == 32) {
        kernel = SELL_kernels_MIC_32
            [ghost_dataTypeIdx(mat->traits->datatype)]
            [ghost_dataTypeIdx(lhs->traits->datatype)];
    }
    }
#endif
#else
    kernel = SELL_kernels_plain
        [ghost_dataTypeIdx(mat->traits->datatype)]
        [ghost_dataTypeIdx(lhs->traits->datatype)];
#endif

    if (kernel == NULL) {
        WARNING_LOG("Selected kernel cannot be found. Falling back to plain C version!");
        kernel = SELL_kernels_plain
            [ghost_dataTypeIdx(mat->traits->datatype)]
            [ghost_dataTypeIdx(lhs->traits->datatype)];
    }

    kernel(mat,lhs,rhs,options);
}


#ifdef GHOST_HAVE_CUDA
static void SELL_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
    DEBUG_LOG(1,"Calling SELL CUDA kernel");
    DEBUG_LOG(2,"lhs vector has %s data",ghost_datatypeName(lhs->traits->datatype));

    /*if (lhs->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
      if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
      c_SELL_kernel_wrap(mat, lhs, rhs, options);
      else
      s_SELL_kernel_wrap(mat, lhs, rhs, options);
      } else {
      if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
      z_SELL_kernel_wrap(mat, lhs, rhs, options);
      else
      d_SELL_kernel_wrap(mat, lhs, rhs, options);
      }*/
    SELL_kernels_CU
        [ghost_dataTypeIdx(mat->traits->datatype)]
        [ghost_dataTypeIdx(lhs->traits->datatype)](mat,lhs,rhs,options);


}
#endif

#ifdef GHOST_HAVE_OPENCL
static void SELL_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
    cl_kernel kernel = mat->clkernel[ghost_dataTypeIdx(rhs->traits->datatype)];
    CL_safecall(clSetKernelArg(kernel,0,sizeof(cl_mem), &(lhs->CL_val_gpu)));
    CL_safecall(clSetKernelArg(kernel,1,sizeof(cl_mem), &(rhs->CL_val_gpu)));
    CL_safecall(clSetKernelArg(kernel,2,sizeof(int), &options));
    if (mat->traits->shift != NULL) {
        CL_safecall(clSetKernelArg(kernel,10,ghost_sizeofDataType(mat->traits->datatype), mat->traits->shift));
    } else {
        if (options & GHOST_SPMVM_APPLY_SHIFT)
            ABORT("A shift should be applied but the pointer is NULL!");
        complex double foo = 0.+I*0.; // should never be needed
        CL_safecall(clSetKernelArg(kernel,10,ghost_sizeofDataType(mat->traits->datatype), &foo))
    }

    size_t gSize = (size_t)SELL(mat)->clmat->nrowsPadded;
    size_t lSize = SELL(mat)->chunkHeight;

    CL_enqueueKernel(kernel,1,&gSize,&lSize);
}
#endif

#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * invec, int options)
{
    ghost_midx_t c,j;
    ghost_mnnz_t offs;
    vector double tmp;
    vector double val;
    vector double rhs;


#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
    for (c=0; c<SELL(mat)->nrowsPadded>>1; c++) 
    { // loop over chunks
        tmp = vec_splats(0.);
        offs = SELL(mat)->chunkStart[c];

        for (j=0; j<(SELL(mat)->chunkStart[c+1]-SELL(mat)->chunkStart[c])>>1; j++) 
        { // loop inside chunk
            val = vec_xld2(offs*sizeof(ghost_dt),SELL(mat)->val);                      // load values
            rhs = vec_insert(invec->val[SELL(mat)->col[offs++]],rhs,0);
            rhs = vec_insert(invec->val[SELL(mat)->col[offs++]],rhs,1);
            tmp = vec_madd(val,rhs,tmp);
        }
        if (options & GHOST_SPMVM_AXPY) {
            vec_xstd2(vec_add(tmp,vec_xld2(c*SELL(mat)->chunkHeight*sizeof(ghost_dt),lhs->val)),c*SELL(mat)->chunkHeight*sizeof(ghost_dt),lhs->val);
        } else {
            vec_xstd2(tmp,c*SELL(mat)->chunkHeight*sizeof(ghost_dt),lhs->val);
        }
    }
}
#endif
