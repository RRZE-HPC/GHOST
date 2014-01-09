#include "ghost/sell.h"
#include "ghost/crs.h"
#include "ghost/util.h"
#include "ghost/affinity.h"
#include "ghost/mat.h"
#include "ghost/constants.h"
#include "ghost/context.h"
#include "ghost/io.h"

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

void (*SELL_print_funcs[4]) (ghost_mat_t *) = 
{&s_SELL_print, &d_SELL_print, &c_SELL_print, &z_SELL_print}; 

static ghost_mnnz_t SELL_nnz(ghost_mat_t *mat);
static ghost_midx_t SELL_nrows(ghost_mat_t *mat);
static ghost_midx_t SELL_ncols(ghost_mat_t *mat);
static void SELL_printInfo(ghost_mat_t *mat);
static char * SELL_formatName(ghost_mat_t *mat);
static ghost_midx_t SELL_rowLen (ghost_mat_t *mat, ghost_midx_t i);
static size_t SELL_byteSize (ghost_mat_t *mat);
static void SELL_fromCRS(ghost_mat_t *mat, void *crs);
static void SELL_print(ghost_mat_t *mat);
static void SELL_split(ghost_mat_t *mat);
static void SELL_upload(ghost_mat_t* mat); 
static void SELL_CUupload(ghost_mat_t *mat);
static ghost_error_t SELL_fromBin(ghost_mat_t *mat, char *);
static void SELL_fromRowFunc(ghost_mat_t *mat, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, int flags);
static void SELL_free(ghost_mat_t *mat);
static void SELL_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
static int ghost_selectSellChunkHeight(int datatype);
#ifdef GHOST_HAVE_OPENCL
static void SELL_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef GHOST_HAVE_CUDA
static void SELL_kernel_CU (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options);
#endif
#ifdef VSX_INTR
static void SELL_kernel_VSX (ghost_mat_t *mat, ghost_vec_t *lhs, ghost_vec_t *rhs, int options);
#endif

ghost_mat_t * ghost_SELL_init(ghost_context_t *ctx, ghost_mtraits_t * traits)
{
    ghost_mat_t *mat = (ghost_mat_t *)ghost_malloc(sizeof(ghost_mat_t));
    mat->data = (SELL_TYPE *)ghost_malloc(sizeof(SELL_TYPE));
    mat->context = ctx;
    mat->traits = traits;
    DEBUG_LOG(1,"Setting functions for SELL matrix");
    if (!(mat->traits->flags & (GHOST_SPM_HOST | GHOST_SPM_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting matrix placement");
        if (ghost_type == GHOST_TYPE_CUDAMGMT) {
            mat->traits->flags |= GHOST_SPM_DEVICE;
        } else {
            mat->traits->flags |= GHOST_SPM_HOST;
        }
    }
    //TODO is it reasonable that a matrix has HOST&DEVICE?

    mat->CLupload = &SELL_upload;
    mat->CUupload = &SELL_CUupload;
    mat->fromFile = &SELL_fromBin;
    mat->fromRowFunc = &SELL_fromRowFunc;
    mat->printInfo = &SELL_printInfo;
    mat->formatName = &SELL_formatName;
    mat->rowLen     = &SELL_rowLen;
    mat->byteSize   = &SELL_byteSize;
    mat->spmv     = &SELL_kernel_plain;
    mat->fromCRS    = &SELL_fromCRS;
    mat->print    = &SELL_print;
    mat->split = &SELL_split;
#ifdef VSX_INTR
    mat->kernel = &SELL_kernel_VSX;
#endif
#ifdef GHOST_HAVE_OPENCL
    if (!(traits->flags & GHOST_SPM_HOST))
        mat->kernel   = &SELL_kernel_CL;
#endif
#if GHOST_HAVE_CUDA
    if (ghost_type == GHOST_TYPE_CUDAMGMT) {
        mat->spmv   = &SELL_kernel_CU;
    }
#endif
    mat->nnz      = &SELL_nnz;
    mat->nrows    = &SELL_nrows;
    mat->ncols    = &SELL_ncols;
    mat->destroy  = &SELL_free;

    mat->localPart = NULL;
    mat->remotePart = NULL;
   
    int me = ghost_getRank(mat->context->mpicomm);

    SELL(mat)->nrows = mat->context->lnrows[me];
   // SELL(mat)->ncols = mat->context->gncols;

    if (mat->traits->aux == NULL) {
        SELL(mat)->scope = 1;
        SELL(mat)->T = 1;
        SELL(mat)->chunkHeight = ghost_selectSellChunkHeight(mat->traits->datatype);
        SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,SELL(mat)->chunkHeight);
    } else {
        SELL(mat)->scope = *(int *)(mat->traits->aux);
        if (SELL(mat)->scope == GHOST_SELL_SORT_GLOBALLY) {
            SELL(mat)->scope = mat->context->lnrows[me];
        }

        if (mat->traits->nAux == 1 || ((int *)(mat->traits->aux))[1] == GHOST_SELL_CHUNKHEIGHT_AUTO) {
            SELL(mat)->chunkHeight = ghost_selectSellChunkHeight(mat->traits->datatype);
            SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,SELL(mat)->chunkHeight);
        } else {
            if (((int *)(mat->traits->aux))[1] == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
                SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,GHOST_PAD_MAX); // TODO padding anpassen an architektur
                SELL(mat)->chunkHeight = SELL(mat)->nrowsPadded;
            } else {
                SELL(mat)->chunkHeight = ((int *)(mat->traits->aux))[1];
                SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,SELL(mat)->chunkHeight);
            }
        }
        SELL(mat)->T = ((int *)(mat->traits->aux))[2];
    }
    SELL(mat)->nrowsPadded = ghost_pad(SELL(mat)->nrows,SELL(mat)->chunkHeight);;

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
{ UNUSED(mat);
    return 0;
}

static void SELL_printInfo(ghost_mat_t *mat)
{
    ghost_printLine("Max row length (# rows)",NULL,"%d (%d)",SELL(mat)->maxRowLen,SELL(mat)->nMaxRows);
    ghost_printLine("Chunk height (C)",NULL,"%d",SELL(mat)->chunkHeight);
    ghost_printLine("Chunk occupancy (beta)",NULL,"%f",SELL(mat)->beta);
    ghost_printLine("Row length variance",NULL,"%f",SELL(mat)->variance);
    ghost_printLine("Row length standard deviation",NULL,"%f",SELL(mat)->deviation);
    ghost_printLine("Row length coefficient of variation",NULL,"%f",SELL(mat)->cv);
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

static void SELL_fromRowFunc(ghost_mat_t *mat, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, int flags)
{
    if (SELL(mat)->scope > 1) {
        WARNING_LOG("Sorted SELL from Func not implemented");
    }
 //   WARNING_LOG("SELL-%d-%d from row func",SELL(mat)->chunkHeight,SELL(mat)->scope);
    UNUSED(base);
    UNUSED(flags);
    int nprocs = 1;
#if GHOST_HAVE_MPI
    nprocs = ghost_getNumberOfRanks(mat->context->mpicomm);
#endif
    
    ghost_midx_t i;
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);
    char *tmpval = ghost_malloc(SELL(mat)->chunkHeight*maxrowlen*sizeofdt);
    ghost_midx_t *tmpcol = (ghost_midx_t *)ghost_malloc(SELL(mat)->chunkHeight*maxrowlen*sizeof(ghost_midx_t));
    int me = ghost_getRank(mat->context->mpicomm);
    
    ghost_midx_t nChunks = SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight;
    SELL(mat)->chunkStart = (ghost_mnnz_t *)ghost_malloc((nChunks+1)*sizeof(ghost_mnnz_t));
    SELL(mat)->chunkMin = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->chunkLen = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->chunkLenPadded = (ghost_midx_t *)ghost_malloc((nChunks)*sizeof(ghost_midx_t));
    SELL(mat)->rowLen = (ghost_midx_t *)ghost_malloc((SELL(mat)->nrowsPadded)*sizeof(ghost_midx_t));
    SELL(mat)->rowLenPadded = (ghost_midx_t *)ghost_malloc((SELL(mat)->nrowsPadded)*sizeof(ghost_midx_t));
    SELL(mat)->chunkStart[0] = 0;
    SELL(mat)->maxRowLen = 0;
    SELL(mat)->nEnts = 0;
    SELL(mat)->nnz = 0;
   
    ghost_midx_t maxRowLenInChunk = 0;
    ghost_midx_t curChunk = 0;
    SELL(mat)->chunkStart[0] = 0;
    for( i = 0; i < SELL(mat)->nrowsPadded; i++ ) {

        if (i < SELL(mat)->nrows) {
            func(mat->context->lfRow[me]+i,&SELL(mat)->rowLen[i],tmpcol,tmpval);
        } else {
            SELL(mat)->rowLen[i] = 0;
        }
        
        SELL(mat)->rowLenPadded[i] = ghost_pad(SELL(mat)->rowLen[i],SELL(mat)->T);

        SELL(mat)->nnz += SELL(mat)->rowLen[i];
        maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[i]);

        if ((i+1)%SELL(mat)->chunkHeight == 0) {

            SELL(mat)->maxRowLen = MAX(SELL(mat)->maxRowLen,maxRowLenInChunk);
            SELL(mat)->chunkLen[curChunk] = maxRowLenInChunk;
            SELL(mat)->chunkLenPadded[curChunk] = ghost_pad(maxRowLenInChunk,SELL(mat)->T);
            SELL(mat)->nEnts += SELL(mat)->chunkLenPadded[curChunk]*SELL(mat)->chunkHeight;
            SELL(mat)->chunkStart[curChunk+1] = SELL(mat)->nEnts;
            maxRowLenInChunk = 0;
            curChunk++;
        }
    }
    
    SELL(mat)->val = (char *)ghost_malloc_align(ghost_sizeofDataType(mat->traits->datatype)*(size_t)SELL(mat)->nEnts,GHOST_DATA_ALIGNMENT);
    SELL(mat)->col = (ghost_midx_t *)ghost_malloc_align(sizeof(ghost_midx_t)*(size_t)SELL(mat)->nEnts,GHOST_DATA_ALIGNMENT);
 
    memset(tmpval,0,sizeofdt*maxrowlen*SELL(mat)->chunkHeight);
    memset(tmpcol,0,sizeof(ghost_midx_t)*maxrowlen*SELL(mat)->chunkHeight);
    curChunk = 0;
    ghost_midx_t rowInChunk = 0; 
    for( i = 0; i < SELL(mat)->nrowsPadded; i++ ) {

        if (i < SELL(mat)->nrows) {
            func(mat->context->lfRow[me]+i,&SELL(mat)->rowLen[i],&tmpcol[maxrowlen*rowInChunk],&tmpval[maxrowlen*rowInChunk*sizeofdt]);
        }
        rowInChunk++;

        if ((i+1)%SELL(mat)->chunkHeight == 0) {
            ghost_midx_t row,col;
            for (row = 0; row<SELL(mat)->chunkHeight; row++) {
                for (col = 0; col<SELL(mat)->chunkLenPadded[curChunk]; col++) {
                   memcpy(&SELL(mat)->val[sizeofdt*(SELL(mat)->chunkStart[curChunk]+col*SELL(mat)->chunkHeight+row)],&tmpval[sizeofdt*(row*maxrowlen+col)],sizeofdt);
                   memcpy(&SELL(mat)->col[SELL(mat)->chunkStart[curChunk]+col*SELL(mat)->chunkHeight+row],&tmpcol[row*maxrowlen+col],sizeof(ghost_midx_t));

                   //printf("%f ",((double *)(SELL(mat)->val))[(SELL(mat)->chunkStart[curChunk]+col*SELL(mat)->chunkHeight+row)]);
                }
                //printf("\n");
            }

            memset(tmpval,0,sizeofdt*maxrowlen*SELL(mat)->chunkHeight);
            memset(tmpcol,0,sizeof(ghost_midx_t)*maxrowlen*SELL(mat)->chunkHeight);
            curChunk++;
            rowInChunk = 0;
        }
    }
    
    if (!(mat->context->flags & GHOST_CONTEXT_GLOBAL)) {
#if GHOST_HAVE_MPI
        
        mat->context->lnEnts[me] = SELL(mat)->nEnts;

        ghost_mnnz_t nents[nprocs];
        nents[me] = mat->context->lnEnts[me];
        MPI_safecall(MPI_Bcast(&nents[me],1,ghost_mpi_dt_mnnz,me,mat->context->mpicomm));
        
        for (i=0; i<nprocs; i++) {
           mat->context->lfEnt[i] = 0;
        } 

        for (i=1; i<nprocs; i++) {
           mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+nents[i-1];
        } 

        mat->split(mat);
#endif
    }
    free(tmpval);
    free(tmpcol);

}

static void SELL_split(ghost_mat_t *mat)
    {
    SELL_TYPE *fullSELL = SELL(mat);
    SELL_TYPE *localSELL = NULL, *remoteSELL = NULL;
    DEBUG_LOG(1,"Splitting the SELL matrix into a local and remote part");
    int j;
    int i;
    int me;

    ghost_mnnz_t lnEnts_l, lnEnts_r;
    int current_l, current_r;

    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

    me = ghost_getRank(mat->context->mpicomm);
    ghost_midx_t chunk;
    ghost_midx_t idx, row,tmp;

    
    ghost_setupCommunication(mat->context,fullSELL->col);

   // mat->print(mat);
        
    mat->localPart = ghost_createMatrix(mat->context,&mat->traits[0],1);
    localSELL = mat->localPart->data;
    mat->localPart->symmetry = mat->symmetry;

    mat->remotePart = ghost_createMatrix(mat->context,&mat->traits[0],1);
    remoteSELL = mat->remotePart->data; 
        
    localSELL->T = fullSELL->T;
    remoteSELL->T = fullSELL->T;
    
    localSELL->chunkStart = (ghost_midx_t*) ghost_malloc((fullSELL->nrowsPadded/fullSELL->chunkHeight+1)*sizeof( ghost_midx_t )); 
    localSELL->chunkMin = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    localSELL->chunkLen = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    localSELL->chunkLenPadded = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    localSELL->rowLen = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded*sizeof( ghost_midx_t )); 
    localSELL->rowLenPadded = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded*sizeof( ghost_midx_t )); 

    remoteSELL->chunkStart = (ghost_midx_t*) ghost_malloc((fullSELL->nrowsPadded/fullSELL->chunkHeight+1)*sizeof( ghost_midx_t )); 
    remoteSELL->chunkMin = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    remoteSELL->chunkLen = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    remoteSELL->chunkLenPadded = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded/fullSELL->chunkHeight*sizeof( ghost_midx_t )); 
    remoteSELL->rowLen = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded*sizeof( ghost_midx_t )); 
    remoteSELL->rowLenPadded = (ghost_midx_t*) ghost_malloc(fullSELL->nrowsPadded*sizeof( ghost_midx_t )); 

    for (i=0; i<fullSELL->nrowsPadded; i++) {
        localSELL->rowLen[i] = 0;
        remoteSELL->rowLen[i] = 0;
        localSELL->rowLenPadded[i] = 0;
        remoteSELL->rowLenPadded[i] = 0;
    }
    for(chunk = 0; chunk < fullSELL->nrowsPadded/fullSELL->chunkHeight; chunk++) {
        localSELL->chunkLen[chunk] = 0;
        remoteSELL->chunkLen[chunk] = 0;
        localSELL->chunkLenPadded[chunk] = 0;
        remoteSELL->chunkLenPadded[chunk] = 0;
        localSELL->chunkMin[chunk] = 0;
        remoteSELL->chunkMin[chunk] = 0;
    }
    localSELL->chunkStart[0] = 0;
    remoteSELL->chunkStart[0] = 0;
  
    if (!(mat->context->flags & GHOST_CONTEXT_NO_SPLIT_SOLVERS)) { // split computation

        lnEnts_l = 0;
        lnEnts_r = 0;
        
        for(chunk = 0; chunk < fullSELL->nrowsPadded/fullSELL->chunkHeight; chunk++) {

            for (i=0; i<fullSELL->chunkLen[chunk]; i++) {
                for (j=0; j<fullSELL->chunkHeight; j++) {
                    row = chunk*fullSELL->chunkHeight+j;
                    idx = fullSELL->chunkStart[chunk]+i*fullSELL->chunkHeight+j;
                      
                    if (i < fullSELL->rowLen[row]) {
                        if (fullSELL->col[idx] < mat->context->lnrows[me]) {
                            localSELL->rowLen[row]++;
                        } else {
                            remoteSELL->rowLen[row]++;
                        }
                        localSELL->rowLenPadded[row] = ghost_pad(localSELL->rowLen[row],localSELL->T);
                        remoteSELL->rowLenPadded[row] = ghost_pad(remoteSELL->rowLen[row],remoteSELL->T);
                    }
                }
            }
           
            for (j=0; j<fullSELL->chunkHeight; j++) {
                row = chunk*fullSELL->chunkHeight+j;
                localSELL->chunkLen[chunk] = MAX(localSELL->chunkLen[chunk],localSELL->rowLen[row]);
                remoteSELL->chunkLen[chunk] = MAX(remoteSELL->chunkLen[chunk],remoteSELL->rowLen[row]);
            }
            lnEnts_l += localSELL->chunkLen[chunk]*fullSELL->chunkHeight;
            lnEnts_r += remoteSELL->chunkLen[chunk]*fullSELL->chunkHeight;
            localSELL->chunkStart[chunk+1] = lnEnts_l;
            remoteSELL->chunkStart[chunk+1] = lnEnts_r;

            localSELL->chunkLenPadded[chunk] = ghost_pad(localSELL->chunkLen[chunk],localSELL->T);
            remoteSELL->chunkLenPadded[chunk] = ghost_pad(remoteSELL->chunkLen[chunk],remoteSELL->T);
             
        }
            


        /*
        for (i=0; i<fullSELL->nEnts;i++) {
            if (fullSELL->col[i]<mat->context->lnrows[me]) lnEnts_l++;
        }
        lnEnts_r = mat->context->lnEnts[me]-lnEnts_l;*/
        

        localSELL->val = ghost_malloc(lnEnts_l*sizeofdt); 
        localSELL->col = (ghost_midx_t*) ghost_malloc(lnEnts_l*sizeof( ghost_midx_t )); 

        remoteSELL->val = ghost_malloc(lnEnts_r*sizeofdt); 
        remoteSELL->col = (ghost_midx_t*) ghost_malloc(lnEnts_r*sizeof( ghost_midx_t )); 

        localSELL->nrows = fullSELL->nrows;
        localSELL->nrowsPadded = fullSELL->nrowsPadded;
        localSELL->nEnts = lnEnts_l;
        localSELL->chunkHeight = fullSELL->chunkHeight;
        localSELL->scope = fullSELL->scope;

        remoteSELL->nrows = fullSELL->nrows;
        remoteSELL->nrowsPadded = fullSELL->nrowsPadded;
        remoteSELL->nEnts = lnEnts_r;
        remoteSELL->chunkHeight = fullSELL->chunkHeight;
        remoteSELL->scope = 1;

#pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < localSELL->nrowsPadded/localSELL->chunkHeight; chunk++) {
            for (i=0; i<localSELL->chunkLenPadded[chunk]; i++) {
                for (j=0; j<localSELL->chunkHeight; j++) {
                    idx = localSELL->chunkStart[chunk]+i*localSELL->chunkHeight+j;
                    memset(&((char *)(localSELL->val))[idx*sizeofdt],0,sizeofdt);
                    localSELL->col[idx] = 0;
                }
            }
        }

#pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < remoteSELL->nrowsPadded/remoteSELL->chunkHeight; chunk++) {
            for (i=0; i<remoteSELL->chunkLenPadded[chunk]; i++) {
                for (j=0; j<remoteSELL->chunkHeight; j++) {
                    idx = remoteSELL->chunkStart[chunk]+i*remoteSELL->chunkHeight+j;
                    memset(&((char *)(remoteSELL->val))[idx*sizeofdt],0,sizeofdt);
                    remoteSELL->col[idx] = 0;
                }
            }
        }

        current_l = 0;
        current_r = 0;
        ghost_midx_t col_l[fullSELL->chunkHeight], col_r[fullSELL->chunkHeight];
           
        for(chunk = 0; chunk < fullSELL->nrowsPadded/fullSELL->chunkHeight; chunk++) {
            
            for (j=0; j<fullSELL->chunkHeight; j++) {
                col_l[j] = 0;
                col_r[j] = 0;
            }

            for (i=0; i<fullSELL->chunkLen[chunk]; i++) {
                for (j=0; j<fullSELL->chunkHeight; j++) {
                    row = chunk*fullSELL->chunkHeight+j;
                    idx = fullSELL->chunkStart[chunk]+i*fullSELL->chunkHeight+j;
                   
                    if (i<fullSELL->rowLen[row]) {
                        if (fullSELL->col[idx] < mat->context->lnrows[me]) {
                            if (col_l[j] < localSELL->rowLen[row]) {
                                ghost_midx_t lidx = localSELL->chunkStart[chunk]+col_l[j]*localSELL->chunkHeight+j;
                                localSELL->col[lidx] = fullSELL->col[idx];
                                memcpy(&localSELL->val[lidx*sizeofdt],&fullSELL->val[idx*sizeofdt],sizeofdt);
                                current_l++;
                            }
                            col_l[j]++;
                        }
                        else{
                            if (col_r[j] < remoteSELL->rowLen[row]) {
                                ghost_midx_t ridx = remoteSELL->chunkStart[chunk]+col_r[j]*remoteSELL->chunkHeight+j;
                                remoteSELL->col[ridx] = fullSELL->col[idx];
                                memcpy(&remoteSELL->val[ridx*sizeofdt],&fullSELL->val[idx*sizeofdt],sizeofdt);
                                current_r++;
                            }
                            col_r[j]++;
                        }
                    }
                }
            }
        }
    }

/*    INFO_LOG("+ local +++++++");
    for(chunk = 0; chunk < localSELL->nrowsPadded/localSELL->chunkHeight; chunk++) {
        INFO_LOG("chunk %d, start %d, len %d",chunk,localSELL->chunkStart[chunk],localSELL->chunkLen[chunk]);
        for (i=0; i<localSELL->chunkLen[chunk]; i++) {
            for (j=0; j<localSELL->chunkHeight; j++) {
                idx = localSELL->chunkStart[chunk]+i*localSELL->chunkHeight+j;
                INFO_LOG("%d,%d,%d:%d: %f|%d",chunk,i,j,idx,((double *)(localSELL->val))[idx],localSELL->col[idx]);
            }
        }
    }
    
    INFO_LOG("+ remote +++++++");
    for(chunk = 0; chunk < remoteSELL->nrowsPadded/remoteSELL->chunkHeight; chunk++) {
        INFO_LOG("chunk %d, start %d, len %d",chunk,remoteSELL->chunkStart[chunk],remoteSELL->chunkLen[chunk]);
        for (i=0; i<remoteSELL->chunkLen[chunk]; i++) {
            for (j=0; j<remoteSELL->chunkHeight; j++) {
                idx = remoteSELL->chunkStart[chunk]+i*remoteSELL->chunkHeight+j;
                INFO_LOG("%d,%d,%d:%d: %f|%d",chunk,i,j,idx,((double *)(remoteSELL->val))[idx],remoteSELL->col[idx]);
            }
        }
    }*/

}

static ghost_error_t SELL_fromBin(ghost_mat_t *mat, char *matrixPath)
{
    DEBUG_LOG(1,"Creating SELL matrix from binary file");

    ghost_context_t *context = mat->context;
    int me = ghost_getRank(context->mpicomm); 
    int nprocs = ghost_getNumberOfRanks(context->mpicomm);

    ghost_midx_t *rpt;
    ghost_midx_t i;
    ghost_midx_t chunk,j;
    int swapReq = 0;
    ghost_matfile_header_t header;

    ghost_readMatFileHeader(matrixPath,&header);

    if (header.endianess == GHOST_BINCRS_LITTLE_ENDIAN && ghost_archIsBigEndian()) {
        DEBUG_LOG(1,"Need to convert from little to big endian.");
        swapReq = 1;
    } else if (header.endianess != GHOST_BINCRS_LITTLE_ENDIAN && !ghost_archIsBigEndian()) {
        DEBUG_LOG(1,"Need to convert from big to little endian.");
        swapReq = 1;
    } else {
        DEBUG_LOG(1,"OK, file and library have same endianess.");
    }

    if (header.version != 1)
        ABORT("Can not read version %d of binary CRS format!",header.version);

    if (header.base != 0)
        ABORT("Can not read matrix with %d-based indices!",header.base);

    if (!ghost_symmetryValid(header.symmetry))
        ABORT("Symmetry is invalid! (%d)",header.symmetry);
    if (header.symmetry != GHOST_BINCRS_SYMM_GENERAL)
        ABORT("Can not handle symmetry different to general at the moment!");
    mat->symmetry = header.symmetry;

    if (!ghost_datatypeValid(header.datatype))
        ABORT("Datatype is invalid! (%d)",header.datatype);

    SELL(mat)->nrows = context->lnrows[me];
    SELL(mat)->nrowsPadded = ghost_pad(context->lnrows[me],SELL(mat)->chunkHeight);
    SELL(mat)->ncols = context->gncols;
       
    if (ghost_getRank(mat->context->mpicomm) == 0) {
        if (context->flags & GHOST_CONTEXT_WORKDIST_ROWS) { // lnents and lfent have to be filled!
            rpt = CRS_readRpt(context->gnrows+1,matrixPath);  // read rpt
            
            INFO_LOG("Adjust lfrow and lnrows according to the SELL chunk height of %d",SELL(mat)->chunkHeight);
            for (i=0; i<nprocs; i++) {
                if (context->lfRow[i] % SELL(mat)->chunkHeight > SELL(mat)->chunkHeight/2) {
                    ghost_midx_t old = context->lfRow[i];
                    context->lfRow[i] += SELL(mat)->chunkHeight - (context->lfRow[i] % SELL(mat)->chunkHeight); 
                    INFO_LOG("PE%d: %d->%d",i,old,context->lfRow[i]); 
                } else if (context->lfRow[i] % SELL(mat)->chunkHeight > 0) {
                    ghost_midx_t old = context->lfRow[i];
                    context->lfRow[i] -= context->lfRow[i] % SELL(mat)->chunkHeight;
                    if (i>0 && (context->lfRow[i] == context->lfRow[i-1])) {
                        context->lfRow[i] += SELL(mat)->chunkHeight;
                    }
                    INFO_LOG("PE%d: %d->%d",i,old,context->lfRow[i]); 
                } else {
                    INFO_LOG("PE%d: %d",i,context->lfRow[i]);
                }
            }
            

            mat->context->lfEnt[0] = 0;

            for (i=1; i<nprocs; i++){
                mat->context->lfEnt[i] = rpt[mat->context->lfRow[i]];
            }
            for (i=0; i<nprocs-1; i++){
                mat->context->lnEnts[i] = mat->context->lfEnt[i+1] - mat->context->lfEnt[i] ;
            }

            mat->context->lnEnts[nprocs-1] = header.nnz - mat->context->lfEnt[nprocs-1];
        }
    }
    MPI_safecall(MPI_Bcast(mat->context->lfRow,  nprocs, ghost_mpi_dt_midx, 0, mat->context->mpicomm));
    MPI_safecall(MPI_Bcast(mat->context->lnrows, nprocs, ghost_mpi_dt_midx, 0, mat->context->mpicomm));
    MPI_safecall(MPI_Bcast(mat->context->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, mat->context->mpicomm));
    MPI_safecall(MPI_Bcast(mat->context->lnEnts, nprocs, ghost_mpi_dt_midx, 0, mat->context->mpicomm));
    
    SELL(mat)->nnz = context->lnEnts[me];

       
    ghost_midx_t nChunks =  SELL(mat)->nrowsPadded/SELL(mat)->chunkHeight;
    SELL(mat)->chunkStart = (ghost_midx_t *)ghost_malloc((nChunks+1)*sizeof(ghost_midx_t));
    SELL(mat)->chunkLen = (ghost_midx_t *)ghost_malloc(nChunks*sizeof(ghost_midx_t));
    SELL(mat)->chunkLenPadded = (ghost_midx_t *)ghost_malloc(nChunks*sizeof(ghost_midx_t));
    SELL(mat)->rowLen = (ghost_midx_t *)ghost_malloc(SELL(mat)->nrowsPadded*sizeof(ghost_midx_t));
    SELL(mat)->rowLenPadded = (ghost_midx_t *)ghost_malloc(SELL(mat)->nrowsPadded*sizeof(ghost_midx_t));

    if (ghost_getRank(context->mpicomm) != 0) {
        rpt = (ghost_midx_t *)ghost_malloc((context->lnrows[me]+1)*sizeof(ghost_midx_t));
    }
    MPI_Request req[nprocs];
    MPI_Status stat[nprocs];
    int msgcount = 0;

    for (i=0;i<nprocs;i++) 
        req[i] = MPI_REQUEST_NULL;

    if (ghost_getRank(context->mpicomm) != 0) {
        MPI_safecall(MPI_Irecv(rpt,context->lnrows[me]+1,ghost_mpi_dt_midx,0,me,context->mpicomm,&req[msgcount]));
        msgcount++;
    } else {
        for (i=1;i<nprocs;i++) {
            MPI_safecall(MPI_Isend(&rpt[context->lfRow[i]],context->lnrows[i]+1,ghost_mpi_dt_midx,i,i,context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    }
    MPI_safecall(MPI_Waitall(msgcount,req,stat));
    
    
    for (i=0;i<context->lnrows[me]+1;i++) {
        rpt[i] -= context->lfEnt[me]; 
    }


    ghost_midx_t maxRowLenInChunk = 0;
    ghost_midx_t curChunk = 0;

    SELL(mat)->chunkStart[0] = 0;    

    DEBUG_LOG(1,"Extracting row lenghts");
    for (i=0;i<SELL(mat)->nrowsPadded;i++) {
        if (i < SELL(mat)->nrows) {
            SELL(mat)->rowLen[i] = rpt[i+1]-rpt[i];
        } else {
            SELL(mat)->rowLen[i] = 0;
        }
        SELL(mat)->rowLenPadded[i] = ghost_pad(SELL(mat)->rowLen[i],SELL(mat)->T);
        INFO_LOG("rowlen[%d] = %d",context->lfRow[me]+i,SELL(mat)->rowLen[i]);
        
        maxRowLenInChunk = MAX(maxRowLenInChunk,SELL(mat)->rowLen[i]);

        if ((i+1)%SELL(mat)->chunkHeight == 0) {

            SELL(mat)->maxRowLen = MAX(SELL(mat)->maxRowLen,maxRowLenInChunk);
            SELL(mat)->chunkLen[curChunk] = maxRowLenInChunk;
            SELL(mat)->chunkLenPadded[curChunk] = ghost_pad(maxRowLenInChunk,SELL(mat)->T);
            SELL(mat)->nEnts += SELL(mat)->chunkLenPadded[curChunk]*SELL(mat)->chunkHeight;
            SELL(mat)->chunkStart[curChunk+1] = SELL(mat)->nEnts;
            maxRowLenInChunk = 0;
            curChunk++;
        }
    }
    
    mat->context->lnEnts[me] = SELL(mat)->nEnts;
    MPI_safecall(MPI_Bcast(&mat->context->lnEnts[me],1,ghost_mpi_dt_mnnz,me,mat->context->mpicomm));
    
    INFO_LOG("SELL matrix has %"PRmatIDX" (padded to %"PRmatIDX") rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros and %"PRmatNNZ" entries",SELL(mat)->nrows,SELL(mat)->nrowsPadded,SELL(mat)->ncols,SELL(mat)->nnz,mat->context->lnEnts[me]);
    
    SELL(mat)->val = (char *)ghost_malloc_align(ghost_sizeofDataType(mat->traits->datatype)*(size_t)SELL(mat)->nEnts,GHOST_DATA_ALIGNMENT);
    SELL(mat)->col = (ghost_midx_t *)ghost_malloc_align(sizeof(ghost_midx_t)*(size_t)SELL(mat)->nEnts,GHOST_DATA_ALIGNMENT);
    
#pragma omp parallel for schedule(runtime) private (i,j)
    for (chunk = 0; chunk < nChunks; chunk++) {
        for (j=0; j<SELL(mat)->chunkLenPadded[chunk]; j++) {
            for (i=0; i<SELL(mat)->chunkHeight; i++) {
                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = 0;
                SELL(mat)->val[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = 0;
            }
        }
    }

    FILE *filed;
    
    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s",matrixPath);
        return GHOST_ERR_IO;
    }


    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);


    ghost_midx_t *tmpcol = (ghost_midx_t *)ghost_malloc(SELL(mat)->maxRowLen*SELL(mat)->chunkHeight*sizeof(ghost_midx_t));
    char *tmpval = (char *)ghost_malloc(SELL(mat)->maxRowLen*SELL(mat)->chunkHeight*sizeofdt);
    for (chunk = 0; chunk < nChunks; chunk++) {
        memset(tmpcol,0,SELL(mat)->maxRowLen*SELL(mat)->chunkHeight*sizeof(ghost_midx_t));
        memset(tmpval,0,SELL(mat)->maxRowLen*SELL(mat)->chunkHeight*sizeofdt);

        ghost_midx_t firstNzOfChunk = context->lfEnt[me]+rpt[chunk*SELL(mat)->chunkHeight];
        ghost_midx_t nnzInChunk = rpt[(chunk+1)*SELL(mat)->chunkHeight]-rpt[chunk*SELL(mat)->chunkHeight];

        GHOST_SAFECALL(ghost_readColOpen(tmpcol,matrixPath,firstNzOfChunk,nnzInChunk,filed));
        GHOST_SAFECALL(ghost_readValOpen(tmpval,mat->traits->datatype,matrixPath,firstNzOfChunk,nnzInChunk,filed));

        ghost_midx_t idx = 0;
        for (i=0; i<SELL(mat)->chunkHeight; i++) {
            ghost_midx_t row = chunk*SELL(mat)->chunkHeight+i;
            for (j=0; j<SELL(mat)->rowLen[row]; j++) {
                SELL(mat)->col[SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i] = tmpcol[idx];
                memcpy(SELL(mat)->val+(SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i)*sizeofdt,&tmpval[idx*sizeofdt],sizeofdt);
                idx++;
            }
        }

    }
    

    fclose(filed);

    mat->split(mat);
    

#ifdef GHOST_HAVE_OPENCL
    if (!(mat->traits->flags & GHOST_SPM_HOST))
        mat->CLupload(mat);
#endif
#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits->flags & GHOST_SPM_HOST))
        mat->CUupload(mat);
#endif

    DEBUG_LOG(1,"SELL matrix successfully created");

    return GHOST_SUCCESS;
}

static void SELL_print(ghost_mat_t *mat)
{
    SELL_print_funcs[ghost_dataTypeIdx(mat->traits->datatype)](mat);
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
    if (!((options & GHOST_SPMVM_AXPBY) ||
            (options & GHOST_SPMVM_APPLY_SCALE) ||
            (options & GHOST_SPMVM_APPLY_SHIFT))) {
    kernel = SELL_kernels_SSE
        [ghost_dataTypeIdx(mat->traits->datatype)]
        [ghost_dataTypeIdx(lhs->traits->datatype)];
}
#endif
#elif GHOST_HAVE_AVX
#ifndef LONGIDX
    if (!((options & GHOST_SPMVM_AXPBY) ||
            (options & GHOST_SPMVM_APPLY_SCALE) ||
            (options & GHOST_SPMVM_APPLY_SHIFT))) {
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
#elif GHOST_HAVE_MIC
#ifndef LONGIDX
    if (!((options & GHOST_SPMVM_AXPBY) ||
            (options & GHOST_SPMVM_APPLY_SCALE) ||
            (options & GHOST_SPMVM_APPLY_SHIFT))) {
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
        //WARNING_LOG("Selected kernel cannot be found. Falling back to plain C version!");
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

static int ghost_selectSellChunkHeight(int datatype) {
    int ch = 1;

    if (datatype & GHOST_BINCRS_DT_FLOAT)
        ch *= 2;

    if (datatype & GHOST_BINCRS_DT_REAL)
        ch *= 2;

#ifdef AVX
    ch *= 2;
#endif

#ifdef MIC
    ch *= 4;
#ifndef LONGIDX
    ch *= 2;
#endif
#endif

#if defined (OPENCL) || defined (CUDA)
    ch = 256;
#endif

    return ch;
}

