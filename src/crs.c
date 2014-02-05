#define _XOPEN_SOURCE 500

#include "ghost/crs.h"
#include "ghost/util.h"
#include "ghost/mat.h"
#include "ghost/constants.h"
#include "ghost/affinity.h"
#include "ghost/context.h"
#include "ghost/io.h"
#include "ghost/log.h"

#include <unistd.h>
#include <sys/types.h>
#include <libgen.h>
#include <limits.h>
#include <errno.h>
#include <math.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <byteswap.h>

#include <dlfcn.h>

ghost_error_t (*CRS_kernels_plain[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t options) = 
{{&ss_CRS_kernel_plain,&sd_CRS_kernel_plain,&sc_CRS_kernel_plain,&sz_CRS_kernel_plain},
    {&ds_CRS_kernel_plain,&dd_CRS_kernel_plain,&dc_CRS_kernel_plain,&dz_CRS_kernel_plain},
    {&cs_CRS_kernel_plain,&cd_CRS_kernel_plain,&cc_CRS_kernel_plain,&cz_CRS_kernel_plain},
    {&zs_CRS_kernel_plain,&zd_CRS_kernel_plain,&zc_CRS_kernel_plain,&zz_CRS_kernel_plain}};

void (*CRS_valToStr_funcs[4]) (void *, char *, int) = 
{&s_CRS_valToStr,&d_CRS_valToStr,&c_CRS_valToStr,&z_CRS_valToStr};

const char * (*CRS_stringify_funcs[4]) (ghost_mat_t *, int) = 
{&s_CRS_stringify, &d_CRS_stringify, &c_CRS_stringify, &z_CRS_stringify}; 

static ghost_error_t CRS_fromBin(ghost_mat_t *mat, char *matrixPath);
static ghost_error_t CRS_toBin(ghost_mat_t *mat, char *matrixPath);
static ghost_error_t CRS_fromRowFunc(ghost_mat_t *mat, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, ghost_spmFromRowFunc_flags_t flags);
static ghost_error_t CRS_permute(ghost_mat_t *mat, ghost_midx_t *perm, ghost_midx_t *invPerm);
static void CRS_printInfo(ghost_mat_t *mat);
static const char * CRS_formatName(ghost_mat_t *mat);
static ghost_midx_t CRS_rowLen (ghost_mat_t *mat, ghost_midx_t i);
static size_t CRS_byteSize (ghost_mat_t *mat);
static const char * CRS_stringify(ghost_mat_t *mat, int dense);
static void CRS_free(ghost_mat_t * mat);
static ghost_error_t CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, ghost_spmv_flags_t);
static void CRS_fromCRS(ghost_mat_t *mat, ghost_mat_t *crsmat);
#ifdef GHOST_HAVE_MPI
static ghost_error_t CRS_split(ghost_mat_t *mat);
#endif
static void CRS_upload(ghost_mat_t *mat);


ghost_error_t ghost_CRS_init(ghost_context_t *ctx, ghost_mtraits_t *traits, ghost_mat_t **mat)
{
    *mat = (ghost_mat_t *)ghost_malloc(sizeof(ghost_mat_t));
    (*mat)->context = ctx;
    (*mat)->traits = traits;

    DEBUG_LOG(1,"Initializing CRS functions");
    if (!((*mat)->traits->flags & (GHOST_SPM_HOST | GHOST_SPM_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting matrix placement");
        (*mat)->traits->flags |= GHOST_SPM_HOST;
        if (ghost_type == GHOST_TYPE_CUDAMGMT) {
            (*mat)->traits->flags |= GHOST_SPM_DEVICE;
        }
    }

    if ((*mat)->traits->flags & GHOST_SPM_DEVICE)
    {
#if GHOST_HAVE_CUDA
        WARNING_LOG("CUDA CRS SpMV has not yet been implemented!");
        //   mat->spmv = &ghost_cu_crsspmv;
#endif
    }
    else if ((*mat)->traits->flags & GHOST_SPM_HOST)
    {
        (*mat)->spmv   = &CRS_kernel_plain;
    }

    (*mat)->fromFile = &CRS_fromBin;
    (*mat)->toFile = &CRS_toBin;
    (*mat)->fromRowFunc = &CRS_fromRowFunc;
    (*mat)->fromCRS = &CRS_fromCRS;
    (*mat)->printInfo = &CRS_printInfo;
    (*mat)->formatName = &CRS_formatName;
    (*mat)->rowLen   = &CRS_rowLen;
    (*mat)->byteSize = &CRS_byteSize;
    (*mat)->permute = &CRS_permute;
    (*mat)->destroy  = &CRS_free;
    (*mat)->stringify = &CRS_stringify;
#ifdef GHOST_HAVE_MPI
    (*mat)->split = &CRS_split;
#endif
    (*mat)->data = (CR_TYPE *)ghost_malloc(sizeof(CR_TYPE));

    //    mat->rowPerm = NULL;
    //    mat->invRowPerm = NULL;
    (*mat)->localPart = NULL;
    (*mat)->remotePart = NULL;
    (*mat)->name = NULL;

    (*mat)->bandwidth = 0;
    (*mat)->lowerBandwidth = 0;
    (*mat)->upperBandwidth = 0;
    (*mat)->nzDist = NULL;

    return GHOST_SUCCESS;

}

static ghost_error_t CRS_permute(ghost_mat_t *mat, ghost_midx_t *perm, ghost_midx_t *invPerm)
{
    if (perm == NULL) {
        return GHOST_SUCCESS;
    }
    if (mat->data == NULL) {
        ERROR_LOG("The matrix data to be permuted is NULL");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_midx_t i,j,c;
    ghost_midx_t rowLen;
    CR_TYPE *cr = CR(mat);

    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

    ghost_mnnz_t *rpt_perm = (ghost_midx_t *)ghost_malloc((mat->nrows+1)*sizeof(ghost_midx_t));
    ghost_mnnz_t *col_perm = (ghost_midx_t *)ghost_malloc(mat->nnz*sizeof(ghost_midx_t));
    char *val_perm = (char *)ghost_malloc(mat->nnz*sizeofdt);

    /*for (i=0; i<mat->nrows; i++) {
      printf("perm/inv[%"PRmatIDX"] = %"PRmatIDX" %"PRmatIDX"\n",i,perm[i],invPerm[i]);
      }*/


    rpt_perm[0] = 0;
    for (i=1; i<mat->nrows+1; i++) {
        rowLen = cr->rpt[invPerm[i-1]+1]-cr->rpt[invPerm[i-1]];
        rpt_perm[i] = rpt_perm[i-1]+rowLen;
        //printf("rpt_perm[%"PRmatIDX"] = %"PRmatIDX", rowLen: %"PRmatIDX"\n",i,rpt_perm[i],rowLen);
    }
    if (rpt_perm[mat->nrows] != mat->nnz) {
        free(rpt_perm);
        free(col_perm);
        free(val_perm);

        ERROR_LOG("Error in row pointer permutation: %"PRmatIDX" != %"PRmatIDX,rpt_perm[mat->nrows],mat->nnz);
        return GHOST_ERR_UNKNOWN;
    }

    mat->bandwidth = 0;
    for (i = 0; i < mat->nrows; i++) {
        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
        }
    }
    mat->lowerBandwidth = 0;
    mat->upperBandwidth = 0;
    ghost_midx_t col;
    memset(mat->nzDist,0,sizeof(ghost_mnnz_t)*(2*mat->nrows-1));
    for (i=0; i<mat->nrows; i++) {
        rowLen = rpt_perm[i+1]-rpt_perm[i];
        memcpy(&val_perm[rpt_perm[i]*sizeofdt],&cr->val[cr->rpt[invPerm[i]]*sizeofdt],rowLen*sizeofdt);
        //memcpy(&col_perm[rpt_perm[i]],&cr->col[cr->rpt[invPerm[i]]],rowLen*sizeof(ghost_midx_t));
        for (j=rpt_perm[i], c=0; j<rpt_perm[i+1]; j++, c++) {
            col_perm[j] = perm[cr->col[cr->rpt[invPerm[i]]+c]];
            col = col_perm[j];
            if (col < i) {
                mat->lowerBandwidth = MAX(mat->lowerBandwidth, i-col);
                mat->nzDist[mat->nrows-1-(i-col)]++;
            } else if (col > i) {
                mat->upperBandwidth = MAX(mat->upperBandwidth, col-i);
                mat->nzDist[mat->nrows-1+col-i]++;
            } else {
                mat->nzDist[mat->nrows-1]++;
            }
        } 
        ghost_midx_t n;
        ghost_midx_t tmpcol;
        char *tmpval = ghost_malloc(sizeofdt);
        for (n=rowLen; n>1; n--) {
            for (j=rpt_perm[i]; j<rpt_perm[i]+n-1; j++) {
                if (col_perm[j] > col_perm[j+1]) {
                    tmpcol = col_perm[j];
                    col_perm[j] = col_perm[j+1];
                    col_perm[j+1] = tmpcol;

                    memcpy(&tmpval,&val_perm[sizeofdt*j],sizeofdt);
                    memcpy(&val_perm[sizeofdt*j],&val_perm[sizeofdt*(j+1)],sizeofdt);
                    memcpy(&val_perm[sizeofdt*(j+1)],&tmpval,sizeofdt);
                }
            }
        }
    }
    mat->bandwidth = mat->lowerBandwidth+mat->upperBandwidth+1;

    free(cr->rpt);
    free(cr->col);
    free(cr->val);

    cr->rpt = rpt_perm;
    cr->col = col_perm;
    cr->val = val_perm;


    return GHOST_SUCCESS;

}

static const char * CRS_stringify(ghost_mat_t *mat, int dense)
{
    return CRS_stringify_funcs[ghost_dataTypeIdx(mat->traits->datatype)](mat, dense);
}

static void CRS_printInfo(ghost_mat_t *mat)
{
    UNUSED(mat);
    return;
}

static const char * CRS_formatName(ghost_mat_t *mat)
{
    UNUSED(mat);
    return "CRS";
}

static ghost_midx_t CRS_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
    if (mat && i<mat->nrows) {
        return CR(mat)->rpt[i+1] - CR(mat)->rpt[i];
    }

    return 0;
}

static size_t CRS_byteSize (ghost_mat_t *mat)
{
    if (mat->data == NULL) {
        return 0;
    }

    return (size_t)((mat->nrows+1)*sizeof(ghost_mnnz_t) + 
            mat->nEnts*(sizeof(ghost_midx_t)+ghost_sizeofDataType(mat->traits->datatype)));
}

static ghost_error_t CRS_fromRowFunc(ghost_mat_t *mat, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, ghost_spmFromRowFunc_flags_t flags)
{
    UNUSED(base);
    UNUSED(flags);
    int nprocs = 1;
    int me;
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs));
    GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&me));

    ghost_midx_t rowlen;
    ghost_midx_t i,j;
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);
    mat->ncols = mat->context->gncols;
    mat->nrows = mat->context->lnrows[me];
    CR(mat)->rpt = (ghost_midx_t *)ghost_malloc((mat->nrows+1)*sizeof(ghost_midx_t));
    mat->nEnts = 0;

#pragma omp parallel for schedule(runtime)
    for (i = 0; i < mat->nrows+1; i++) {
        CR(mat)->rpt[i] = 0;
    }

    ghost_mnnz_t nEnts = 0;

#pragma omp parallel private(i,rowlen) reduction (+:nEnts)
    { 
        char * tmpval = ghost_malloc(maxrowlen*sizeofdt);
        ghost_midx_t * tmpcol = (ghost_midx_t *)ghost_malloc(maxrowlen*sizeof(ghost_midx_t));
#pragma omp for ordered
        for( i = 0; i < mat->nrows; i++ ) {
            func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
            nEnts += rowlen;
#pragma omp ordered
            CR(mat)->rpt[i+1] = CR(mat)->rpt[i]+rowlen;
        }
        free(tmpval);
        free(tmpcol);
    }


    mat->nEnts = nEnts;
    mat->nnz = mat->nEnts;

    CR(mat)->col = (ghost_midx_t *)ghost_malloc(mat->nEnts*sizeof(ghost_midx_t));
    CR(mat)->val = ghost_malloc(mat->nEnts*sizeofdt);

#pragma omp parallel for schedule(runtime) private (j)
    for (i = 0; i < mat->nrows; i++) {
        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
            CR(mat)->col[j] = 0;
            memset(&CR(mat)->val[j*sizeofdt],0,sizeofdt);
        }
    }

    // TODO load balancing if distribution by nnz

#pragma omp parallel private(i,rowlen)
    { 
        char * tmpval = ghost_malloc(maxrowlen*sizeofdt);
        ghost_midx_t * tmpcol = (ghost_midx_t *)ghost_malloc(maxrowlen*sizeof(ghost_midx_t));
        memset(tmpval,0,sizeofdt*maxrowlen);
        memset(tmpcol,0,sizeof(ghost_midx_t)*maxrowlen);
#pragma omp for schedule(runtime)
        for( i = 0; i < mat->nrows; i++ ) {
            func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
            memcpy(&CR(mat)->col[CR(mat)->rpt[i]],tmpcol,rowlen*sizeof(ghost_midx_t));
            memcpy(&((char *)CR(mat)->val)[CR(mat)->rpt[i]*sizeofdt],tmpval,rowlen*sizeofdt);
        }
        free(tmpval);
        free(tmpcol);
    }

    if (!(mat->context->flags & GHOST_CONTEXT_GLOBAL)) {
#if GHOST_HAVE_MPI

        mat->context->lnEnts[me] = mat->nEnts;

        ghost_mnnz_t nents;
        nents = mat->context->lnEnts[me];
        MPI_safecall(MPI_Allgather(&nents,1,ghost_mpi_dt_mnnz,mat->context->lnEnts,1,ghost_mpi_dt_mnnz,mat->context->mpicomm));

        for (i=0; i<nprocs; i++) {
            mat->context->lfEnt[i] = 0;
        } 

        for (i=1; i<nprocs; i++) {
            mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
        } 

        mat->split(mat);
#endif
    }
    mat->nrows = mat->context->lnrows[me];

    return GHOST_SUCCESS;
}

static void CRS_fromCRS(ghost_mat_t *mat, ghost_mat_t *crsmat)
{
    DEBUG_LOG(1,"Creating CRS matrix from CRS matrix");
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);
    CR_TYPE *cr = (CR_TYPE*)(crsmat->data);
    ghost_midx_t i,j;


    //    mat->data = (CR_TYPE *)ghost_malloc(sizeof(CR_TYPE));
    mat->nrows = crsmat->nrows;
    mat->ncols = crsmat->ncols;
    mat->nEnts = crsmat->nEnts;

    CR(mat)->rpt = (ghost_midx_t *)ghost_malloc((crsmat->nrows+1)*sizeof(ghost_midx_t));
    CR(mat)->col = (ghost_midx_t *)ghost_malloc(crsmat->nEnts*sizeof(ghost_midx_t));
    CR(mat)->val = ghost_malloc(crsmat->nEnts*sizeofdt);

#pragma omp parallel for schedule(runtime)
    for( i = 0; i < mat->nrows+1; i++ ) {
        CR(mat)->rpt[i] = cr->rpt[i];
    }

#pragma omp parallel for schedule(runtime) private(j)
    for( i = 0; i < mat->nrows; i++ ) {
        for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
            CR(mat)->col[j] = cr->col[j];
            memcpy(&CR(mat)->val[j*sizeofdt],&cr->val[j*sizeofdt],sizeofdt);
        }
    }

    DEBUG_LOG(1,"Successfully created CRS matrix from CRS data");

}

#ifdef GHOST_HAVE_MPI

static ghost_error_t CRS_split(ghost_mat_t *mat)
{
    if (!mat) {
        ERROR_LOG("Matrix is NULL");
        return GHOST_ERR_INVALID_ARG;
    }

    CR_TYPE *fullCR = CR(mat);
    CR_TYPE *localCR = NULL, *remoteCR = NULL;
    DEBUG_LOG(1,"Splitting the CRS matrix into a local and remote part");
    int j;
    int i;
    int me;

    ghost_mnnz_t lnEnts_l, lnEnts_r;
    int current_l, current_r;

    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

    GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&me));

    ghost_setupCommunication(mat->context,fullCR->col);

    if (!(mat->context->flags & GHOST_CONTEXT_NO_SPLIT_SOLVERS)) { // split computation

        lnEnts_l=0;
        for (i=0; i<mat->context->lnEnts[me];i++) {
            if (fullCR->col[i]<mat->context->lnrows[me]) lnEnts_l++;
        }


        lnEnts_r = mat->context->lnEnts[me]-lnEnts_l;

        DEBUG_LOG(1,"PE%d: Rows=%"PRmatIDX"\t Ents=%"PRmatNNZ"(l),%"PRmatNNZ"(r),%"PRmatNNZ"(g)\t pdim=%"PRmatIDX, 
                me, mat->context->lnrows[me], lnEnts_l, lnEnts_r, mat->context->lnEnts[me],mat->context->lnrows[me]+mat->context->halo_elements  );

        localCR = (CR_TYPE *) ghost_malloc(sizeof(CR_TYPE));
        remoteCR = (CR_TYPE *) ghost_malloc(sizeof(CR_TYPE));
        ghost_createMatrix(mat->context,&mat->traits[0],1,&(mat->localPart));
        free(mat->localPart->data); // has been allocated in init()
        mat->localPart->traits->symmetry = mat->traits->symmetry;
        mat->localPart->data = localCR;
        //CR(mat->localPart)->rpt = localCR->rpt;

        ghost_createMatrix(mat->context,&mat->traits[0],1,&(mat->remotePart));
        free(mat->remotePart->data); // has been allocated in init()
        mat->remotePart->data = remoteCR;

        localCR->val = ghost_malloc(lnEnts_l*sizeofdt); 
        localCR->col = (ghost_midx_t*) ghost_malloc(lnEnts_l*sizeof( ghost_midx_t )); 
        localCR->rpt = (ghost_midx_t*) ghost_malloc((mat->context->lnrows[me]+1)*sizeof( ghost_midx_t )); 

        remoteCR->val = ghost_malloc(lnEnts_r*sizeofdt); 
        remoteCR->col = (ghost_midx_t*) ghost_malloc(lnEnts_r*sizeof( ghost_midx_t )); 
        remoteCR->rpt = (ghost_midx_t*) ghost_malloc((mat->context->lnrows[me]+1)*sizeof( ghost_midx_t )); 

        mat->localPart->nrows = mat->context->lnrows[me];
        mat->localPart->nEnts = lnEnts_l;
        mat->localPart->nnz = lnEnts_l;

        mat->remotePart->nrows = mat->context->lnrows[me];
        mat->remotePart->nEnts = lnEnts_r;
        mat->remotePart->nnz = lnEnts_r;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_l; i++) localCR->val[i*sizeofdt] = 0;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_l; i++) localCR->col[i] = 0.0;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_r; i++) remoteCR->val[i*sizeofdt] = 0;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_r; i++) remoteCR->col[i] = 0.0;


        localCR->rpt[0] = 0;
        remoteCR->rpt[0] = 0;

        MPI_safecall(MPI_Barrier(mat->context->mpicomm));
        DEBUG_LOG(1,"PE%d: lnrows=%"PRmatIDX" row_ptr=%"PRmatIDX"..%"PRmatIDX,
                me, mat->context->lnrows[me], fullCR->rpt[0], fullCR->rpt[mat->context->lnrows[me]]);
        fflush(stdout);
        MPI_safecall(MPI_Barrier(mat->context->mpicomm));

        for (i=0; i<mat->context->lnrows[me]; i++){

            current_l = 0;
            current_r = 0;

            for (j=fullCR->rpt[i]; j<fullCR->rpt[i+1]; j++){

                if (fullCR->col[j]<mat->context->lnrows[me]){
                    localCR->col[ localCR->rpt[i]+current_l ] = fullCR->col[j];
                    memcpy(&localCR->val[(localCR->rpt[i]+current_l)*sizeofdt],&fullCR->val[j*sizeofdt],sizeofdt);
                    current_l++;
                }
                else{
                    remoteCR->col[ remoteCR->rpt[i]+current_r ] = fullCR->col[j];
                    memcpy(&remoteCR->val[(remoteCR->rpt[i]+current_r)*sizeofdt],&fullCR->val[j*sizeofdt],sizeofdt);
                    current_r++;
                }

            }  

            localCR->rpt[i+1] = localCR->rpt[i] + current_l;
            remoteCR->rpt[i+1] = remoteCR->rpt[i] + current_r;
        }

        IF_DEBUG(3){
            for (i=0; i<mat->context->lnrows[me]+1; i++)
                DEBUG_LOG(3,"--Row_ptrs-- PE %d: i=%d local=%"PRmatIDX" remote=%"PRmatIDX, 
                        me, i, localCR->rpt[i], remoteCR->rpt[i]);
            for (i=0; i<localCR->rpt[mat->context->lnrows[me]]; i++)
                DEBUG_LOG(3,"-- local -- PE%d: localCR->col[%d]=%"PRmatIDX, me, i, localCR->col[i]);
            for (i=0; i<remoteCR->rpt[mat->context->lnrows[me]]; i++)
                DEBUG_LOG(3,"-- remote -- PE%d: remoteCR->col[%d]=%"PRmatIDX, me, i, remoteCR->col[i]);
        }
        fflush(stdout);
        MPI_safecall(MPI_Barrier(mat->context->mpicomm));

    }


    return GHOST_SUCCESS;

}

#endif

/*int compareNZEPos( const void* a, const void* b ) 
  {

  int aRow = ((NZE_TYPE*)a)->row,
  bRow = ((NZE_TYPE*)b)->row,
  aCol = ((NZE_TYPE*)a)->col,
  bCol = ((NZE_TYPE*)b)->col;

  if( aRow == bRow ) {
  return aCol - bCol;
  }
  else return aRow - bRow;
  }*/

/**
 * @brief Creates a CRS matrix from a binary file.
 *
 * @param mat The matrix.
 * @param matrixPath Path to the file.
 *
 * If the row pointers have already been read-in and stored in the context
 * they will not be read in again.
 */
static ghost_error_t CRS_fromBin(ghost_mat_t *mat, char *matrixPath)
{
    DEBUG_LOG(1,"Reading CRS matrix from file");
    mat->name = basename(matrixPath);
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

    ghost_midx_t i;
    ghost_mnnz_t j;

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
    mat->traits->symmetry = header.symmetry;

    if (!ghost_datatypeValid(header.datatype))
        ABORT("Datatype is invalid! (%d)",header.datatype);

    mat->ncols = (ghost_midx_t)header.ncols;

    DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",mat->nrows,mat->ncols,mat->nEnts);

    if (mat->context->flags & GHOST_CONTEXT_GLOBAL) {
        CR(mat)->rpt = (ghost_mnnz_t *) ghost_malloc_align((mat->nrows+1) * sizeof(ghost_mnnz_t), GHOST_DATA_ALIGNMENT);
        CR(mat)->col = (ghost_midx_t *) ghost_malloc_align(mat->nEnts * sizeof(ghost_midx_t), GHOST_DATA_ALIGNMENT);
        CR(mat)->val = ghost_malloc_align(mat->nEnts * sizeofdt,GHOST_DATA_ALIGNMENT);

#pragma omp parallel for schedule(runtime)
        for (i = 0; i < mat->nrows+1; i++) {
            CR(mat)->rpt[i] = 0;
        }

        GHOST_CALL_RETURN(ghost_readRpt(CR(mat)->rpt, matrixPath, 0, mat->nrows+1));

#pragma omp parallel for schedule(runtime) private (j)
        for (i = 0; i < mat->nrows; i++) {
            for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                CR(mat)->col[j] = 0;
                memset(&CR(mat)->val[j*sizeofdt],0,sizeofdt);
            }
        }

        mat->nrows = (ghost_midx_t)header.nrows;
        mat->nEnts = (ghost_midx_t)header.nnz;
        mat->nnz = mat->nEnts;

        GHOST_CALL_RETURN(ghost_readCol(CR(mat)->col, matrixPath, 0, mat->nEnts));
        GHOST_CALL_RETURN(ghost_readVal(CR(mat)->val, mat->traits->datatype, matrixPath, 0, mat->nEnts));


    } else {
#ifdef GHOST_HAVE_MPI
        DEBUG_LOG(1,"Reading in a distributed context");
        DEBUG_LOG(1,"Creating distributed context with parallel MPI-IO");

        ghost_context_t *context = mat->context;
        int nprocs = 1;
        int me;
        GHOST_CALL_RETURN(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs));
        GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&me));

        if (me == 0) {
            if (context->flags & GHOST_CONTEXT_WORKDIST_NZE) { // rpt has already been read
                ((CR_TYPE *)(mat->data))->rpt = context->rpt;
            } else {
                CR(mat)->rpt = (ghost_mnnz_t *) ghost_malloc_align((header.nrows+1) * sizeof(ghost_mnnz_t), GHOST_DATA_ALIGNMENT);
#pragma omp parallel for schedule(runtime) 
                for (i = 0; i < header.nrows+1; i++) {
                    CR(mat)->rpt[i] = 0;
                }
                GHOST_CALL_RETURN(ghost_readRpt(CR(mat)->rpt, matrixPath, 0, header.nrows+1));
                context->lfEnt[0] = 0;

                for (i=1; i<nprocs; i++){
                    context->lfEnt[i] = CR(mat)->rpt[context->lfRow[i]];
                }
                for (i=0; i<nprocs-1; i++){
                    context->lnEnts[i] = context->lfEnt[i+1] - context->lfEnt[i] ;
                }

                context->lnEnts[nprocs-1] = header.nnz - context->lfEnt[nprocs-1];
            }
        }
        MPI_safecall(MPI_Bcast(context->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
        MPI_safecall(MPI_Bcast(context->lnEnts, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));

        mat->nnz = context->lnEnts[me];
        mat->nEnts = mat->nnz;
        mat->nrows = context->lnrows[me];

        DEBUG_LOG(1,"Mallocing space for %"PRmatIDX" rows",context->lnrows[me]);

        if (me != 0) {
            CR(mat)->rpt = (ghost_midx_t *)ghost_malloc_align((context->lnrows[me]+1)*sizeof(ghost_midx_t),GHOST_DATA_ALIGNMENT);
#pragma omp parallel for schedule(runtime)
            for (i = 0; i < context->lnrows[me]+1; i++) {
                CR(mat)->rpt[i] = 0;
            }
        }

        MPI_Request req[nprocs];
        MPI_Status stat[nprocs];
        int msgcount = 0;

        for (i=0;i<nprocs;i++) 
            req[i] = MPI_REQUEST_NULL;

        if (me != 0) {
            MPI_safecall(MPI_Irecv(CR(mat)->rpt,context->lnrows[me]+1,ghost_mpi_dt_midx,0,me,context->mpicomm,&req[msgcount]));
            msgcount++;
        } else {
            for (i=1;i<nprocs;i++) {
                MPI_safecall(MPI_Isend(&CR(mat)->rpt[context->lfRow[i]],context->lnrows[i]+1,ghost_mpi_dt_midx,i,i,context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
        MPI_safecall(MPI_Waitall(msgcount,req,stat));

        DEBUG_LOG(1,"Adjusting row pointers");
        for (i=0;i<context->lnrows[me]+1;i++) {
            CR(mat)->rpt[i] -= context->lfEnt[me]; 
        }

        CR(mat)->rpt[context->lnrows[me]] = context->lnEnts[me];

        DEBUG_LOG(1,"local rows          = %"PRmatIDX,context->lnrows[me]);
        DEBUG_LOG(1,"local rows (offset) = %"PRmatIDX,context->lfRow[me]);
        DEBUG_LOG(1,"local entries          = %"PRmatNNZ,context->lnEnts[me]);
        DEBUG_LOG(1,"local entires (offset) = %"PRmatNNZ,context->lfEnt[me]);

        mat->nrows = context->lnrows[me];
        mat->nEnts = context->lnEnts[me];

        CR(mat)->col = (ghost_midx_t *) ghost_malloc_align(mat->nEnts * sizeof(ghost_midx_t), GHOST_DATA_ALIGNMENT);
        CR(mat)->val = ghost_malloc_align(mat->nEnts * sizeofdt,GHOST_DATA_ALIGNMENT);

#pragma omp parallel for schedule(runtime) private (j)
        for (i = 0; i < mat->nrows; i++) {
            for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                CR(mat)->col[j] = 0;
                memset(&CR(mat)->val[j*sizeofdt],0,sizeofdt);
            }
        }

        GHOST_CALL_RETURN(ghost_readCol(CR(mat)->col, matrixPath, context->lfEnt[me], mat->nEnts));
        GHOST_CALL_RETURN(ghost_readVal(CR(mat)->val, mat->traits->datatype, matrixPath, context->lfEnt[me], mat->nEnts));

        mat->nzDist = (ghost_mnnz_t *)ghost_malloc(sizeof(ghost_mnnz_t)*(2*mat->nrows-1));
        memset(mat->nzDist,0,sizeof(ghost_mnnz_t)*(2*mat->nrows-1));

        ghost_midx_t col;
    
        mat->lowerBandwidth = 0;
        mat->upperBandwidth = 0;
        for (i = 0; i < mat->nrows; i++) {
            for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                col = CR(mat)->col[j];
                if (col >= mat->nrows) {
                    continue;
                }
                if (col < i) { // lower
                    mat->lowerBandwidth = MAX(mat->lowerBandwidth, i-col);
                    mat->nzDist[mat->nrows-1-(i-col)]++;
                } else if (col > i) { // upper
                    mat->upperBandwidth = MAX(mat->upperBandwidth, col-i);
                    mat->nzDist[mat->nrows-1+col-i]++;
                } else { // diag
                    mat->nzDist[mat->nrows-1]++;
                }

            }
        }
        mat->bandwidth = mat->lowerBandwidth+mat->upperBandwidth+1;


        DEBUG_LOG(1,"Adjust number of rows and number of nonzeros");
        mat->split(mat);
#else
        ABORT("Trying to create a distributed context without MPI!");
#endif
    }
    DEBUG_LOG(1,"Matrix read in successfully");

    return GHOST_SUCCESS;

}

static ghost_error_t CRS_toBin(ghost_mat_t *mat, char *matrixPath)
{
    ghost_midx_t i;
    ghost_mnnz_t j;
    INFO_LOG("Writing sparse matrix to file %s",matrixPath);

    ghost_midx_t mnrows,mncols,mnnz;
    mnrows = ghost_getMatNrows(mat);
    mncols = mnrows;
    mnnz = ghost_getMatNnz(mat);
    size_t ret;
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

    int32_t endianess = ghost_archIsBigEndian();
    int32_t version = 1;
    int32_t base = 0;
    int32_t symmetry = GHOST_BINCRS_SYMM_GENERAL;
    int32_t datatype = mat->traits->datatype;
    int64_t nrows = (int64_t)mnrows;
    int64_t ncols = (int64_t)mncols;
    int64_t nnz = (int64_t)mnnz;

    FILE *filed;

    if ((filed = fopen64(matrixPath, "w")) == NULL){
        ABORT("Could not vector file %s",matrixPath);
    }

    if ((ret = fwrite(&endianess,sizeof(endianess),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&version,sizeof(version),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&base,sizeof(base),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&symmetry,sizeof(symmetry),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&datatype,sizeof(datatype),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&nrows,sizeof(nrows),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&ncols,sizeof(ncols),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&nnz,sizeof(nnz),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));

    int64_t rpt,col;

    for (i = 0; i < mat->nrows+1; i++) {
        rpt = (int64_t)CR(mat)->rpt[i];
        if ((ret = fwrite(&rpt,sizeof(rpt),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    }


    for (i = 0; i < mat->nrows; i++) {
        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
            col = (int64_t)CR(mat)->col[j];
            if ((ret = fwrite(&col,sizeof(col),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
        }
    }
    if ((ret = fwrite(CR(mat)->val,sizeofdt,nnz,filed)) != nnz) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));

    fclose(filed);

    return GHOST_SUCCESS;

}

static void CRS_free(ghost_mat_t * mat)
{
    if (mat) {
        DEBUG_LOG(1,"Freeing CRS matrix");
        free(CR(mat)->rpt);
        free(CR(mat)->col);
        free(CR(mat)->val);

        free(mat->data);

        if (mat->localPart)
            CRS_free(mat->localPart);

        if (mat->remotePart)
            CRS_free(mat->remotePart);

        free(mat);
        DEBUG_LOG(1,"CRS matrix freed successfully");
    }
}

static ghost_error_t CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, ghost_spmv_flags_t options)
{
    /*    if (mat->symmetry == GHOST_BINCRS_SYMM_SYMMETRIC) {
          ghost_midx_t i, j;
          ghost_dt hlp1;
          ghost_midx_t col;
          ghost_dt val;

#pragma omp    parallel for schedule(runtime) private (hlp1, j, col, val)
for (i=0; i<mat->nrows; i++){
hlp1 = 0.0;

j = CR(mat)->rpt[i];

if (CR(mat)->col[j] == i) {
col = CR(mat)->col[j];
val = CR(mat)->val[j];

hlp1 += val * rhs->val[col];

j++;
} else {
printf("row %d has diagonal 0\n",i);
}


for (; j<CR(mat)->rpt[i+1]; j++){
col = CR(mat)->col[j];
val = CR(mat)->val[j];


hlp1 += val * rhs->val[col];

if (i!=col) {    
#pragma omp atomic
lhs->val[col] += val * rhs->val[i];  // FIXME non-axpy case maybe doesnt work
}

}
if (options & GHOST_SPMVM_AXPY) {
lhs->val[i] += hlp1;
} else {
lhs->val[i] = hlp1;
}
}

} else {

DEBUG_LOG(0,"lhs vector has %s data",ghost_datatypeName(lhs->traits->datatype));

double *rhsv = (double *)rhs->val;    
double *lhsv = (double *)lhs->val;    
ghost_midx_t i, j;
double hlp1;
CR_TYPE *cr = CR(mat);

#pragma omp parallel for schedule(runtime) private (hlp1, j)
for (i=0; i<cr->nrows; i++){
hlp1 = 0.0;
for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
hlp1 = hlp1 + (double)cr->val[j] * rhsv[cr->col[j]];
    //        printf("%d: %d: %f*%f (%d) = %f\n",ghost_getRank(mat->context->mpicomm),i,cr->val[j],rhsv[cr->col[j]],cr->col[j],hlp1);
    }
    if (options & GHOST_SPMVM_AXPY) 
    lhsv[i] += hlp1;
    else
    lhsv[i] = hlp1;
    }
     */
    DEBUG_LOG(2,"lhs vector has %s data and %"PRvecIDX" sub-vectors",ghost_datatypeName(lhs->traits->datatype),lhs->traits->nvecs);
    //    lhs->print(lhs);
    //    rhs->print(rhs);

    /*if (lhs->traits->nvecs == 1) {
      if (lhs->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
      if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
      c_CRS_kernel_plain(mat,lhs,rhs,options);
      else
      s_CRS_kernel_plain(mat,lhs,rhs,options);
      } else {
      if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
      z_CRS_kernel_plain(mat,lhs,rhs,options);
      else
      d_CRS_kernel_plain(mat,lhs,rhs,options);
      }
      } else {
      if (lhs->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
      if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
      c_CRS_kernel_plain_multvec(mat,lhs,rhs,options);
      else
      s_CRS_kernel_plain_multvec(mat,lhs,rhs,options);
      } else {
      if (lhs->traits->datatype & GHOST_BINCRS_DT_COMPLEX)
      z_CRS_kernel_plain_multvec(mat,lhs,rhs,options);
      else
      d_CRS_kernel_plain_multvec(mat,lhs,rhs,options);
      }
      }*/



    //}


    return CRS_kernels_plain[ghost_dataTypeIdx(mat->traits->datatype)][ghost_dataTypeIdx(lhs->traits->datatype)](mat,lhs,rhs,options);
    }

