#define _XOPEN_SOURCE 500

#include "ghost/crs.h"
#include "ghost/util.h"
#include "ghost/core.h"
#include "ghost/sparsemat.h"
#include "ghost/constants.h"
#include "ghost/locality.h"
#include "ghost/context.h"
#include "ghost/machine.h"
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

ghost_error_t (*CRS_kernels_plain[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t options) = 
{{&ss_CRS_kernel_plain,&sd_CRS_kernel_plain,&sc_CRS_kernel_plain,&sz_CRS_kernel_plain},
    {&ds_CRS_kernel_plain,&dd_CRS_kernel_plain,&dc_CRS_kernel_plain,&dz_CRS_kernel_plain},
    {&cs_CRS_kernel_plain,&cd_CRS_kernel_plain,&cc_CRS_kernel_plain,&cz_CRS_kernel_plain},
    {&zs_CRS_kernel_plain,&zd_CRS_kernel_plain,&zc_CRS_kernel_plain,&zz_CRS_kernel_plain}};

const char * (*CRS_stringify_funcs[4]) (ghost_sparsemat_t *, int) = 
{&s_CRS_stringify, &d_CRS_stringify, &c_CRS_stringify, &z_CRS_stringify}; 

static ghost_error_t CRS_fromBin(ghost_sparsemat_t *mat, char *matrixPath);
static ghost_error_t CRS_toBin(ghost_sparsemat_t *mat, char *matrixPath);
static ghost_error_t CRS_fromRowFunc(ghost_sparsemat_t *mat, ghost_midx_t maxrowlen, int base, ghost_sparsemat_fromRowFunc_t func, ghost_sparsemat_fromRowFunc_flags_t flags);
static ghost_error_t CRS_permute(ghost_sparsemat_t *mat, ghost_midx_t *perm, ghost_midx_t *invPerm);
static void CRS_printInfo(char **str, ghost_sparsemat_t *mat);
static const char * CRS_formatName(ghost_sparsemat_t *mat);
static ghost_midx_t CRS_rowLen (ghost_sparsemat_t *mat, ghost_midx_t i);
static size_t CRS_byteSize (ghost_sparsemat_t *mat);
static const char * CRS_stringify(ghost_sparsemat_t *mat, int dense);
static void CRS_free(ghost_sparsemat_t * mat);
static ghost_error_t CRS_kernel_plain (ghost_sparsemat_t *mat, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t);
static ghost_error_t CRS_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crsmat);
#ifdef GHOST_HAVE_MPI
static ghost_error_t CRS_split(ghost_sparsemat_t *mat);
#endif
static ghost_error_t CRS_upload(ghost_sparsemat_t *mat);


ghost_error_t ghost_CRS_init(ghost_sparsemat_t *mat)
{
    ghost_error_t ret = GHOST_SUCCESS;

    DEBUG_LOG(1,"Initializing CRS functions");
    if (!(mat->traits->flags & (GHOST_SPARSEMAT_HOST | GHOST_SPARSEMAT_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting matrix placement");
        mat->traits->flags |= GHOST_SPARSEMAT_HOST;
        ghost_type_t ghost_type;
        GHOST_CALL_GOTO(ghost_getType(&ghost_type),err,ret);
        if (ghost_type == GHOST_TYPE_CUDA) {
            mat->traits->flags |= GHOST_SPARSEMAT_DEVICE;
        }
    }

    if (mat->traits->flags & GHOST_SPARSEMAT_DEVICE)
    {
#ifdef GHOST_HAVE_CUDA
        WARNING_LOG("CUDA CRS SpMV has not yet been implemented!");
        //   mat->spmv = &ghost_cu_crsspmv;
#endif
    }
    else if (mat->traits->flags & GHOST_SPARSEMAT_HOST)
    {
        mat->spmv   = &CRS_kernel_plain;
    }

    mat->fromFile = &CRS_fromBin;
    mat->toFile = &CRS_toBin;
    mat->fromRowFunc = &CRS_fromRowFunc;
    mat->fromCRS = &CRS_fromCRS;
    mat->printInfo = &CRS_printInfo;
    mat->formatName = &CRS_formatName;
    mat->rowLen   = &CRS_rowLen;
    mat->byteSize = &CRS_byteSize;
    mat->permute = &CRS_permute;
    mat->destroy  = &CRS_free;
    mat->stringify = &CRS_stringify;
    mat->upload = &CRS_upload;
#ifdef GHOST_HAVE_MPI
    mat->split = &CRS_split;
#endif
    GHOST_CALL_GOTO(ghost_malloc((void **)&(mat->data),sizeof(ghost_crs_t)),err,ret);

    CR(mat)->rpt = NULL;
    CR(mat)->col = NULL;
    CR(mat)->val = NULL;

    goto out;
err:
    free(mat->data); mat->data = NULL;

out:
    return ret;

}

static ghost_error_t CRS_upload(ghost_sparsemat_t *mat)
{
    UNUSED(mat);
    return GHOST_ERR_NOT_IMPLEMENTED;
}

static ghost_error_t CRS_permute(ghost_sparsemat_t *mat, ghost_midx_t *perm, ghost_midx_t *invPerm)
{
    if (perm == NULL) {
        return GHOST_SUCCESS;
    }
    if (mat->data == NULL) {
        ERROR_LOG("The matrix data to be permuted is NULL");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_error_t ret = GHOST_SUCCESS;
    ghost_midx_t i,j,c;
    ghost_midx_t rowLen;
    ghost_crs_t *cr = CR(mat);


    ghost_mnnz_t *rpt_perm = NULL;
    ghost_mnnz_t *col_perm = NULL;
    char *val_perm = NULL;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt_perm,(mat->nrows+1)*sizeof(ghost_midx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&col_perm,mat->nnz*sizeof(ghost_midx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&val_perm,mat->nnz*mat->traits->elSize),err,ret);

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
        ERROR_LOG("Error in row pointer permutation: %"PRmatIDX" != %"PRmatIDX,rpt_perm[mat->nrows],mat->nnz);
        goto err;
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
        memcpy(&val_perm[rpt_perm[i]*mat->traits->elSize],&cr->val[cr->rpt[invPerm[i]]*mat->traits->elSize],rowLen*mat->traits->elSize);
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
        char tmpval[mat->traits->elSize];
        for (n=rowLen; n>1; n--) {
            for (j=rpt_perm[i]; j<rpt_perm[i]+n-1; j++) {
                if (col_perm[j] > col_perm[j+1]) {
                    tmpcol = col_perm[j];
                    col_perm[j] = col_perm[j+1];
                    col_perm[j+1] = tmpcol;

                    memcpy(&tmpval,&val_perm[mat->traits->elSize*j],mat->traits->elSize);
                    memcpy(&val_perm[mat->traits->elSize*j],&val_perm[mat->traits->elSize*(j+1)],mat->traits->elSize);
                    memcpy(&val_perm[mat->traits->elSize*(j+1)],&tmpval,mat->traits->elSize);
                }
            }
        }
    }
    mat->bandwidth = mat->lowerBandwidth+mat->upperBandwidth+1;

    free(cr->rpt); cr->rpt = NULL;
    free(cr->col); cr->col = NULL;
    free(cr->val); cr->val = NULL;

    cr->rpt = rpt_perm;
    cr->col = col_perm;
    cr->val = val_perm;


    goto out;

err:
    free(rpt_perm); rpt_perm = NULL;
    free(col_perm); col_perm = NULL;
    free(val_perm); val_perm = NULL;

out:

    return ret;

}

static const char * CRS_stringify(ghost_sparsemat_t *mat, int dense)
{
    ghost_datatype_idx_t dtIdx;
    if (ghost_datatypeIdx(&dtIdx,mat->traits->datatype) != GHOST_SUCCESS) {
        return "Invalid";
    }

    return CRS_stringify_funcs[dtIdx](mat, dense);
}

static void CRS_printInfo(char **str, ghost_sparsemat_t *mat)
{
    UNUSED(mat);
    UNUSED(str);
    return;
}

static const char * CRS_formatName(ghost_sparsemat_t *mat)
{
    UNUSED(mat);
    return "CRS";
}

static ghost_midx_t CRS_rowLen (ghost_sparsemat_t *mat, ghost_midx_t i)
{
    if (mat && i<mat->nrows) {
        return CR(mat)->rpt[i+1] - CR(mat)->rpt[i];
    }

    return 0;
}

static size_t CRS_byteSize (ghost_sparsemat_t *mat)
{
    if (mat->data == NULL) {
        return 0;
    }

    return (size_t)((mat->nrows+1)*sizeof(ghost_mnnz_t) + 
            mat->nEnts*(sizeof(ghost_midx_t)+mat->traits->elSize));
}

static ghost_error_t CRS_fromRowFunc(ghost_sparsemat_t *mat, ghost_midx_t maxrowlen, int base, ghost_sparsemat_fromRowFunc_t func, ghost_sparsemat_fromRowFunc_flags_t flags)
{
    ghost_error_t ret = GHOST_SUCCESS;
    UNUSED(base);
    UNUSED(flags);
    
    char * tmpval = NULL;
    ghost_midx_t * tmpcol = NULL;
    
    int nprocs = 1;
    int me;
    GHOST_CALL_GOTO(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs),err,ret);
    GHOST_CALL_GOTO(ghost_getRank(mat->context->mpicomm,&me),err,ret);

    ghost_midx_t rowlen;
    ghost_midx_t i,j;
    mat->ncols = mat->context->gncols;
    mat->nrows = mat->context->lnrows[me];
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->rpt),(mat->nrows+1)*sizeof(ghost_midx_t)),err,ret);
    mat->nEnts = 0;

#pragma omp parallel for schedule(runtime)
    for (i = 0; i < mat->nrows+1; i++) {
        CR(mat)->rpt[i] = 0;
    }

    ghost_mnnz_t nEnts = 0;

#pragma omp parallel private(i,rowlen,tmpval,tmpcol) reduction (+:nEnts)
    { 
        GHOST_CALL(ghost_malloc((void **)&tmpval,maxrowlen*mat->traits->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,maxrowlen*sizeof(ghost_midx_t)),ret);
#pragma omp for ordered
        for( i = 0; i < mat->nrows; i++ ) {
            func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
            nEnts += rowlen;
#pragma omp ordered
            CR(mat)->rpt[i+1] = CR(mat)->rpt[i]+rowlen;
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
    if (ret != GHOST_SUCCESS){
        goto err;
    }
        


    mat->nEnts = nEnts;
    mat->nnz = mat->nEnts;

    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->col),mat->nEnts*sizeof(ghost_midx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->val),mat->nEnts*mat->traits->elSize),err,ret);

#pragma omp parallel for schedule(runtime) private (j)
    for (i = 0; i < mat->nrows; i++) {
        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
            CR(mat)->col[j] = 0;
            memset(&CR(mat)->val[j*mat->traits->elSize],0,mat->traits->elSize);
        }
    }

    // TODO load balancing if distribution by nnz

#pragma omp parallel private(i,rowlen,tmpval,tmpcol)
    { 
        GHOST_CALL(ghost_malloc((void **)&tmpval,maxrowlen*mat->traits->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,maxrowlen*sizeof(ghost_midx_t)),ret);
        memset(tmpval,0,mat->traits->elSize*maxrowlen);
        memset(tmpcol,0,sizeof(ghost_midx_t)*maxrowlen);
#pragma omp for schedule(runtime)
        for( i = 0; i < mat->nrows; i++ ) {
            func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
            memcpy(&CR(mat)->col[CR(mat)->rpt[i]],tmpcol,rowlen*sizeof(ghost_midx_t));
            memcpy(&((char *)CR(mat)->val)[CR(mat)->rpt[i]*mat->traits->elSize],tmpval,rowlen*mat->traits->elSize);
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
    if (ret != GHOST_SUCCESS) {
        goto err;
    }

    if (!(mat->context->flags & GHOST_CONTEXT_REDUNDANT)) {
#ifdef GHOST_HAVE_MPI

        mat->context->lnEnts[me] = mat->nEnts;

        ghost_mnnz_t nents;
        nents = mat->context->lnEnts[me];
        MPI_CALL_GOTO(MPI_Allgather(&nents,1,ghost_mpi_dt_mnnz,mat->context->lnEnts,1,ghost_mpi_dt_mnnz,mat->context->mpicomm),err,ret);

        for (i=0; i<nprocs; i++) {
            mat->context->lfEnt[i] = 0;
        } 

        for (i=1; i<nprocs; i++) {
            mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
        } 

        GHOST_CALL_GOTO(mat->split(mat),err,ret);

#endif
    }
    mat->nrows = mat->context->lnrows[me];

    goto out;
err:
    free(CR(mat)->rpt); CR(mat)->rpt = NULL;
    free(CR(mat)->col); CR(mat)->col = NULL;
    free(CR(mat)->val); CR(mat)->val = NULL;

out:

    return ret;
}

static ghost_error_t CRS_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crsmat)
{
    DEBUG_LOG(1,"Creating CRS matrix from CRS matrix");
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_crs_t *cr = (ghost_crs_t*)(crsmat->data);
    ghost_midx_t i,j;


    //    mat->data = (ghost_crs_t *)ghost_malloc(sizeof(ghost_crs_t));
    mat->nrows = crsmat->nrows;
    mat->ncols = crsmat->ncols;
    mat->nEnts = crsmat->nEnts;

    CR(mat)->rpt = NULL;
    CR(mat)->col = NULL;
    CR(mat)->val = NULL;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->rpt),(crsmat->nrows+1)*sizeof(ghost_midx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->col),crsmat->nEnts*sizeof(ghost_midx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->val),crsmat->nEnts*mat->traits->elSize),err,ret);

#pragma omp parallel for schedule(runtime)
    for( i = 0; i < mat->nrows+1; i++ ) {
        CR(mat)->rpt[i] = cr->rpt[i];
    }

#pragma omp parallel for schedule(runtime) private(j)
    for( i = 0; i < mat->nrows; i++ ) {
        for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
            CR(mat)->col[j] = cr->col[j];
            memcpy(&CR(mat)->val[j*mat->traits->elSize],&cr->val[j*mat->traits->elSize],mat->traits->elSize);
        }
    }

    DEBUG_LOG(1,"Successfully created CRS matrix from CRS data");

    goto out;

err:
    free(CR(mat)->rpt); CR(mat)->rpt = NULL;
    free(CR(mat)->col); CR(mat)->col = NULL;
    free(CR(mat)->val); CR(mat)->val = NULL;

out:
    return ret;
}

#ifdef GHOST_HAVE_MPI

static ghost_error_t CRS_split(ghost_sparsemat_t *mat)
{
    if (!mat) {
        ERROR_LOG("Matrix is NULL");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_error_t ret = GHOST_SUCCESS;
    ghost_crs_t *fullCR = CR(mat);
    ghost_crs_t *localCR = NULL, *remoteCR = NULL;
    DEBUG_LOG(1,"Splitting the CRS matrix into a local and remote part");
    int j;
    int i;
    int me;

    ghost_mnnz_t lnEnts_l, lnEnts_r;
    int current_l, current_r;


    GHOST_CALL_GOTO(ghost_getRank(mat->context->mpicomm,&me),err,ret);
    GHOST_CALL_GOTO(ghost_setupCommunication(mat->context,fullCR->col),err,ret);

    if (mat->traits->flags & GHOST_SPARSEMAT_STORE_SPLIT) { // split computation

        lnEnts_l=0;
        for (i=0; i<mat->context->lnEnts[me];i++) {
            if (fullCR->col[i]<mat->context->lnrows[me]) lnEnts_l++;
        }


        lnEnts_r = mat->context->lnEnts[me]-lnEnts_l;

        DEBUG_LOG(1,"PE%d: Rows=%"PRmatIDX"\t Ents=%"PRmatNNZ"(l),%"PRmatNNZ"(r),%"PRmatNNZ"(g)\t pdim=%"PRmatIDX, 
                me, mat->context->lnrows[me], lnEnts_l, lnEnts_r, mat->context->lnEnts[me],mat->context->lnrows[me]+mat->context->halo_elements  );

        ghost_createMatrix(&(mat->localPart),mat->context,&mat->traits[0],1);
        localCR = mat->localPart->data;
        mat->localPart->traits->symmetry = mat->traits->symmetry;

        ghost_createMatrix(&(mat->remotePart),mat->context,&mat->traits[0],1);
        remoteCR = mat->remotePart->data;

        GHOST_CALL_GOTO(ghost_malloc((void **)&(localCR->val),lnEnts_l*mat->traits->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&(localCR->col),lnEnts_l*sizeof(ghost_midx_t)),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&(localCR->rpt),(mat->context->lnrows[me]+1)*sizeof(ghost_midx_t)),err,ret); 

        GHOST_CALL_GOTO(ghost_malloc((void **)&(remoteCR->val),lnEnts_r*mat->traits->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&(remoteCR->col),lnEnts_r*sizeof(ghost_midx_t)),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&(remoteCR->rpt),(mat->context->lnrows[me]+1)*sizeof(ghost_midx_t)),err,ret); 

        mat->localPart->nrows = mat->context->lnrows[me];
        mat->localPart->nEnts = lnEnts_l;
        mat->localPart->nnz = lnEnts_l;

        mat->remotePart->nrows = mat->context->lnrows[me];
        mat->remotePart->nEnts = lnEnts_r;
        mat->remotePart->nnz = lnEnts_r;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_l; i++) localCR->val[i*mat->traits->elSize] = 0;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_l; i++) localCR->col[i] = 0.0;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_r; i++) remoteCR->val[i*mat->traits->elSize] = 0;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_r; i++) remoteCR->col[i] = 0.0;


        localCR->rpt[0] = 0;
        remoteCR->rpt[0] = 0;

        for (i=0; i<mat->context->lnrows[me]; i++){

            current_l = 0;
            current_r = 0;

            for (j=fullCR->rpt[i]; j<fullCR->rpt[i+1]; j++){

                if (fullCR->col[j]<mat->context->lnrows[me]){
                    localCR->col[ localCR->rpt[i]+current_l ] = fullCR->col[j];
                    memcpy(&localCR->val[(localCR->rpt[i]+current_l)*mat->traits->elSize],&fullCR->val[j*mat->traits->elSize],mat->traits->elSize);
                    current_l++;
                }
                else{
                    remoteCR->col[ remoteCR->rpt[i]+current_r ] = fullCR->col[j];
                    memcpy(&remoteCR->val[(remoteCR->rpt[i]+current_r)*mat->traits->elSize],&fullCR->val[j*mat->traits->elSize],mat->traits->elSize);
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
    }

    goto out;
err:
    mat->localPart->destroy(mat->localPart); mat->localPart = NULL;
    mat->remotePart->destroy(mat->remotePart); mat->remotePart = NULL;

out:
    return ret;

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
static ghost_error_t CRS_fromBin(ghost_sparsemat_t *mat, char *matrixPath)
{
    DEBUG_LOG(1,"Reading CRS matrix from file");
    ghost_error_t ret = GHOST_SUCCESS;
    mat->name = basename(matrixPath);

    ghost_midx_t i;
    ghost_mnnz_t j;

    ghost_matfile_header_t header;

    ghost_readMatFileHeader(matrixPath,&header);

    if (header.version != 1) {
        ERROR_LOG("Can not read version %d of binary CRS format!",header.version);
        return GHOST_ERR_IO;
    }

    if (header.base != 0) {
        ERROR_LOG("Can not read matrix with %d-based indices!",header.base);
        return GHOST_ERR_IO;
    }

    if (!ghost_symmetryValid(header.symmetry)) {
        ERROR_LOG("Symmetry is invalid! (%d)",header.symmetry);
        return GHOST_ERR_IO;
    }

    if (header.symmetry != GHOST_BINCRS_SYMM_GENERAL) {
        ERROR_LOG("Can not handle symmetry different to general at the moment!");
        return GHOST_ERR_IO;
    }

    if (!ghost_datatypeValid(header.datatype)) {
        ERROR_LOG("Datatype is invalid! (%d)",header.datatype);
        return GHOST_ERR_IO;
    }

    mat->traits->symmetry = header.symmetry;
    mat->ncols = (ghost_midx_t)header.ncols;

    DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",mat->nrows,mat->ncols,mat->nEnts);

    if (mat->context->flags & GHOST_CONTEXT_REDUNDANT) {
        mat->nrows = (ghost_midx_t)header.nrows;
        mat->nEnts = (ghost_midx_t)header.nnz;
        mat->nnz = mat->nEnts;
        
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&(CR(mat)->rpt),(mat->nrows+1) * sizeof(ghost_mnnz_t), GHOST_DATA_ALIGNMENT),err,ret);
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&(CR(mat)->col),mat->nEnts * sizeof(ghost_midx_t), GHOST_DATA_ALIGNMENT),err,ret);
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&(CR(mat)->val),mat->nEnts * mat->traits->elSize,GHOST_DATA_ALIGNMENT),err,ret);

#pragma omp parallel for schedule(runtime)
        for (i = 0; i < mat->nrows+1; i++) {
            CR(mat)->rpt[i] = 0;
        }

        GHOST_CALL_GOTO(ghost_readRpt(CR(mat)->rpt, matrixPath, 0, mat->nrows+1),err,ret);

#pragma omp parallel for schedule(runtime) private (j)
        for (i = 0; i < mat->nrows; i++) {
            for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                CR(mat)->col[j] = 0;
                memset(&CR(mat)->val[j*mat->traits->elSize],0,mat->traits->elSize);
            }
        }

        GHOST_CALL_GOTO(ghost_readCol(CR(mat)->col, matrixPath, 0, mat->nEnts),err,ret);
        GHOST_CALL_GOTO(ghost_readVal(CR(mat)->val, mat->traits->datatype, matrixPath, 0, mat->nEnts),err,ret);


    } else {
#ifdef GHOST_HAVE_MPI
        DEBUG_LOG(1,"Reading in a distributed context");
        DEBUG_LOG(1,"Creating distributed context with parallel MPI-IO");

        ghost_context_t *context = mat->context;
        int nprocs = 1;
        int me;
        GHOST_CALL_GOTO(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs),err,ret);
        GHOST_CALL_GOTO(ghost_getRank(mat->context->mpicomm,&me),err,ret);

        if (me == 0) {
            if (context->flags & GHOST_CONTEXT_DIST_NZ) { // rpt has already been read
                ((ghost_crs_t *)(mat->data))->rpt = context->rpt;
            } else {
                GHOST_CALL_GOTO(ghost_malloc_align((void **)&(CR(mat)->rpt),(header.nrows+1) * sizeof(ghost_mnnz_t), GHOST_DATA_ALIGNMENT),err,ret);
#pragma omp parallel for schedule(runtime) 
                for (i = 0; i < header.nrows+1; i++) {
                    CR(mat)->rpt[i] = 0;
                }
                GHOST_CALL_GOTO(ghost_readRpt(CR(mat)->rpt, matrixPath, 0, header.nrows+1),err,ret);
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
        MPI_CALL_GOTO(MPI_Bcast(context->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Bcast(context->lnEnts, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm),err,ret);

        mat->nnz = context->lnEnts[me];
        mat->nEnts = mat->nnz;
        mat->nrows = context->lnrows[me];

        DEBUG_LOG(1,"Mallocing space for %"PRmatIDX" rows",context->lnrows[me]);

        if (me != 0) {
            GHOST_CALL_GOTO(ghost_malloc_align((void **)&(CR(mat)->rpt),(context->lnrows[me]+1)*sizeof(ghost_midx_t),GHOST_DATA_ALIGNMENT),err,ret);
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
            MPI_CALL_GOTO(MPI_Irecv(CR(mat)->rpt,context->lnrows[me]+1,ghost_mpi_dt_midx,0,me,context->mpicomm,&req[msgcount]),err,ret);
            msgcount++;
        } else {
            for (i=1;i<nprocs;i++) {
                MPI_CALL_GOTO(MPI_Isend(&CR(mat)->rpt[context->lfRow[i]],context->lnrows[i]+1,ghost_mpi_dt_midx,i,i,context->mpicomm,&req[msgcount]),err,ret);
                msgcount++;
            }
        }
        MPI_CALL_GOTO(MPI_Waitall(msgcount,req,stat),err,ret);

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

        GHOST_CALL_GOTO(ghost_malloc_align((void **)&(CR(mat)->col),mat->nEnts * sizeof(ghost_midx_t), GHOST_DATA_ALIGNMENT),err,ret);
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&(CR(mat)->val),mat->nEnts * mat->traits->elSize,GHOST_DATA_ALIGNMENT),err,ret);

#pragma omp parallel for schedule(runtime) private (j)
        for (i = 0; i < mat->nrows; i++) {
            for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                CR(mat)->col[j] = 0;
                memset(&CR(mat)->val[j*mat->traits->elSize],0,mat->traits->elSize);
            }
        }

        GHOST_CALL_GOTO(ghost_readCol(CR(mat)->col, matrixPath, context->lfEnt[me], mat->nEnts),err,ret);
        GHOST_CALL_GOTO(ghost_readVal(CR(mat)->val, mat->traits->datatype, matrixPath, context->lfEnt[me], mat->nEnts),err,ret);

        GHOST_CALL_GOTO(ghost_malloc((void **)&(mat->nzDist),sizeof(ghost_mnnz_t)*(2*mat->nrows-1)),err,ret);
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


        DEBUG_LOG(1,"Split matrix");
        GHOST_CALL_GOTO(mat->split(mat),err,ret);
#else
        ERROR_LOG("Trying to create a distributed context without MPI!");
        return GHOST_ERR_INVALID_ARG;
#endif
    }
    DEBUG_LOG(1,"Matrix read in successfully");

    goto out;
err:
    free(CR(mat)->rpt); CR(mat)->rpt = NULL;
    free(CR(mat)->col); CR(mat)->col = NULL;
    free(CR(mat)->val); CR(mat)->val = NULL;
    free(mat->nzDist); mat->nzDist = NULL;

out:

    return ret;

}

static ghost_error_t CRS_toBin(ghost_sparsemat_t *mat, char *matrixPath)
{
    ghost_midx_t i;
    ghost_mnnz_t j;
    INFO_LOG("Writing sparse matrix to file %s",matrixPath);

    ghost_midx_t mnrows,mncols,mnnz;
    GHOST_CALL_RETURN(ghost_getMatNrows(&mnrows,mat));
    mncols = mnrows;
    GHOST_CALL_RETURN(ghost_getMatNnz(&mnnz,mat));
    size_t ret;

    int32_t endianess = ghost_machineIsBigEndian();
    int32_t version = 1;
    int32_t base = 0;
    int32_t symmetry = GHOST_BINCRS_SYMM_GENERAL;
    int32_t datatype = mat->traits->datatype;
    int64_t nrows = (int64_t)mnrows;
    int64_t ncols = (int64_t)mncols;
    int64_t nnz = (int64_t)mnnz;

    FILE *filed;

    if ((filed = fopen64(matrixPath, "w")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s: %s",matrixPath,strerror(errno));
        return GHOST_ERR_IO;
    }

    if ((ret = fwrite(&endianess,sizeof(endianess),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&version,sizeof(version),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&base,sizeof(base),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&symmetry,sizeof(symmetry),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&datatype,sizeof(datatype),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&nrows,sizeof(nrows),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&ncols,sizeof(ncols),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&nnz,sizeof(nnz),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }

    int64_t rpt,col;

    for (i = 0; i < mat->nrows+1; i++) {
        rpt = (int64_t)CR(mat)->rpt[i];
        if ((ret = fwrite(&rpt,sizeof(rpt),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
        }
    }


    for (i = 0; i < mat->nrows; i++) {
        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
            col = (int64_t)CR(mat)->col[j];
            if ((ret = fwrite(&col,sizeof(col),1,filed)) != 1) {
                ERROR_LOG("fwrite failed: %zu",ret);
                fclose(filed);
                return GHOST_ERR_IO;
            }
        }
    }
    if ((ret = fwrite(CR(mat)->val,mat->traits->elSize,nnz,filed)) != nnz) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }

    fclose(filed);

    return GHOST_SUCCESS;

}

static void CRS_free(ghost_sparsemat_t * mat)
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

static ghost_error_t CRS_kernel_plain (ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options)
{
    ghost_datatype_idx_t matDtIdx;
    ghost_datatype_idx_t vecDtIdx;
    GHOST_CALL_RETURN(ghost_datatypeIdx(&matDtIdx,mat->traits->datatype));
    GHOST_CALL_RETURN(ghost_datatypeIdx(&vecDtIdx,lhs->traits->datatype));

    return CRS_kernels_plain[matDtIdx][vecDtIdx](mat,lhs,rhs,options);
}

