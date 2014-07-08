#define _XOPEN_SOURCE 500

#include "ghost/crs.h"
#include "ghost/util.h"
#include "ghost/core.h"
#include "ghost/sparsemat.h"
#include "ghost/locality.h"
#include "ghost/context.h"
#include "ghost/machine.h"
#include "ghost/bincrs.h"
#include "ghost/log.h"
#include "ghost/instr.h"

#include <unistd.h>
#include <sys/types.h>
#include <limits.h>
#include <errno.h>
#include <math.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <byteswap.h>

#include <dlfcn.h>

static ghost_error_t (*CRS_kernels_plain[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t options, va_list argp) = 
{{&ss_CRS_kernel_plain,&sd_CRS_kernel_plain,&sc_CRS_kernel_plain,&sz_CRS_kernel_plain},
    {&ds_CRS_kernel_plain,&dd_CRS_kernel_plain,&dc_CRS_kernel_plain,&dz_CRS_kernel_plain},
    {&cs_CRS_kernel_plain,&cd_CRS_kernel_plain,&cc_CRS_kernel_plain,&cz_CRS_kernel_plain},
    {&zs_CRS_kernel_plain,&zd_CRS_kernel_plain,&zc_CRS_kernel_plain,&zz_CRS_kernel_plain}};

static ghost_error_t (*CRS_stringify_funcs[4]) (ghost_sparsemat_t *, char **, int) = 
{&s_CRS_stringify, &d_CRS_stringify, &c_CRS_stringify, &z_CRS_stringify}; 

static ghost_error_t CRS_fromBin(ghost_sparsemat_t *mat, char *matrixPath);
static ghost_error_t CRS_toBin(ghost_sparsemat_t *mat, char *matrixPath);
static ghost_error_t CRS_fromRowFunc(ghost_sparsemat_t *mat, ghost_sparsemat_src_rowfunc_t *src);
static ghost_error_t CRS_permute(ghost_sparsemat_t *mat, ghost_lidx_t *perm, ghost_lidx_t *invPerm);
static void CRS_printInfo(ghost_sparsemat_t *mat, char **str);
static const char * CRS_formatName(ghost_sparsemat_t *mat);
static ghost_lidx_t CRS_rowLen (ghost_sparsemat_t *mat, ghost_lidx_t i);
static size_t CRS_byteSize (ghost_sparsemat_t *mat);
static ghost_error_t CRS_stringify(ghost_sparsemat_t *mat, char **str, int dense);
static void CRS_free(ghost_sparsemat_t * mat);
static ghost_error_t CRS_kernel_plain (ghost_sparsemat_t *mat, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list);
static ghost_error_t CRS_fromCRS(ghost_sparsemat_t *mat, ghost_sparsemat_t *crsmat);
#ifdef GHOST_HAVE_MPI
static ghost_error_t CRS_split(ghost_sparsemat_t *mat);
#endif
static ghost_error_t CRS_upload(ghost_sparsemat_t *mat);


ghost_error_t ghost_crs_init(ghost_sparsemat_t *mat)
{
    ghost_error_t ret = GHOST_SUCCESS;

    DEBUG_LOG(1,"Initializing CRS functions");
    if (!(mat->traits->flags & (GHOST_SPARSEMAT_HOST | GHOST_SPARSEMAT_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting matrix placement");
        mat->traits->flags |= GHOST_SPARSEMAT_HOST;
        ghost_type_t ghost_type;
        GHOST_CALL_GOTO(ghost_type_get(&ghost_type),err,ret);
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
    mat->auxString = &CRS_printInfo;
    mat->formatName = &CRS_formatName;
    mat->rowLen   = &CRS_rowLen;
    mat->byteSize = &CRS_byteSize;
    mat->permute = &CRS_permute;
    mat->destroy  = &CRS_free;
    mat->string = &CRS_stringify;
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

static ghost_error_t CRS_permute(ghost_sparsemat_t *mat, ghost_lidx_t *perm, ghost_lidx_t *invPerm)
{
#if 0
    if (perm == NULL) {
        return GHOST_SUCCESS;
    }
    if (mat->data == NULL) {
        ERROR_LOG("The matrix data to be permuted is NULL");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_error_t ret = GHOST_SUCCESS;
    ghost_lidx_t i,j,c;
    ghost_lidx_t rowLen;
    ghost_crs_t *cr = CR(mat);
    int me;


    ghost_lidx_t *rpt_perm = NULL;
    ghost_gidx_t *col_perm = NULL;
    char *val_perm = NULL;

    GHOST_CALL_GOTO(ghost_rank(&me, MPI_COMM_WORLD),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt_perm,(mat->nrows+1)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&col_perm,mat->nnz*sizeof(ghost_gidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&val_perm,mat->nnz*mat->elSize),err,ret);

    /*for (i=0; i<mat->nrows; i++) {
      printf("perm/inv[%"PRLIDX"] = %"PRLIDX" %"PRLIDX"\n",i,perm[i],invPerm[i]);
      }*/


    rpt_perm[0] = 0;
    for (i=1; i<mat->nrows+1; i++) {
        rowLen = cr->rpt[invPerm[i-1]+1]-cr->rpt[invPerm[i-1]];
        rpt_perm[i] = rpt_perm[i-1]+rowLen;
        //printf("rpt_perm[%"PRLIDX"] = %"PRLIDX", rowLen: %"PRLIDX"\n",i,rpt_perm[i],rowLen);
    }
    if (rpt_perm[mat->nrows] != mat->nnz) {
        ERROR_LOG("Error in row pointer permutation: %"PRLIDX" != %"PRLIDX,rpt_perm[mat->nrows],mat->nnz);
        goto err;
    }

    for (i = 0; i < mat->nrows; i++) {
        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
        }
    }
    ghost_lidx_t col;
    ghost_gidx_t globrow;
    for (i=0; i<mat->nrows; i++) {
        globrow = mat->context->lfRow[me]+i;
        rowLen = rpt_perm[i+1]-rpt_perm[i];
        memcpy(&val_perm[rpt_perm[i]*mat->elSize],&cr->val[cr->rpt[invPerm[i]]*mat->elSize],rowLen*mat->elSize);
        //memcpy(&col_perm[rpt_perm[i]],&cr->col[cr->rpt[invPerm[i]]],rowLen*sizeof(ghost_lidx_t));
        for (j=rpt_perm[i], c=0; j<rpt_perm[i+1]; j++, c++) {
            if (
                    cr->col[cr->rpt[invPerm[i]]+c] >= mat->context->lfRow[me] &&
                    cr->col[cr->rpt[invPerm[i]]+c] < mat->context->lfRow[me]+mat->context->lnrows[me]) {
                col = perm[cr->col[cr->rpt[invPerm[i]]+c]-mat->context->lfRow[me]]+mat->context->lfRow[me];
            } else {
                col = cr->col[cr->rpt[invPerm[i]]+c];
            }
            col_perm[j] = col;
            if (col < globrow) {
                mat->lowerBandwidth = MAX(mat->lowerBandwidth, globrow-col);
#ifdef GHOST_GATHER_GLOBAL_INFO
                mat->nzDist[mat->context->gnrows-1-(globrow-col)]++;
#endif
            } else if (col > globrow) {
                mat->upperBandwidth = MAX(mat->upperBandwidth, col-globrow);
#ifdef GHOST_GATHER_GLOBAL_INFO
                mat->nzDist[mat->context->gnrows-1+col-globrow]++;
#endif
            } else {
#ifdef GHOST_GATHER_GLOBAL_INFO
                mat->nzDist[mat->context->gnrows-1]++;
#endif
            }
        } 
        ghost_lidx_t n;
        ghost_lidx_t tmpcol;
        char tmpval[mat->elSize];
        for (n=rowLen; n>1; n--) {
            for (j=rpt_perm[i]; j<rpt_perm[i]+n-1; j++) {
                if (col_perm[j] > col_perm[j+1]) {
                    tmpcol = col_perm[j];
                    col_perm[j] = col_perm[j+1];
                    col_perm[j+1] = tmpcol;

                    memcpy(&tmpval,&val_perm[mat->elSize*j],mat->elSize);
                    memcpy(&val_perm[mat->elSize*j],&val_perm[mat->elSize*(j+1)],mat->elSize);
                    memcpy(&val_perm[mat->elSize*(j+1)],&tmpval,mat->elSize);
                }
            }
        }
    }
#ifdef GHOST_HAVE_MPI
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,&mat->lowerBandwidth,1,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,&mat->upperBandwidth,1,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);
#ifdef GHOST_GATHER_GLOBAL_INFO
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,mat->nzDist,2*mat->context->gnrows-1,ghost_mpi_dt_idx,MPI_SUM,mat->context->mpicomm),err,ret);
#endif
#endif
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
#endif
    UNUSED(mat);
    UNUSED(perm);
    UNUSED(invPerm);
    ERROR_LOG("Not implemented");
    return GHOST_ERR_NOT_IMPLEMENTED;

}

static ghost_error_t CRS_stringify(ghost_sparsemat_t *mat, char **str, int dense)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,mat->traits->datatype));

    return CRS_stringify_funcs[dtIdx](mat, str, dense);
}

static void CRS_printInfo(ghost_sparsemat_t *mat, char **str)
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

static ghost_lidx_t CRS_rowLen (ghost_sparsemat_t *mat, ghost_lidx_t i)
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

    return (size_t)((mat->nrows+1)*sizeof(ghost_lidx_t) + 
            mat->nEnts*(sizeof(ghost_lidx_t)+mat->elSize));
}

static ghost_error_t CRS_fromRowFunc(ghost_sparsemat_t *mat, ghost_sparsemat_src_rowfunc_t *src)
{
    ghost_error_t ret = GHOST_SUCCESS;

    char * tmpval = NULL;
    ghost_gidx_t * tmpcol = NULL;

    int nprocs = 1;
    int me;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);

    GHOST_CALL_GOTO(ghost_sparsemat_fromfunc_common(mat,src),err,ret);
    ghost_lidx_t rowlen;
    ghost_gidx_t i,j;
    mat->ncols = mat->context->gncols;
    mat->nrows = mat->context->lnrows[me];
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->rpt),(mat->nrows+1)*sizeof(ghost_lidx_t)),err,ret);
    mat->nEnts = 0;

#pragma omp parallel for schedule(runtime)
    for (i = 0; i < mat->nrows+1; i++) {
        CR(mat)->rpt[i] = 0;
    }

    ghost_gidx_t gnents = 0;
    int funcret = 0;

    GHOST_INSTR_START(crs_fromrowfunc_extractrpt)
#pragma omp parallel private(i,rowlen,tmpval,tmpcol) reduction (+:gnents,funcret) 
    {
        GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t)),ret);
#pragma omp for ordered
        for(i = 0; i < mat->nrows; i++) {
            if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && mat->permutation) {
                if (mat->permutation->scope == GHOST_PERMUTATION_GLOBAL) {
                    funcret += src->func(mat->permutation->invPerm[mat->context->lfRow[me]+i],&rowlen,tmpcol,tmpval);
                } else {
                    funcret += src->func(mat->context->lfRow[me]+mat->permutation->invPerm[i],&rowlen,tmpcol,tmpval);
                }
            } else {
                funcret += src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
            }
            gnents += rowlen;
#pragma omp ordered
            {
                if ((ghost_gidx_t)CR(mat)->rpt[i]+rowlen > (ghost_gidx_t)GHOST_LIDX_MAX) {
                    ERROR_LOG("rpt too large!");
                    ret = GHOST_ERR_UNKNOWN;
                }
                CR(mat)->rpt[i+1] = CR(mat)->rpt[i]+rowlen;
            }
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
    if (funcret) {
        ERROR_LOG("Matrix construction function returned error");
        ret = GHOST_ERR_UNKNOWN;
    }
    if (ret != GHOST_SUCCESS){
        goto err;
    }
    
    if (gnents > (ghost_gidx_t)GHOST_LIDX_MAX) {
        ERROR_LOG("The local number of entries is too large.");
        return GHOST_ERR_UNKNOWN;
    }

    GHOST_INSTR_STOP(crs_fromrowfunc_extractrpt)

    mat->nnz = CR(mat)->rpt[mat->nrows];
    mat->nEnts = mat->nnz;

    GHOST_CALL_GOTO(ghost_malloc((void **)&(mat->col_orig),mat->nEnts*sizeof(ghost_gidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->val),mat->nEnts*mat->elSize),err,ret);

#pragma omp parallel for schedule(runtime) private (j)
    for (i = 0; i < mat->nrows; i++) {
        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
            mat->col_orig[j] = 0;
            memset(&CR(mat)->val[j*mat->elSize],0,mat->elSize);
        }
    }

    int funcerrs = 0;

    GHOST_INSTR_START(crs_fromrowfunc_readcolval)
#pragma omp parallel private(i,j,rowlen,tmpcol) reduction(+:funcerrs)
    {
        int funcret = 0;
        GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t)),ret);
        memset(tmpcol,0,sizeof(ghost_gidx_t)*src->maxrowlen);
    
#pragma omp for schedule(runtime)
        for( i = 0; i < mat->nrows; i++ ) {
            if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && mat->permutation) {
                if (mat->permutation->scope == GHOST_PERMUTATION_GLOBAL) {
                    if (mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                        funcret = src->func(mat->context->lfRow[me]+mat->permutation->invPerm[i],&rowlen,&mat->col_orig[CR(mat)->rpt[i]],&CR(mat)->val[CR(mat)->rpt[i]*mat->elSize]);
                    } else {
                        funcret = src->func(mat->permutation->invPerm[mat->context->lfRow[me]+i],&rowlen,tmpcol,&CR(mat)->val[CR(mat)->rpt[i]*mat->elSize]);
                        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                            mat->col_orig[j] = mat->permutation->perm[tmpcol[j-CR(mat)->rpt[i]]];
                        }
                    }
                } else {
                    if (mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                        funcret = src->func(mat->context->lfRow[me]+mat->permutation->invPerm[i],&rowlen,&mat->col_orig[CR(mat)->rpt[i]],&CR(mat)->val[CR(mat)->rpt[i]*mat->elSize]);
                    } else {
                        funcret = src->func(mat->context->lfRow[me]+mat->permutation->invPerm[i],&rowlen,tmpcol,&CR(mat)->val[CR(mat)->rpt[i]*mat->elSize]);
                        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                            if ((tmpcol[j-CR(mat)->rpt[i]] >= mat->context->lfRow[me]) && (tmpcol[j-CR(mat)->rpt[i]] < mat->context->lfRow[me]+mat->context->lnrows[me])) {
                                mat->col_orig[j] = mat->permutation->perm[tmpcol[j-CR(mat)->rpt[i]]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                            //INFO_LOG("if %d>=%d && %d<%d: %f|%d (%d->%d) goes to local part permuted",tmpcol[j-CR(mat)->rpt[i]], mat->context->lfRow[me], tmpcol[j-CR(mat)->rpt[i]], mat->context->lfRow[me]+mat->context->lnrows[me],((double *)(CR(mat)->val))[j],CR(mat)->col[j],tmpcol[j-CR(mat)->rpt[i]],mat->permutation->perm[tmpcol[j-CR(mat)->rpt[i]]-mat->context->lfRow[me]]+mat->context->lfRow[me]);
                            } else {
                                mat->col_orig[j] = tmpcol[j-CR(mat)->rpt[i]];
                            }
                        }
                    }
                }
            } else {
                funcret = src->func(mat->context->lfRow[me]+i,&rowlen,&mat->col_orig[CR(mat)->rpt[i]],&CR(mat)->val[CR(mat)->rpt[i]*mat->elSize]);
                 /*for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                    if (CR(mat)->col[j] >= mat->context->gncols) {
                        ERROR_LOG("col idx too large! %"PRLIDX" >= %"PRLIDX" (row %"PRLIDX" el %"PRLIDX"/%"PRLIDX" (%"PRLIDX"))",CR(mat)->col[j],mat->context->gncols,i,j-CR(mat)->rpt[i],CR(mat)->rpt[i+1]-CR(mat)->rpt[i],rowlen);
                        ret = GHOST_ERR_UNKNOWN;
                        break;
                    }
                     //INFO_LOG("%d/%d: %f|%d",i,j,((double *)(CR(mat)->val))[j],CR(mat)->col[j]);
                 }*/
            }
            if (funcret) {
                funcerrs++;
            }
            if (!(mat->traits->flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
                // sort rows by ascending column indices
                GHOST_CALL(ghost_sparsemat_sortrow(&mat->col_orig[CR(mat)->rpt[i]],&CR(mat)->val[CR(mat)->rpt[i]*mat->elSize],mat->elSize,CR(mat)->rpt[i+1]-CR(mat)->rpt[i],1),ret);
            }
#pragma omp critical
            GHOST_CALL(ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+i,&mat->col_orig[CR(mat)->rpt[i]],CR(mat)->rpt[i+1]-CR(mat)->rpt[i],1),ret);
        }
        free(tmpcol);
    }
    if (funcerrs) {
        ERROR_LOG("Matrix construction function returned error");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    ghost_sparsemat_registerrow_finalize(mat);
    GHOST_INSTR_STOP(crs_fromrowfunc_readcolval)


    if (!(mat->context->flags & GHOST_CONTEXT_REDUNDANT)) {
#ifdef GHOST_HAVE_MPI

        mat->context->lnEnts[me] = mat->nEnts;

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
    ghost_gidx_t i,j;


    //    mat->data = (ghost_crs_t *)ghost_malloc(sizeof(ghost_crs_t));
    mat->nrows = crsmat->nrows;
    mat->ncols = crsmat->ncols;
    mat->nEnts = crsmat->nEnts;

    CR(mat)->rpt = NULL;
    CR(mat)->col = NULL;
    CR(mat)->val = NULL;

    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->rpt),(crsmat->nrows+1)*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->col),crsmat->nEnts*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&(CR(mat)->val),crsmat->nEnts*mat->elSize),err,ret);

#pragma omp parallel for schedule(runtime)
    for( i = 0; i < mat->nrows+1; i++ ) {
        CR(mat)->rpt[i] = cr->rpt[i];
    }

#pragma omp parallel for schedule(runtime) private(j)
    for( i = 0; i < mat->nrows; i++ ) {
        for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
            CR(mat)->col[j] = cr->col[j];
            memcpy(&CR(mat)->val[j*mat->elSize],&cr->val[j*mat->elSize],mat->elSize);
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
    ghost_gidx_t j,i;
    int me;

    ghost_lidx_t lnEnts_l, lnEnts_r;
    ghost_lidx_t current_l, current_r;


    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    if (mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS) {
//        memcpy(mat->col_orig, fullCR->col, mat->nnz*sizeof(ghost_gidx_t));
    }
    
    WARNING_LOG("This does not have to be present twice in all cases!");
    GHOST_CALL_GOTO(ghost_malloc((void **)&CR(mat)->col,sizeof(ghost_gidx_t)*mat->nnz),err,ret);
    
#ifdef GHOST_HAVE_UNIFORM_IDX
    if (!(mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
        INFO_LOG("In-place column compression!");
        CR(mat)->col = mat->col_orig;
    } else 
#endif
    {
        INFO_LOG("Duplicate col array!");
        GHOST_CALL_GOTO(ghost_malloc((void **)&CR(mat)->col,sizeof(ghost_lidx_t)*mat->nnz),err,ret);
    }
   
    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED) {
        GHOST_CALL_GOTO(ghost_context_comm_init(mat->context,mat->col_orig,fullCR->col),err,ret);
    } else {
        for (i=0; i<mat->nEnts; i++) {
            CR(mat)->col[i] = (ghost_lidx_t)mat->col_orig[i];
        }
    }
#ifndef GHOST_HAVE_UNIFORM_IDX
    if (!(mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
        INFO_LOG("Free orig cols");
        free(mat->col_orig);
        mat->col_orig = NULL;
    }
#endif


    if (!(mat->traits->flags & GHOST_SPARSEMAT_NOT_STORE_SPLIT)) { // split computation

        lnEnts_l=0;
        for (i=0; i<mat->context->lnEnts[me];i++) {
            if (fullCR->col[i]<mat->context->lnrows[me]) lnEnts_l++;
        }


        lnEnts_r = mat->context->lnEnts[me]-lnEnts_l;

        DEBUG_LOG(1,"PE%d: Rows=%"PRLIDX"\t Ents=%"PRLIDX"(l),%"PRLIDX"(r),%"PRLIDX"(g)\t pdim=%"PRLIDX, 
                me, mat->context->lnrows[me], lnEnts_l, lnEnts_r, mat->context->lnEnts[me],mat->context->lnrows[me]+mat->context->halo_elements  );

        ghost_sparsemat_create(&(mat->localPart),mat->context,&mat->traits[0],1);
        localCR = mat->localPart->data;
        mat->localPart->traits->symmetry = mat->traits->symmetry;

        ghost_sparsemat_create(&(mat->remotePart),mat->context,&mat->traits[0],1);
        remoteCR = mat->remotePart->data;

        GHOST_CALL_GOTO(ghost_malloc((void **)&(localCR->val),lnEnts_l*mat->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&(localCR->col),lnEnts_l*sizeof(ghost_lidx_t)),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&(localCR->rpt),(mat->context->lnrows[me]+1)*sizeof(ghost_lidx_t)),err,ret); 

        GHOST_CALL_GOTO(ghost_malloc((void **)&(remoteCR->val),lnEnts_r*mat->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&(remoteCR->col),lnEnts_r*sizeof(ghost_lidx_t)),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&(remoteCR->rpt),(mat->context->lnrows[me]+1)*sizeof(ghost_lidx_t)),err,ret); 

        mat->localPart->nrows = mat->context->lnrows[me];
        mat->localPart->nEnts = lnEnts_l;
        mat->localPart->nnz = lnEnts_l;

        mat->remotePart->nrows = mat->context->lnrows[me];
        mat->remotePart->nEnts = lnEnts_r;
        mat->remotePart->nnz = lnEnts_r;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_l; i++) localCR->val[i*mat->elSize] = 0;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_l; i++) localCR->col[i] = 0.0;

#pragma omp parallel for schedule(runtime)
        for (i=0; i<lnEnts_r; i++) remoteCR->val[i*mat->elSize] = 0;

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
                    memcpy(&localCR->val[(localCR->rpt[i]+current_l)*mat->elSize],&fullCR->val[j*mat->elSize],mat->elSize);
                    current_l++;
                }
                else{
                    remoteCR->col[ remoteCR->rpt[i]+current_r ] = fullCR->col[j];
                    memcpy(&remoteCR->val[(remoteCR->rpt[i]+current_r)*mat->elSize],&fullCR->val[j*mat->elSize],mat->elSize);
                    current_r++;
                }

            }  

            localCR->rpt[i+1] = localCR->rpt[i] + current_l;
            remoteCR->rpt[i+1] = remoteCR->rpt[i] + current_r;
        }

        IF_DEBUG(3){
            for (i=0; i<mat->context->lnrows[me]+1; i++)
                DEBUG_LOG(3,"--Row_ptrs-- PE %d: i=%"PRGIDX" local=%"PRLIDX" remote=%"PRLIDX, 
                        me, i, localCR->rpt[i], remoteCR->rpt[i]);
            for (i=0; i<localCR->rpt[mat->context->lnrows[me]]; i++)
                DEBUG_LOG(3,"-- local -- PE%d: localCR->col[%"PRGIDX"]=%"PRLIDX, me, i, localCR->col[i]);
            for (i=0; i<remoteCR->rpt[mat->context->lnrows[me]]; i++)
                DEBUG_LOG(3,"-- remote -- PE%d: remoteCR->col[%"PRGIDX"]=%"PRLIDX, me, i, remoteCR->col[i]);
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

    ghost_gidx_t i, j;
    int me;
    
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_sparsemat_fromfile_common(mat,matrixPath,&(CR(mat)->rpt)),err,ret);
    mat->nEnts = mat->nnz;

    GHOST_CALL_GOTO(ghost_malloc_align((void **)&(mat->col_orig),mat->nEnts * sizeof(ghost_gidx_t), GHOST_DATA_ALIGNMENT),err,ret);
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&(CR(mat)->val),mat->nEnts * mat->elSize,GHOST_DATA_ALIGNMENT),err,ret);

#pragma omp parallel for schedule(runtime) private (j)
    for (i = 0; i < mat->nrows; i++) {
        for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
            mat->col_orig = 0;
            memset(&CR(mat)->val[j*mat->elSize],0,mat->elSize);
        }
    }

    GHOST_CALL_GOTO(ghost_bincrs_col_read(mat->col_orig, matrixPath, mat->context->lfRow[me], mat->nrows, mat->permutation,mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS),err,ret);
    GHOST_CALL_GOTO(ghost_bincrs_val_read(CR(mat)->val, mat->traits->datatype, matrixPath, mat->context->lfRow[me], mat->nrows, mat->permutation),err,ret);

    for (i=0;i<mat->context->lnrows[me];i++) {
        if (!(mat->traits->flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
            // sort rows by ascending column indices
            GHOST_CALL_GOTO(ghost_sparsemat_sortrow(&mat->col_orig[CR(mat)->rpt[i]],&CR(mat)->val[CR(mat)->rpt[i]*mat->elSize],mat->elSize,CR(mat)->rpt[i+1]-CR(mat)->rpt[i],1),err,ret);
        }
        GHOST_CALL_GOTO(ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+i,&mat->col_orig[CR(mat)->rpt[i]],CR(mat)->rpt[i+1]-CR(mat)->rpt[i],1),err,ret);
    }
    ghost_sparsemat_registerrow_finalize(mat);

#ifdef GHOST_HAVE_MPI
    DEBUG_LOG(1,"Split matrix");
    GHOST_CALL_GOTO(mat->split(mat),err,ret);
    DEBUG_LOG(1,"Matrix read in successfully");
#endif

    goto out;
err:
    free(CR(mat)->rpt); CR(mat)->rpt = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
    free(CR(mat)->val); CR(mat)->val = NULL;
    free(mat->nzDist); mat->nzDist = NULL;

out:

    return ret;

}

static ghost_error_t CRS_toBin(ghost_sparsemat_t *mat, char *matrixPath)
{
    DEBUG_LOG(1,"Writing CRS matrix to file %s",matrixPath);
    GHOST_CALL_RETURN(ghost_sparsemat_tofile_header(mat,matrixPath));
   
#ifdef GHOST_HAVE_MPI
    int me, nrank;
    ghost_gidx_t i;
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nrank, mat->context->mpicomm));
    MPI_File fileh;
    MPI_Status status;
    MPI_CALL_RETURN(MPI_File_open(mat->context->mpicomm,matrixPath,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&fileh));
    
    MPI_CALL_RETURN(MPI_File_set_view(fileh, GHOST_BINCRS_SIZE_HEADER + mat->context->lfRow[me]*GHOST_BINCRS_SIZE_RPT_EL,MPI_LONG_LONG,MPI_LONG_LONG,"native",MPI_INFO_NULL)); 

#ifdef GHOST_HAVE_LONGIDX_GLOBAL

    ghost_gidx_t *rpt_w_offs = NULL;
    size_t to_write = mat->nrows;
    if (me > 0) { // adjust rpt
        GHOST_CALL_RETURN(ghost_malloc((void **)&rpt_w_offs,(mat->nrows+1)*sizeof(ghost_gidx_t)));
        for (i=0; i<mat->nrows+1; i++) {
            rpt_w_offs[i] = CR(mat)->rpt[i]+mat->context->lfEnt[me];
        }
    } else {
        rpt_w_offs = (ghost_gidx_t *)CR(mat)->rpt;
    }

    if (me == nrank-1) { // last process writes n+1st element
       to_write++;
    } 
    MPI_CALL_RETURN(MPI_File_write(fileh,rpt_w_offs,to_write,MPI_LONG_LONG,&status));
    
    if (me > 0) {
        free(rpt_w_offs); rpt_w_offs = NULL;
    }
  
    ghost_gidx_t *col;

    if (!(mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
       WARNING_LOG("The original column indices of the matrix have not been saved! The output will probably be corrupted");
       col = (ghost_gidx_t *)CR(mat)->col;
    } else {
       col = mat->col_orig;
    }

    MPI_CALL_RETURN(MPI_File_set_view(fileh, GHOST_BINCRS_SIZE_HEADER + (mat->context->gnrows+1)*GHOST_BINCRS_SIZE_RPT_EL + mat->context->lfEnt[me] * GHOST_BINCRS_SIZE_COL_EL, MPI_LONG_LONG, MPI_LONG_LONG, "native",MPI_INFO_NULL)); 
    MPI_CALL_RETURN(MPI_File_write(fileh,col,mat->context->lnEnts[me],MPI_LONG_LONG,&status));

#else // GHOST_HAVE_LONGIDX

    int64_t *rpt_cast_w_offs = NULL;
    size_t to_write = mat->nrows;
    GHOST_CALL_RETURN(ghost_malloc((void **)&rpt_cast_w_offs,(mat->nrows+1)*sizeof(int64_t)));
    for (i=0; i<mat->nrows+1; i++) {
        rpt_cast_w_offs[i] = CR(mat)->rpt[i]+mat->context->lfEnt[me];
    }

    if (me == nrank-1) { // last process writes n+1st element
       to_write++;
    } 
    MPI_CALL_RETURN(MPI_File_write(fileh,rpt_cast_w_offs,to_write,MPI_LONG_LONG,&status));

    free(rpt_cast_w_offs);
    
    ghost_lidx_t *col;
    int64_t *col_cast;
    
    if (!(mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
       WARNING_LOG("The original column indices of the matrix have not been saved! The output will probably be corrupted");
       col = CR(mat)->col;
    } else {
       col = mat->col_orig;
    }
    
    GHOST_CALL_RETURN(ghost_malloc((void **)&col_cast,(mat->context->lnEnts[me])*sizeof(int64_t)));
    
    for (i=0; i<mat->context->lnEnts[me]; i++) {
        col_cast[i] = (int64_t)col[i];
    }

    MPI_CALL_RETURN(MPI_File_set_view(fileh, GHOST_BINCRS_SIZE_HEADER + (mat->context->gnrows+1)*GHOST_BINCRS_SIZE_RPT_EL + mat->context->lfEnt[me] * GHOST_BINCRS_SIZE_COL_EL, MPI_LONG_LONG, MPI_LONG_LONG, "native",MPI_INFO_NULL)); 
    MPI_CALL_RETURN(MPI_File_write(fileh,col_cast,mat->context->lnEnts[me],MPI_LONG_LONG,&status));

#endif // GHOST_HAVE_LONGIDX

    ghost_gidx_t nnz;
    ghost_sparsemat_nnz(&nnz,mat);
    size_t sizeofEl;
    ghost_datatype_size(&sizeofEl,mat->traits->datatype);
    ghost_mpi_datatype_t mpiDT;
    ghost_mpi_datatype(&mpiDT,mat->traits->datatype);

    MPI_CALL_RETURN(MPI_File_set_view(fileh, GHOST_BINCRS_SIZE_HEADER + (mat->context->gnrows+1)*GHOST_BINCRS_SIZE_RPT_EL + nnz*GHOST_BINCRS_SIZE_COL_EL + mat->context->lfEnt[me]*sizeofEl, mpiDT, mpiDT,"native",MPI_INFO_NULL)); 
    MPI_CALL_RETURN(MPI_File_write(fileh,CR(mat)->val,mat->context->lnEnts[me],mpiDT,&status));
    
    MPI_CALL_RETURN(MPI_File_close(&fileh));

#else // GHOST_HAVE_MPI 
    size_t ret;
    FILE *filed;

    if ((filed = fopen64(matrixPath, "a")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s: %s",matrixPath,strerror(errno));
        return GHOST_ERR_IO;
    }
#ifdef GHOST_HAVE_LONGIDX_LOCAL
    if ((ret = fwrite(CR(mat)->rpt,8,mat->nrows+1,filed)) != mat->nrows+1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(CR(mat)->col,8,mat->nnz,filed)) != mat->nnz) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
#else
    ghost_gidx_t i,j;
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

#endif
    if ((ret = fwrite(CR(mat)->val,mat->elSize,mat->nnz,filed)) != mat->nnz) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }

    fclose(filed);
#endif

    return GHOST_SUCCESS;

}

static void CRS_free(ghost_sparsemat_t * mat)
{
    if (mat) {
        DEBUG_LOG(1,"Freeing CRS matrix");
        free(CR(mat)->rpt);
        free(CR(mat)->col);
        free(CR(mat)->val);

        ghost_sparsemat_destroy_common(mat);

        if (mat->localPart) {
            CRS_free(mat->localPart);
        }

        if (mat->remotePart) {
            CRS_free(mat->remotePart);
        }

        free(mat);
        DEBUG_LOG(1,"CRS matrix freed successfully");
    }
}

static ghost_error_t CRS_kernel_plain (ghost_sparsemat_t *mat, ghost_densemat_t * lhs, ghost_densemat_t * rhs, ghost_spmv_flags_t options, va_list argp)
{
    ghost_datatype_idx_t matDtIdx;
    ghost_datatype_idx_t vecDtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&matDtIdx,mat->traits->datatype));
    GHOST_CALL_RETURN(ghost_datatype_idx(&vecDtIdx,lhs->traits.datatype));

    return CRS_kernels_plain[matDtIdx][vecDtIdx](mat,lhs,rhs,options,argp);
}

