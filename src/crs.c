#define _XOPEN_SOURCE 500

#include "ghost/crs.h"
#include "ghost/util.h"
#include "ghost/mat.h"
#include "ghost/constants.h"
#include "ghost/affinity.h"
#include "ghost/context.h"
#include "ghost/io.h"

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

void (*CRS_kernels_plain[4][4]) (ghost_mat_t *, ghost_vec_t *, ghost_vec_t *, int options) = 
{{&ss_CRS_kernel_plain,&sd_CRS_kernel_plain,&sc_CRS_kernel_plain,&sz_CRS_kernel_plain},
    {&ds_CRS_kernel_plain,&dd_CRS_kernel_plain,&dc_CRS_kernel_plain,&dz_CRS_kernel_plain},
    {&cs_CRS_kernel_plain,&cd_CRS_kernel_plain,&cc_CRS_kernel_plain,&cz_CRS_kernel_plain},
    {&zs_CRS_kernel_plain,&zd_CRS_kernel_plain,&zc_CRS_kernel_plain,&zz_CRS_kernel_plain}};

void (*CRS_valToStr_funcs[4]) (void *, char *, int) = 
{&s_CRS_valToStr,&d_CRS_valToStr,&c_CRS_valToStr,&z_CRS_valToStr};

static ghost_mnnz_t CRS_nnz(ghost_mat_t *mat);
static ghost_midx_t CRS_nrows(ghost_mat_t *mat);
static ghost_midx_t CRS_ncols(ghost_mat_t *mat);
static ghost_error_t CRS_fromBin(ghost_mat_t *mat, char *matrixPath);
static void CRS_fromRowFunc(ghost_mat_t *mat, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, int flags);
static void CRS_printInfo(ghost_mat_t *mat);
static char * CRS_formatName(ghost_mat_t *mat);
static ghost_midx_t CRS_rowLen (ghost_mat_t *mat, ghost_midx_t i);
static size_t CRS_byteSize (ghost_mat_t *mat);
static void CRS_free(ghost_mat_t * mat);
static void CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
static void CRS_fromCRS(ghost_mat_t *mat, void *crs);
static void CRS_readColValOffset(ghost_mat_t *mat, char *matrixPath, ghost_mnnz_t offsetEnts, ghost_midx_t offsetRows, ghost_midx_t nRows, ghost_mnnz_t nEnts, int IOtype);
static void CRS_readHeader(ghost_mat_t *mat, char *matrixPath);
#ifdef GHOST_HAVE_MPI
static void CRS_split(ghost_mat_t *mat);
#endif
static void CRS_upload(ghost_mat_t *mat);
#ifdef GHOST_HAVE_OPENCL
static void CRS_kernel_CL (ghost_mat_t *mat, ghost_vec_t *, ghost_vec_t *, int);
#endif

static int swapReq = 0;


ghost_mat_t *ghost_CRS_init(ghost_context_t *ctx, ghost_mtraits_t *traits)
{
    ghost_mat_t *mat = (ghost_mat_t *)ghost_malloc(sizeof(ghost_mat_t));
    mat->context = ctx;
    mat->traits = traits;

    DEBUG_LOG(1,"Initializing CRS functions");
    if (!(mat->traits->flags & (GHOST_SPM_HOST | GHOST_SPM_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting matrix placement");
        mat->traits->flags |= GHOST_SPM_HOST;
        if (ghost_type == GHOST_TYPE_CUDAMGMT) {
            mat->traits->flags |= GHOST_SPM_DEVICE;
        }
    }

    if (mat->traits->flags & GHOST_SPM_DEVICE)
    {
#if GHOST_HAVE_CUDA
        WARNING_LOG("CUDA CRS SpMV has not yet been implemented!");
        //   mat->spmv = &ghost_cu_crsspmv;
#endif
    }
    else if (mat->traits->flags & GHOST_SPM_HOST)
    {
        mat->spmv   = &CRS_kernel_plain;
    }

    mat->fromFile = &CRS_fromBin;
    mat->fromRowFunc = &CRS_fromRowFunc;
    mat->fromCRS = &CRS_fromCRS;
    mat->printInfo = &CRS_printInfo;
    mat->formatName = &CRS_formatName;
    mat->rowLen   = &CRS_rowLen;
    mat->byteSize = &CRS_byteSize;
    mat->nnz      = &CRS_nnz;
    mat->nrows    = &CRS_nrows;
    mat->ncols    = &CRS_ncols;
    mat->destroy  = &CRS_free;
    mat->CLupload = &CRS_upload;
#ifdef GHOST_HAVE_MPI
    mat->split = &CRS_split;
#endif
#ifdef GHOST_HAVE_OPENCL
    if (traits->flags & GHOST_SPM_HOST)
        mat->spmv   = &CRS_kernel_plain;
    else 
        mat->spmv   = &CRS_kernel_CL;
#endif
    mat->data = (CR_TYPE *)ghost_malloc(sizeof(CR_TYPE));

    //    mat->rowPerm = NULL;
    //    mat->invRowPerm = NULL;
    mat->localPart = NULL;
    mat->remotePart = NULL;

    return mat;

}

static ghost_mnnz_t CRS_nnz(ghost_mat_t *mat)
{
    return CR(mat)->nEnts;
}
static ghost_midx_t CRS_nrows(ghost_mat_t *mat)
{
    return CR(mat)->nrows;
}
static ghost_midx_t CRS_ncols(ghost_mat_t *mat)
{
    return CR(mat)->ncols;
}

static void CRS_printInfo(ghost_mat_t *mat)
{
    UNUSED(mat);
    return;
}

static char * CRS_formatName(ghost_mat_t *mat)
{
    UNUSED(mat);
    return "CRS";
}

static ghost_midx_t CRS_rowLen (ghost_mat_t *mat, ghost_midx_t i)
{
    return CR(mat)->rpt[i+1] - CR(mat)->rpt[i];
}

static size_t CRS_byteSize (ghost_mat_t *mat)
{
    return (size_t)((CR(mat)->nrows+1)*sizeof(ghost_mnnz_t) + 
            CR(mat)->nEnts*(sizeof(ghost_midx_t)+ghost_sizeofDataType(mat->traits->datatype)));
}

static void CRS_fromRowFunc(ghost_mat_t *mat, ghost_midx_t maxrowlen, int base, ghost_spmFromRowFunc_t func, int flags)
{
    UNUSED(base);
    UNUSED(flags);
    int nprocs = 1;
#if GHOST_HAVE_MPI
    nprocs = ghost_getNumberOfRanks(mat->context->mpicomm);
#endif
    
    ghost_midx_t rowlen;
    ghost_midx_t i;
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);
    void *tmpval = ghost_malloc(maxrowlen*sizeofdt);
    ghost_midx_t *tmpcol = (ghost_midx_t *)ghost_malloc(maxrowlen*sizeof(ghost_midx_t));
    int me = ghost_getRank(mat->context->mpicomm);
    CR(mat)->ncols = mat->context->gncols;
    CR(mat)->nrows = mat->context->lnrows[me];
    CR(mat)->rpt = (ghost_midx_t *)ghost_malloc((CR(mat)->nrows+1)*sizeof(ghost_midx_t));
    CR(mat)->nEnts = 0;
    
    for( i = 0; i < CR(mat)->nrows; i++ ) {
        func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
        CR(mat)->nEnts += rowlen;
    }

    CR(mat)->col = (ghost_midx_t *)ghost_malloc(CR(mat)->nEnts*sizeof(ghost_midx_t));
    CR(mat)->val = ghost_malloc(CR(mat)->nEnts*sizeofdt);


    // TODO load balancing if distribution by nnz

    CR(mat)->rpt[0] = 0;
    for( i = 0; i < CR(mat)->nrows; i++ ) {
        func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
        CR(mat)->rpt[i+1] = CR(mat)->rpt[i]+rowlen;
        memcpy(&CR(mat)->col[CR(mat)->rpt[i]],tmpcol,rowlen*sizeof(ghost_midx_t));
        memcpy(&((char *)CR(mat)->val)[CR(mat)->rpt[i]*sizeofdt],tmpval,rowlen*sizeofdt);
    }

    if (!(mat->context->flags & GHOST_CONTEXT_GLOBAL)) {
#if GHOST_HAVE_MPI
        
        mat->context->lnEnts[me] = CR(mat)->nEnts;

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

static void CRS_fromCRS(ghost_mat_t *mat, void *crs)
{
    DEBUG_LOG(1,"Creating CRS matrix from CRS matrix");
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);
    CR_TYPE *cr = (CR_TYPE*)crs;
    ghost_midx_t i,j;


    //    mat->data = (CR_TYPE *)ghost_malloc(sizeof(CR_TYPE));
    CR(mat)->nrows = cr->nrows;
    CR(mat)->ncols = cr->ncols;
    CR(mat)->nEnts = cr->nEnts;

    CR(mat)->rpt = (ghost_midx_t *)ghost_malloc((cr->nrows+1)*sizeof(ghost_midx_t));
    CR(mat)->col = (ghost_midx_t *)ghost_malloc(cr->nEnts*sizeof(ghost_midx_t));
    CR(mat)->val = ghost_malloc(cr->nEnts*sizeofdt);

#pragma omp parallel for schedule(runtime)
    for( i = 0; i < CR(mat)->nrows+1; i++ ) {
        CR(mat)->rpt[i] = cr->rpt[i];
    }

#pragma omp parallel for schedule(runtime) private(j)
    for( i = 0; i < CR(mat)->nrows; i++ ) {
        for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
            CR(mat)->col[j] = cr->col[j];
            memcpy(&CR(mat)->val[j*sizeofdt],&cr->val[j*sizeofdt],sizeofdt);
        }
    }

    DEBUG_LOG(1,"Successfully created CRS matrix from CRS data");

    // TODO OpenCL upload

}

#ifdef GHOST_HAVE_MPI

static void CRS_split(ghost_mat_t *mat)
    {
    CR_TYPE *fullCR = CR(mat);
    CR_TYPE *localCR = NULL, *remoteCR = NULL;
    DEBUG_LOG(1,"Splitting the CRS matrix into a local and remote part");
    int j;
    int i;
    int me;

    ghost_mnnz_t lnEnts_l, lnEnts_r;
    int current_l, current_r;

    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);

    me = ghost_getRank(mat->context->mpicomm);

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

        localCR->val = ghost_malloc(lnEnts_l*sizeofdt); 
        localCR->col = (ghost_midx_t*) ghost_malloc(lnEnts_l*sizeof( ghost_midx_t )); 
        localCR->rpt = (ghost_midx_t*) ghost_malloc((mat->context->lnrows[me]+1)*sizeof( ghost_midx_t )); 

        remoteCR->val = ghost_malloc(lnEnts_r*sizeofdt); 
        remoteCR->col = (ghost_midx_t*) ghost_malloc(lnEnts_r*sizeof( ghost_midx_t )); 
        remoteCR->rpt = (ghost_midx_t*) ghost_malloc((mat->context->lnrows[me]+1)*sizeof( ghost_midx_t )); 

        localCR->nrows = mat->context->lnrows[me];
        localCR->nEnts = lnEnts_l;

        remoteCR->nrows = mat->context->lnrows[me];
        remoteCR->nEnts = lnEnts_r;

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

    mat->localPart = ghost_createMatrix(mat->context,&mat->traits[0],1);
    free(mat->localPart->data); // has been allocated in init()
    mat->localPart->symmetry = mat->symmetry;
    mat->localPart->data = localCR;
    //CR(mat->localPart)->rpt = localCR->rpt;

    mat->remotePart = ghost_createMatrix(mat->context,&mat->traits[0],1);
    free(mat->remotePart->data); // has been allocated in init()
    mat->remotePart->data = remoteCR;
}

#endif

static void CRS_readHeader(ghost_mat_t *mat, char *matrixPath)
{
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

    CR(mat)->nrows = (ghost_midx_t)header.nrows;
    CR(mat)->ncols = (ghost_midx_t)header.ncols;
    CR(mat)->nEnts = (ghost_midx_t)header.nnz;

    DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",CR(mat)->nrows,CR(mat)->ncols,CR(mat)->nEnts);

}

ghost_midx_t * CRS_readRpt(ghost_midx_t nrpt, char *matrixPath)
{
    size_t ret;
    FILE *filed;
    ghost_midx_t i;

    DEBUG_LOG(1,"Reading row pointers from %s",matrixPath);

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ABORT("Could not open binary CRS file %s",matrixPath);
    }

    DEBUG_LOG(2,"Allocate memory for rpt for %d rows",nrpt);
    ghost_midx_t *rpt = (ghost_midx_t *)ghost_malloc( nrpt*sizeof(ghost_midx_t));

    DEBUG_LOG(1,"NUMA-placement for rpt");
#pragma omp parallel for schedule(runtime)
    for( i = 0; i < nrpt; i++ ) {
        rpt[i] = 0;
    }

    DEBUG_LOG(2,"Reading array with row-offsets");
#ifdef LONGIDX
    if (swapReq) {
        int64_t tmp;
        if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
            ABORT("Seek failed");
        }
        for( i = 0; i < nrpt; i++ ) {
            if ((ret = fread(&tmp, GHOST_BINCRS_SIZE_RPT_EL, 1,filed)) != 1){
                ABORT("fread failed: %s (%zu)",strerror(errno),ret);
            }
            tmp = bswap_64(tmp);
            rpt[i] = tmp;
        }
    } else {
        if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
            ABORT("Seek failed");
        }
        if ((ret = fread(rpt, GHOST_BINCRS_SIZE_RPT_EL, nrpt,filed)) != (nrpt)){
            ABORT("fread failed: %s (%zu)",strerror(errno),ret);
        }
    }
#else // casting
    DEBUG_LOG(1,"Casting from 64 bit to 32 bit row pointers");
    int64_t *tmp = (int64_t *)malloc((nrpt)*8);
    if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
        ABORT("Seek failed");
    }
    if ((ghost_midx_t)(ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, nrpt,filed)) != (nrpt)){
        ABORT("fread failed: %s (%zu)",strerror(errno),ret);
    }

    if (swapReq) {
        for( i = 0; i < nrpt; i++ ) {
            if (tmp[i] >= (int64_t)INT_MAX) {
                ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
            }
            rpt[i] = (ghost_midx_t)(bswap_64(tmp[i]));
        }
    } else {
        for( i = 0; i < nrpt; i++ ) {
            if (tmp[i] >= (int64_t)INT_MAX) {
                ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
            }
            rpt[i] = (ghost_midx_t)(tmp[i]);
        }
    }
    free(tmp);
#endif
    fclose(filed);

    return rpt;
}

static void CRS_readColValOffset(ghost_mat_t *mat, char *matrixPath, ghost_mnnz_t offsetEnts, ghost_midx_t offsetRows, ghost_midx_t nRows, ghost_mnnz_t nEnts, int IOtype)
{

    size_t ret;
    size_t sizeofdt = ghost_sizeofDataType(mat->traits->datatype);
    UNUSED(offsetRows);    
    UNUSED(IOtype);

    ghost_midx_t i, j;
    int datatype;
    FILE *filed;

    off64_t offs;

    //    file = open(matrixPath,O_RDONLY);

    DEBUG_LOG(1,"Reading %"PRmatNNZ" cols and vals from binary file %s with offset %"PRmatNNZ,nEnts, matrixPath,offsetEnts);

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ABORT("Could not open binary CRS file %s",matrixPath);
    }

    if (fseeko(filed,16,SEEK_SET)) {
        ABORT("Seek failed");
    }
    if ((ret = fread(&datatype, 4, 1,filed)) != (1)){
        ABORT("fread failed: %s (%zu)",strerror(errno),ret);
    }
    if (swapReq) datatype = bswap_32(datatype);

    DEBUG_LOG(1,"CRS matrix has %"PRmatIDX" rows, %"PRmatIDX" cols and %"PRmatNNZ" nonzeros",CR(mat)->nrows,CR(mat)->ncols,CR(mat)->nEnts);

    DEBUG_LOG(2,"Allocate memory for CR(mat)->col and CR(mat)->val");
    CR(mat)->col       = (ghost_midx_t *) ghost_malloc( nEnts * sizeof(ghost_midx_t));
    CR(mat)->val       = ghost_malloc( nEnts * sizeofdt);

    DEBUG_LOG(2,"NUMA-placement for CR(mat)->val and CR(mat)->col");
#pragma omp parallel for schedule(runtime) private(j)
    for(i = 0 ; i < nRows; ++i) {
        for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
            CR(mat)->val[j*sizeofdt] = (char)0;
            CR(mat)->col[j] = -1;
        }
    }


    DEBUG_LOG(1,"Reading array with column indices");
    offs = GHOST_BINCRS_SIZE_HEADER+
        GHOST_BINCRS_SIZE_RPT_EL*(CR(mat)->nrows+1)+
        GHOST_BINCRS_SIZE_COL_EL*offsetEnts;
    if (fseeko(filed,offs,SEEK_SET)) {
        ABORT("Seek failed");
    }
#ifdef LONGIDX
    if (swapReq) {
        int64_t *tmp = (int64_t *)malloc(nEnts*8);
        if ((ret = fread(tmp, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (nEnts)){
            ABORT("fread failed: %s (%zu)",strerror(errno),ret);
        }
        for( i = 0; i < nEnts; i++ ) {
            CR(mat)->col[i] = bswap_64(tmp[i]);
        }
    } else {
        if ((ret = fread(CR(mat)->col, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (nEnts)){
            ABORT("fread failed: %s (%zu)",strerror(errno),ret);
        }
    }
#else // casting
    DEBUG_LOG(1,"Casting from 64 bit to 32 bit column indices");
    int64_t *tmp = (int64_t *)ghost_malloc(nEnts*8);
    if ((ghost_midx_t)(ret = fread(tmp, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (nEnts)){
        ABORT("fread failed: %s (%zu)",strerror(errno),ret);
    }
    for(i = 0 ; i < nRows; ++i) {
        for(j = CR(mat)->rpt[i]; j < CR(mat)->rpt[i+1] ; j++) {
            if (tmp[j] >= (int64_t)INT_MAX) {
                ABORT("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
            }
            if (swapReq) CR(mat)->col[j] = (ghost_midx_t)(bswap_64(tmp[j]));
            else CR(mat)->col[j] = (ghost_midx_t)tmp[j];
        }
    }
    free(tmp);
#endif
    // minimal size of value
    size_t valSize = sizeof(float);
    if (datatype & GHOST_BINCRS_DT_DOUBLE)
        valSize *= 2;

    if (datatype & GHOST_BINCRS_DT_COMPLEX)
        valSize *= 2;


    DEBUG_LOG(1,"Reading array with values");
    offs = GHOST_BINCRS_SIZE_HEADER+
        GHOST_BINCRS_SIZE_RPT_EL*(CR(mat)->nrows+1)+
        GHOST_BINCRS_SIZE_COL_EL*CR(mat)->nEnts+
        ghost_sizeofDataType(datatype)*offsetEnts;
    if (fseeko(filed,offs,SEEK_SET)) {
        ABORT("Seek failed");
    }

    if (datatype == mat->traits->datatype) {
        if (swapReq) {
            uint8_t *tmpval = (uint8_t *)ghost_malloc(nEnts*valSize);
            if ((ghost_midx_t)(ret = fread(tmpval, valSize, nEnts,filed)) != (nEnts)){
                ABORT("fread failed for val: %s (%zu)",strerror(errno),ret);
            }
            if (mat->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
                if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t *a = (uint32_t *)tmpval;
                        uint32_t rswapped = bswap_32(a[2*i]);
                        uint32_t iswapped = bswap_32(a[2*i+1]);
                        memcpy(&(CR(mat)->val[i]),&rswapped,4);
                        memcpy(&(CR(mat)->val[i])+4,&iswapped,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint64_t *a = (uint64_t *)tmpval;
                        uint64_t rswapped = bswap_64(a[2*i]);
                        uint64_t iswapped = bswap_64(a[2*i+1]);
                        memcpy(&(CR(mat)->val[i]),&rswapped,8);
                        memcpy(&(CR(mat)->val[i])+8,&iswapped,8);
                    }
                }
            } else {
                if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t *a = (uint32_t *)tmpval;
                        uint32_t swapped = bswap_32(a[i]);
                        memcpy(&(CR(mat)->val[i]),&swapped,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint64_t *a = (uint64_t *)tmpval;
                        uint64_t swapped = bswap_64(a[i]);
                        memcpy(&(CR(mat)->val[i]),&swapped,8);
                    }
                }

            }
        } else {
            if ((ghost_midx_t)(ret = fread(CR(mat)->val, ghost_sizeofDataType(datatype), nEnts,filed)) != (nEnts)){
                ABORT("fread failed for val: %s (%zu)",strerror(errno),ret);
            }
        }
    } else {
        INFO_LOG("This matrix is supposed to be of %s data but"
                " the file contains %s data. Casting...",ghost_datatypeName(mat->traits->datatype),ghost_datatypeName(datatype));


        uint8_t *tmpval = (uint8_t *)ghost_malloc(nEnts*valSize);
        if ((ghost_midx_t)(ret = fread(tmpval, valSize, nEnts,filed)) != (nEnts)){
            ABORT("fread failed for val: %s (%zu)",strerror(errno),ret);
        }

        if (swapReq) {
            ABORT("Not yet supported!");
            if (mat->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
                if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t re = bswap_32(tmpval[i*valSize]);
                        uint32_t im = bswap_32(tmpval[i*valSize+valSize/2]);
                        memcpy(&CR(mat)->val[i*sizeofdt],&re,4);
                        memcpy(&CR(mat)->val[i*sizeofdt+4],&im,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t re = bswap_64(tmpval[i*valSize]);
                        uint32_t im = bswap_64(tmpval[i*valSize+valSize/2]);
                        memcpy(&CR(mat)->val[i*sizeofdt],&re,8);
                        memcpy(&CR(mat)->val[i*sizeofdt+8],&im,8);
                    }
                }
            } else {
                if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t val = bswap_32(tmpval[i*valSize]);
                        memcpy(&CR(mat)->val[i*sizeofdt],&val,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t val = bswap_64(tmpval[i*valSize]);
                        memcpy(&CR(mat)->val[i*sizeofdt],&val,8);
                    }
                }

            }

        } else {
            ghost_castArray_funcs[ghost_dataTypeIdx(mat->traits->datatype)][ghost_dataTypeIdx(datatype)](CR(mat)->val,tmpval,nEnts);
        }

        free(tmpval);
    }
    fclose(filed);
}

static void CRS_upload(ghost_mat_t *mat)
{
    DEBUG_LOG(1,"Uploading CRS matrix to device");
#ifdef GHOST_HAVE_OPENCL
    if (!(mat->traits->flags & GHOST_SPM_HOST)) {
        DEBUG_LOG(1,"Creating matrix on OpenCL device");
        CR(mat)->clmat = (CL_CR_TYPE *)ghost_malloc(sizeof(CL_CR_TYPE));
        CR(mat)->clmat->rpt = CL_allocDeviceMemory((CR(mat)->nrows+1)*sizeof(ghost_cl_mnnz_t));
        CR(mat)->clmat->col = CL_allocDeviceMemory((CR(mat)->nEnts)*sizeof(ghost_cl_midx_t));
        CR(mat)->clmat->val = CL_allocDeviceMemory((CR(mat)->nEnts)*ghost_sizeofDataType(mat->traits->datatype));

        CR(mat)->clmat->nrows = CR(mat)->nrows;
        CL_copyHostToDevice(CR(mat)->clmat->rpt, CR(mat)->rpt, (CR(mat)->nrows+1)*sizeof(ghost_cl_mnnz_t));
        CL_copyHostToDevice(CR(mat)->clmat->col, CR(mat)->col, CR(mat)->nEnts*sizeof(ghost_cl_midx_t));
        CL_copyHostToDevice(CR(mat)->clmat->val, CR(mat)->val, CR(mat)->nEnts*ghost_sizeofDataType(mat->traits->datatype));

        cl_int err;
        cl_uint numKernels;
        /*    char shift[32];
              CRS_valToStr_funcs[ghost_dataTypeIdx(mat->traits->datatype)](mat->traits->shift,shift,32);
              printf("SHIFT: %f   %s\n",*((double *)(mat->traits->shift)),shift);
         */
        char options[128];
        if (mat->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
            if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
                strncpy(options," -DGHOST_MAT_C ",15);
            } else {
                strncpy(options," -DGHOST_MAT_Z ",15);
            }
        } else {
            if (mat->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
                strncpy(options," -DGHOST_MAT_S ",15);
            } else {
                strncpy(options," -DGHOST_MAT_D ",15);
            }

        }
        strncpy(options+15," -DGHOST_VEC_S ",15);
        cl_program program = CL_registerProgram("crs_clkernel.cl",options);
        CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
        DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
        mat->clkernel[0] = clCreateKernel(program,"CRS_kernel",&err);
        CL_checkerror(err);

        strncpy(options+15," -DGHOST_VEC_D ",15);
        program = CL_registerProgram("crs_clkernel.cl",options);
        CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
        DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
        mat->clkernel[1] = clCreateKernel(program,"CRS_kernel",&err);
        CL_checkerror(err);

        strncpy(options+15," -DGHOST_VEC_C ",15);
        program = CL_registerProgram("crs_clkernel.cl",options);
        CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
        DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
        mat->clkernel[2] = clCreateKernel(program,"CRS_kernel",&err);
        CL_checkerror(err);

        strncpy(options+15," -DGHOST_VEC_Z ",15);
        program = CL_registerProgram("crs_clkernel.cl",options);
        CL_safecall(clCreateKernelsInProgram(program,0,NULL,&numKernels));
        DEBUG_LOG(1,"There are %u OpenCL kernels",numKernels);
        mat->clkernel[3] = clCreateKernel(program,"CRS_kernel",&err);
        CL_checkerror(err);

        int i;
        for (i=0; i<4; i++) {
            CL_safecall(clSetKernelArg(mat->clkernel[i],3,sizeof(int), &(CR(mat)->clmat->nrows)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],4,sizeof(cl_mem), &(CR(mat)->clmat->rpt)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],5,sizeof(cl_mem), &(CR(mat)->clmat->col)));
            CL_safecall(clSetKernelArg(mat->clkernel[i],6,sizeof(cl_mem), &(CR(mat)->clmat->val)));
        }
    }
#else
    if (mat->traits->flags & GHOST_SPM_DEVICE) {
        ABORT("Device matrix cannot be created without OpenCL");
    }
#endif
}

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

    if (mat->context->flags & GHOST_CONTEXT_GLOBAL) {
        DEBUG_LOG(1,"Reading in a global context");
        CRS_readHeader(mat,matrixPath);
        CR(mat)->rpt = CRS_readRpt(CR(mat)->nrows+1,matrixPath);
        CRS_readColValOffset(mat, matrixPath, 0, 0, CR(mat)->nrows, CR(mat)->nEnts, GHOST_IO_STD);
    } else {
#ifdef GHOST_HAVE_MPI
        DEBUG_LOG(1,"Reading in a distributed context");
        DEBUG_LOG(1,"Creating distributed context with parallel MPI-IO");

        ghost_midx_t i;

        ghost_context_t *context = mat->context;
        int me = ghost_getRank(context->mpicomm); 
        int nprocs = ghost_getNumberOfRanks(context->mpicomm);

        CRS_readHeader(mat,matrixPath);  // read header

        if (ghost_getRank(context->mpicomm) == 0) {
            if (context->flags & GHOST_CONTEXT_WORKDIST_NZE) { // rpt has already been read
                ((CR_TYPE *)(mat->data))->rpt = context->rpt;
            } else {
                CR(mat)->rpt = CRS_readRpt(CR(mat)->nrows+1,matrixPath);  // read rpt
                context->lfEnt[0] = 0;

                for (i=1; i<nprocs; i++){
                    context->lfEnt[i] = CR(mat)->rpt[context->lfRow[i]];
                }
                for (i=0; i<nprocs-1; i++){
                    context->lnEnts[i] = context->lfEnt[i+1] - context->lfEnt[i] ;
                }

                context->lnEnts[nprocs-1] = CR(mat)->nEnts - context->lfEnt[nprocs-1];
            }
        }
        MPI_safecall(MPI_Bcast(context->lfEnt,  nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
        MPI_safecall(MPI_Bcast(context->lnEnts, nprocs, ghost_mpi_dt_midx, 0, context->mpicomm));
            

        DEBUG_LOG(1,"Mallocing space for %"PRmatIDX" rows",context->lnrows[me]);

        if (ghost_getRank(context->mpicomm) != 0) {
            CR(mat)->rpt = (ghost_midx_t *)ghost_malloc((context->lnrows[me]+1)*sizeof(ghost_midx_t));
        }

        MPI_Request req[nprocs];
        MPI_Status stat[nprocs];
        int msgcount = 0;

        for (i=0;i<nprocs;i++) 
            req[i] = MPI_REQUEST_NULL;

        if (ghost_getRank(context->mpicomm) != 0) {
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

        CRS_readColValOffset(mat,matrixPath, context->lfEnt[me], context->lfRow[me], context->lnrows[me], context->lnEnts[me], GHOST_IO_STD);

        DEBUG_LOG(1,"Adjust number of rows and number of nonzeros");
        CR(mat)->nrows = context->lnrows[me];
        CR(mat)->nEnts = context->lnEnts[me];
        mat->split(mat);
#else
        ABORT("Trying to create a distributed context without MPI!");
#endif
    }
    DEBUG_LOG(1,"Matrix read in successfully");
#ifdef GHOST_HAVE_OPENCL
    if (!(mat->traits->flags & GHOST_SPM_HOST))
        mat->CLupload(mat);
#endif

    return GHOST_SUCCESS;

}

static void CRS_free(ghost_mat_t * mat)
{
    if (mat) {
        DEBUG_LOG(1,"Freeing CRS matrix");
#ifdef GHOST_HAVE_OPENCL
        if (mat->traits->flags & GHOST_SPM_DEVICE) {
            CL_freeDeviceMemory(CR(mat)->clmat->rpt);
            CL_freeDeviceMemory(CR(mat)->clmat->col);
            CL_freeDeviceMemory(CR(mat)->clmat->val);
        }
#endif
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

static void CRS_kernel_plain (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
    /*    if (mat->symmetry == GHOST_BINCRS_SYMM_SYMMETRIC) {
          ghost_midx_t i, j;
          ghost_dt hlp1;
          ghost_midx_t col;
          ghost_dt val;

#pragma omp    parallel for schedule(runtime) private (hlp1, j, col, val)
for (i=0; i<CR(mat)->nrows; i++){
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


    CRS_kernels_plain[ghost_dataTypeIdx(mat->traits->datatype)][ghost_dataTypeIdx(lhs->traits->datatype)](mat,lhs,rhs,options);
    }

#ifdef GHOST_HAVE_OPENCL
static void CRS_kernel_CL (ghost_mat_t *mat, ghost_vec_t * lhs, ghost_vec_t * rhs, int options)
{
    cl_kernel kernel = mat->clkernel[ghost_dataTypeIdx(rhs->traits->datatype)];
    CL_safecall(clSetKernelArg(kernel,0,sizeof(cl_mem), &(lhs->CL_val_gpu)));
    CL_safecall(clSetKernelArg(kernel,1,sizeof(cl_mem), &(rhs->CL_val_gpu)));
    CL_safecall(clSetKernelArg(kernel,2,sizeof(int), &options));
    if (mat->traits->shift != NULL) {
        CL_safecall(clSetKernelArg(kernel,7,ghost_sizeofDataType(mat->traits->datatype), mat->traits->shift));
    } else {
        if (options & GHOST_SPMVM_APPLY_SHIFT)
            ABORT("A shift should be applied but the pointer is NULL!");
        complex double foo = 0.+I*0.; // should never be needed
        CL_safecall(clSetKernelArg(kernel,7,ghost_sizeofDataType(mat->traits->datatype), &foo))
    }


    size_t gSize = (size_t)CR(mat)->clmat->nrows;

    CL_enqueueKernel(kernel,1,&gSize,NULL);
}
#endif
