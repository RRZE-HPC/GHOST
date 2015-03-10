#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/crs.h"
#include "ghost/sell.h"
#include "ghost/sparsemat.h"
#include "ghost/context.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#include "ghost/bincrs.h"
#include "ghost/matrixmarket.h"
#include "ghost/instr.h"

#include <libgen.h>

const ghost_sparsemat_src_rowfunc_t GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER = {
    .func = NULL,
    .maxrowlen = 0,
    .base = 0,
    .flags = GHOST_SPARSEMAT_FROMROWFUNC_DEFAULT
};
    

const ghost_sparsemat_traits_t GHOST_SPARSEMAT_TRAITS_INITIALIZER = {
    .format = GHOST_SPARSEMAT_CRS,
    .flags = GHOST_SPARSEMAT_DEFAULT,
    .symmetry = GHOST_SPARSEMAT_SYMM_GENERAL,
    .aux = NULL,
    .scotchStrat = (char*)GHOST_SCOTCH_STRAT_DEFAULT,
    .sortScope = 1,
    .datatype = (ghost_datatype_t) (GHOST_DT_DOUBLE|GHOST_DT_REAL)
};


ghost_error_t ghost_sparsemat_create(ghost_sparsemat_t ** mat, ghost_context_t *context, ghost_sparsemat_traits_t *traits, int nTraits)
{
    UNUSED(nTraits);
    ghost_error_t ret = GHOST_SUCCESS;

    int me;
    GHOST_CALL_GOTO(ghost_rank(&me, context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)mat,sizeof(ghost_sparsemat_t)),err,ret);

    (*mat)->traits = traits;
    (*mat)->context = context;
    (*mat)->localPart = NULL;
    (*mat)->remotePart = NULL;
    (*mat)->name = "Sparse matrix";
    (*mat)->col_orig = NULL;
    (*mat)->data = NULL;
    (*mat)->nzDist = NULL;
    (*mat)->fromFile = NULL;
    (*mat)->toFile = NULL;
    (*mat)->fromRowFunc = NULL;
    (*mat)->auxString = NULL;
    (*mat)->formatName = NULL;
    (*mat)->rowLen = NULL;
    (*mat)->byteSize = NULL;
    (*mat)->permute = NULL;
    (*mat)->destroy = NULL;
    (*mat)->string = NULL;
    (*mat)->upload = NULL;
    (*mat)->permute = NULL;
    (*mat)->spmv = NULL;
    (*mat)->destroy = NULL;
    (*mat)->split = NULL;
    (*mat)->bandwidth = 0;
    (*mat)->lowerBandwidth = 0;
    (*mat)->upperBandwidth = 0;
    (*mat)->avgRowBand = 0.;
    (*mat)->avgAvgRowBand = 0.;
    (*mat)->smartRowBand = 0.;
    (*mat)->maxRowLen = 0;
    (*mat)->nMaxRows = 0;
    (*mat)->variance = 0.;
    (*mat)->deviation = 0.;
    (*mat)->cv = 0.;
    (*mat)->nrows = context->lnrows[me];
    (*mat)->nrowsPadded = (*mat)->nrows;
    (*mat)->ncols = context->gncols;
    (*mat)->nEnts = 0;
    (*mat)->nnz = 0;
    (*mat)->ncolors = 0;
    (*mat)->color_ptr = NULL;

    if ((*mat)->traits->sortScope == GHOST_SPARSEMAT_SORT_GLOBAL) {
        (*mat)->traits->sortScope = (*mat)->context->gnrows;
    } else if ((*mat)->traits->sortScope == GHOST_SPARSEMAT_SORT_LOCAL) {
        (*mat)->traits->sortScope = (*mat)->nrows;
    }

#ifdef GHOST_GATHER_GLOBAL_INFO
    GHOST_CALL_GOTO(ghost_malloc((void **)&((*mat)->nzDist),sizeof(ghost_gidx_t)*(2*context->gnrows-1)),err,ret);
#endif
    GHOST_CALL_GOTO(ghost_datatype_size(&(*mat)->elSize,(*mat)->traits->datatype),err,ret);

    switch (traits->format) {
        case GHOST_SPARSEMAT_CRS:
            GHOST_CALL_GOTO(ghost_crs_init(*mat),err,ret);
            break;
        case GHOST_SPARSEMAT_SELL:
            GHOST_CALL_GOTO(ghost_sell_init(*mat),err,ret);
            break;
        default:
            WARNING_LOG("Invalid sparse matrix format. Falling back to CRS!");
            traits->format = GHOST_SPARSEMAT_CRS;
            GHOST_CALL_GOTO(ghost_crs_init(*mat),err,ret);
    }

    goto out;
err:
    ERROR_LOG("Error. Free'ing resources");
    free(*mat); *mat = NULL;

out:
    return ret;    
}

ghost_error_t ghost_sparsemat_sortrow(ghost_gidx_t *col, char *val, size_t valSize, ghost_lidx_t rowlen, ghost_lidx_t stride)
{
    ghost_lidx_t n;
    ghost_lidx_t c;
    ghost_lidx_t swpcol;
    char swpval[valSize];
    for (n=rowlen; n>1; n--) {
        for (c=0; c<n-1; c++) {
            if (col[c*stride] > col[(c+1)*stride]) {
                swpcol = col[c*stride];
                col[c*stride] = col[(c+1)*stride];
                col[(c+1)*stride] = swpcol; 

                memcpy(&swpval,&val[c*stride*valSize],valSize);
                memcpy(&val[c*stride*valSize],&val[(c+1)*stride*valSize],valSize);
                memcpy(&val[(c+1)*stride*valSize],&swpval,valSize);
            }
        }
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_fromfunc_common(ghost_sparsemat_t *mat, ghost_sparsemat_src_rowfunc_t *src)
{
    ghost_error_t ret = GHOST_SUCCESS;
    
    int nprocs = 1;
    int me;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    
#ifdef GHOST_GATHER_GLOBAL_INFO
    memset(mat->nzDist,0,sizeof(ghost_gidx_t)*(2*mat->context->gnrows-1));
#endif
    mat->lowerBandwidth = 0;
    mat->upperBandwidth = 0;
    
    if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_PERMUTE;
    }

    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
        if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_sparsemat_perm_scotch(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } else if (mat->traits->flags & GHOST_SPARSEMAT_COLOR) {
            ghost_sparsemat_perm_color(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } else {
            ghost_sparsemat_perm_sort(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC,mat->traits->sortScope);
        }
    } else {
        if (mat->traits->sortScope > 1) {
            WARNING_LOG("Ignoring sorting scope");
        }
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_NOT_PERMUTE_COLS;
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_NOT_SORT_COLS;
    }

    goto out;

err:

out:


    return ret;
}

ghost_error_t ghost_sparsemat_fromfile_common(ghost_sparsemat_t *mat, char *matrixPath, ghost_lidx_t **rpt) 
{
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_gidx_t i;
    ghost_bincrs_header_t header;

#ifdef GHOST_HAVE_MPI
    MPI_Request *req = NULL;
    MPI_Status *stat = NULL;
#endif

    ghost_bincrs_header_read(&header,matrixPath);
    int nprocs = 1;
    int me;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);

#ifdef GHOST_HAVE_MPI
    GHOST_CALL_GOTO(ghost_malloc((void **)&req,sizeof(MPI_Request)*nprocs),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&stat,sizeof(MPI_Status)*nprocs),err,ret);
#endif

    if (header.version != 1) {
        ERROR_LOG("Can not read version %d of binary CRS format!",header.version);
        return GHOST_ERR_IO;
    }

    if (header.base != 0) {
        ERROR_LOG("Can not read matrix with %d-based indices!",header.base);
        return GHOST_ERR_IO;
    }

    if (!ghost_sparsemat_symmetry_valid((ghost_sparsemat_symmetry_t)header.symmetry)) {
        ERROR_LOG("Symmetry is invalid! (%d)",header.symmetry);
        return GHOST_ERR_IO;
    }

    if (header.symmetry != GHOST_BINCRS_SYMM_GENERAL) {
        ERROR_LOG("Can not handle symmetry different to general at the moment!");
        return GHOST_ERR_IO;
    }

    if (!ghost_datatype_valid((ghost_datatype_t)header.datatype)) {
        ERROR_LOG("Datatype is invalid! (%d)",header.datatype);
        return GHOST_ERR_IO;
    }

#ifdef GHOST_GATHER_GLOBAL_INFO
    memset(mat->nzDist,0,sizeof(ghost_gidx_t)*(2*mat->context->gnrows-1));
#endif
    mat->lowerBandwidth = 0;
    mat->upperBandwidth = 0;
    mat->name = basename(matrixPath);
    mat->traits->symmetry = (ghost_sparsemat_symmetry_t)header.symmetry;
    mat->ncols = (ghost_gidx_t)header.ncols;
        
    if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_PERMUTE;
    }

    if (mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) {
        if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_sparsemat_perm_scotch(mat,matrixPath,GHOST_SPARSEMAT_SRC_FILE);
        } else if (mat->traits->flags & GHOST_SPARSEMAT_COLOR) {
            ghost_sparsemat_perm_color(mat,matrixPath,GHOST_SPARSEMAT_SRC_FILE);
        } else {
            ghost_sparsemat_perm_sort(mat,matrixPath,GHOST_SPARSEMAT_SRC_FILE,mat->traits->sortScope);
        }
    } else {
        if (mat->traits->sortScope > 1) {
            WARNING_LOG("Ignoring sorting scope");
        }
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_NOT_PERMUTE_COLS;
        mat->traits->flags |= (ghost_sparsemat_flags_t)GHOST_SPARSEMAT_NOT_SORT_COLS;
    }


#ifdef GHOST_HAVE_MPI
        ghost_gidx_t *grpt = NULL;
        int grptAllocated = 1;
        if (mat->context->flags & GHOST_CONTEXT_DIST_NZ) { // rpt has already been read at rank 0
            int msgcount = 0;

            for (i=0;i<nprocs;i++) {
                req[i] = MPI_REQUEST_NULL;
            }

            if (me == 0) {
                grpt = mat->context->rpt;
                for (i=1;i<nprocs;i++) {
                    MPI_CALL_GOTO(MPI_Isend(&grpt[mat->context->lfRow[i]],mat->context->lnrows[i]+1,ghost_mpi_dt_gidx,i,i,mat->context->mpicomm,&req[msgcount]),err,ret);
                    msgcount++;
                }
                for (i=1; i<nprocs; i++){
                    mat->context->lfEnt[i] = grpt[mat->context->lfRow[i]];
                }
                for (i=0; i<nprocs-1; i++){
                    mat->context->lnEnts[i] = mat->context->lfEnt[i+1] - mat->context->lfEnt[i] ;
                }

                mat->context->lnEnts[nprocs-1] = header.nnz - mat->context->lfEnt[nprocs-1];
            } else {
                GHOST_CALL_GOTO(ghost_malloc_align((void **)grpt,(mat->context->lnrows[me]+1)*sizeof(ghost_gidx_t),GHOST_DATA_ALIGNMENT),err,ret);
                if (!grpt) { // This should not happen but Clang SCA complained so I added this test
                    ERROR_LOG("Malloc failed");
                    ret = GHOST_ERR_UNKNOWN;
                    goto err;
                }
                grptAllocated = 1;
#pragma omp parallel for schedule(runtime)
                for (i = 0; i < mat->context->lnrows[me]+1; i++) {
                    grpt[i] = 0;
                }
                MPI_CALL_GOTO(MPI_Irecv(grpt,mat->context->lnrows[me]+1,ghost_mpi_dt_gidx,0,me,mat->context->mpicomm,&req[msgcount]),err,ret);
                msgcount++;
            }
            MPI_CALL_GOTO(MPI_Waitall(msgcount,req,stat),err,ret);
            MPI_CALL_GOTO(MPI_Bcast(mat->context->lfEnt,  nprocs, ghost_mpi_dt_gidx, 0, mat->context->mpicomm),err,ret);
            MPI_CALL_GOTO(MPI_Bcast(mat->context->lnEnts, nprocs, ghost_mpi_dt_lidx, 0, mat->context->mpicomm),err,ret);
            
            
        } else { // read rpt and compute first entry and number of entries
            GHOST_CALL_GOTO(ghost_malloc_align((void **)&grpt,(mat->context->lnrows[me]+1) * sizeof(ghost_gidx_t), GHOST_DATA_ALIGNMENT),err,ret);
            grptAllocated = 1;
            GHOST_CALL_GOTO(ghost_bincrs_rpt_read(grpt, matrixPath, mat->context->lfRow[me], mat->context->lnrows[me]+1, mat->context->permutation),err,ret); 

            ghost_gidx_t lfEnt = grpt[0];
            ghost_lidx_t lnEnts = (ghost_lidx_t)grpt[mat->context->lnrows[me]]-grpt[0]; 

            MPI_CALL_GOTO(MPI_Allgather(&lfEnt,1,ghost_mpi_dt_gidx,mat->context->lfEnt,1,ghost_mpi_dt_gidx,mat->context->mpicomm),err,ret);
            MPI_CALL_GOTO(MPI_Allgather(&lnEnts,1,ghost_mpi_dt_lidx,mat->context->lnEnts,1,ghost_mpi_dt_lidx,mat->context->mpicomm),err,ret);

            DEBUG_LOG(2,"frow %"PRGIDX" nrows %"PRLIDX" fent %"PRGIDX" %"PRGIDX" me %"PRGIDX" nents %"PRLIDX" %"PRLIDX" me %"PRLIDX,mat->context->lfRow[me],mat->context->lnrows[me],mat->context->lfEnt[0],mat->context->lfEnt[1],lfEnt,mat->context->lnEnts[0],mat->context->lnEnts[1],lnEnts);


        }
        GHOST_CALL_GOTO(ghost_malloc_align((void **)rpt,(mat->context->lnrows[me]+1)*sizeof(ghost_lidx_t),GHOST_DATA_ALIGNMENT),err,ret);


#pragma omp parallel for schedule(runtime)
    for (i=0;i<mat->context->lnrows[me]+1;i++) {
        (*rpt)[i] = grpt[i] - mat->context->lfEnt[me];
    }

    (*rpt)[mat->context->lnrows[me]] = mat->context->lnEnts[me];

    if( grptAllocated )
      free(grpt);
#else
    GHOST_CALL_GOTO(ghost_malloc_align((void **)rpt,(header.nrows+1) * sizeof(ghost_lidx_t), GHOST_DATA_ALIGNMENT),err,ret);
#pragma omp parallel for schedule(runtime) 
    for (i = 0; i < header.nrows+1; i++) {
        (*rpt)[i] = 0;
    }
    ghost_gidx_t *grpt = (ghost_gidx_t *)rpt;
    GHOST_CALL_GOTO(ghost_bincrs_rpt_read(grpt, matrixPath, 0, header.nrows+1, mat->context->permutation),err,ret);
    for (i=0; i<nprocs; i++){
        mat->context->lfEnt[i] = 0;
        mat->context->lfRow[i] = 0;
    }
    for (i=0; i<nprocs; i++){
        mat->context->lnEnts[i] = header.nnz;
        mat->context->lnrows[i] = header.nrows;
    }
#endif



    DEBUG_LOG(1,"local rows          = %"PRLIDX,mat->context->lnrows[me]);
    DEBUG_LOG(1,"local rows (offset) = %"PRGIDX,mat->context->lfRow[me]);
    DEBUG_LOG(1,"local entries          = %"PRLIDX,mat->context->lnEnts[me]);
    DEBUG_LOG(1,"local entries (offset) = %"PRGIDX,mat->context->lfEnt[me]);

    mat->nrows = mat->context->lnrows[me];
    mat->nnz = mat->context->lnEnts[me];

    goto out;

err:

out:
#ifdef GHOST_HAVE_MPI
    free(stat);
    free(req);
#endif

    return ret;
}

int ghost_cmp_entsperrow(const void* a, const void* b) 
{
    return  ((ghost_sorting_helper_t*)b)->nEntsInRow - ((ghost_sorting_helper_t*)a)->nEntsInRow;
}

ghost_error_t ghost_sparsemat_perm_global_cols(ghost_gidx_t *col, ghost_lidx_t ncols, ghost_context_t *context) 
{
    int me, nprocs,i;
    ghost_rank(&me,context->mpicomm);
    ghost_nrank(&nprocs,context->mpicomm);

    for (i=0; i<nprocs; i++) {
        ghost_lidx_t nels = 0;
        if (i==me) {
            nels = ncols;
        }
        MPI_Bcast(&nels,1,ghost_mpi_dt_gidx,i,context->mpicomm);

        //printf("rank %d has %d elements\n", i,nels);

        ghost_gidx_t *colsfromi;
        ghost_malloc((void **)&colsfromi,nels*sizeof(ghost_gidx_t));
    
        if (i==me) {
            memcpy(colsfromi,col,nels*sizeof(ghost_gidx_t));
        }
        MPI_Bcast(colsfromi,nels,ghost_mpi_dt_gidx,i,context->mpicomm);

        ghost_lidx_t el;
        for (el=0; el<nels; el++) {
            if ((colsfromi[el] >=context->lfRow[me]) && (colsfromi[el] < (context->lfRow[me]+context->lnrows[me]))) {
                //printf("@%d: colsfrom[%d][%d] %d -> %d\n",me,i,el,colsfromi[el],mat->context->permutation->perm[colsfromi[el]-mat->context->lfRow[me]]);
                colsfromi[el] = context->permutation->perm[colsfromi[el]-context->lfRow[me]];
            } else {
                colsfromi[el] = 0;
            }
        }

        if (i==me) {
            MPI_Reduce(MPI_IN_PLACE,colsfromi,nels,ghost_mpi_dt_gidx,MPI_MAX,i,context->mpicomm);
        } else {
            MPI_Reduce(colsfromi,NULL,nels,ghost_mpi_dt_gidx,MPI_MAX,i,context->mpicomm);
        }

        if (i==me) {
            memcpy(col,colsfromi,nels*sizeof(ghost_gidx_t));
        }

        free(colsfromi);
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_perm_sort(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType, ghost_gidx_t scope)
{
    ghost_error_t ret = GHOST_SUCCESS;
    if (mat->context->permutation) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }
    
    int me;    
    ghost_gidx_t i,c,nrows,rowOffset;
    ghost_sorting_helper_t *rowSort = NULL;
    ghost_gidx_t *rpt = NULL;

    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);

    

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation,sizeof(ghost_permutation_t)),err,ret);
    if (mat->traits->sortScope > mat->nrows) {
        nrows = mat->context->gnrows;
        rowOffset = 0;
        mat->context->permutation->scope = GHOST_PERMUTATION_GLOBAL;
    } else {
        nrows = mat->nrows;
        rowOffset = mat->context->lfRow[me];
        mat->context->permutation->scope = GHOST_PERMUTATION_LOCAL;
    }
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation->perm,sizeof(ghost_gidx_t)*nrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation->invPerm,sizeof(ghost_gidx_t)*nrows),err,ret);
#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->context->permutation->cu_perm,sizeof(ghost_gidx_t)*nrows),err,ret);
#endif

    mat->context->permutation->len = nrows;

    memset(mat->context->permutation->perm,0,sizeof(ghost_gidx_t)*nrows);
    memset(mat->context->permutation->invPerm,0,sizeof(ghost_gidx_t)*nrows);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&rowSort,nrows * sizeof(ghost_sorting_helper_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(nrows+1) * sizeof(ghost_gidx_t)),err,ret);

    if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
        ghost_sparsemat_src_rowfunc_t *src = (ghost_sparsemat_src_rowfunc_t *)matrixSource;
        char *tmpval = NULL;
        ghost_gidx_t *tmpcol = NULL;
        rpt[0] = 0;
#pragma omp parallel private(i,tmpval,tmpcol)
        { 
            GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t)),ret);
#pragma omp for schedule(runtime)
            for (i=0; i<nrows; i++) {
                if (src->func(rowOffset+i,&rowSort[i].nEntsInRow,tmpcol,tmpval)) {
                    ERROR_LOG("Matrix construction function returned error");
                    ret = GHOST_ERR_UNKNOWN;
                }
                rowSort[i].row = i;
            }
            free(tmpval);
            free(tmpcol);
        }
        if (ret != GHOST_SUCCESS) {
            goto err;
        }

    } else {
        char *matrixPath = (char *)matrixSource;

        GHOST_CALL_GOTO(ghost_bincrs_rpt_read(rpt, matrixPath, rowOffset, nrows+1, NULL),err,ret);
        for (i=0; i<nrows; i++) {
            rowSort[i].nEntsInRow = rpt[i+1]-rpt[i];
            rowSort[i].row = i;
        }
    }

    for (c=0; c<nrows/scope; c++) {
        qsort(rowSort+c*scope, scope, sizeof(ghost_sorting_helper_t), ghost_cmp_entsperrow);
    }
    qsort(rowSort+c*scope, nrows-c*scope, sizeof(ghost_sorting_helper_t), ghost_cmp_entsperrow);
    
    for(i=0; i < nrows; ++i) {
        (mat->context->permutation->invPerm)[i] = rowSort[i].row;
        (mat->context->permutation->perm)[rowSort[i].row] = i;
    }
#ifdef GHOST_HAVE_CUDA
    ghost_cu_upload(mat->context->permutation->cu_perm,mat->context->permutation->perm,mat->context->permutation->len*sizeof(ghost_gidx_t));
#endif
    
    goto out;

err:
    ERROR_LOG("Deleting permutations");
    if (mat->context->permutation) {
        free(mat->context->permutation->perm); mat->context->permutation->perm = NULL;
        free(mat->context->permutation->invPerm); mat->context->permutation->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
        ghost_cu_free(mat->context->permutation->cu_perm); mat->context->permutation->cu_perm = NULL;
#endif
    }
    free(mat->context->permutation); mat->context->permutation = NULL;

out:

    free(rpt);
    free(rowSort);

    return ret;


}

ghost_error_t ghost_sparsemat_nrows(ghost_gidx_t *nrows, ghost_sparsemat_t *mat)
{
    if (!nrows) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *nrows = mat->context->gnrows;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_nnz(ghost_gidx_t *nnz, ghost_sparsemat_t *mat)
{
    if (!nnz) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_gidx_t lnnz = mat->nnz;

#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Allreduce(&lnnz,nnz,1,ghost_mpi_dt_gidx,MPI_SUM,mat->context->mpicomm));
#else
    *nnz = lnnz;
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_string(char **str, ghost_sparsemat_t *mat)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    int myrank;
    ghost_gidx_t nrows = 0;
    ghost_gidx_t nnz = 0;

    GHOST_CALL_RETURN(ghost_sparsemat_nrows(&nrows,mat));
    GHOST_CALL_RETURN(ghost_sparsemat_nnz(&nnz,mat));
    GHOST_CALL_RETURN(ghost_rank(&myrank, mat->context->mpicomm));


    char *matrixLocation;
    if (mat->traits->flags & GHOST_SPARSEMAT_DEVICE)
        matrixLocation = "Device";
    else if (mat->traits->flags & GHOST_SPARSEMAT_HOST)
        matrixLocation = "Host";
    else
        matrixLocation = "Default";


    ghost_header_string(str,"%s @ rank %d",mat->name,myrank);
    ghost_line_string(str,"Data type",NULL,"%s",ghost_datatype_string(mat->traits->datatype));
    ghost_line_string(str,"Matrix location",NULL,"%s",matrixLocation);
    ghost_line_string(str,"Total number of rows",NULL,"%"PRGIDX,nrows);
    ghost_line_string(str,"Total number of nonzeros",NULL,"%"PRGIDX,nnz);
    ghost_line_string(str,"Avg. nonzeros per row",NULL,"%.3f",(double)nnz/nrows);
    ghost_line_string(str,"Bandwidth",NULL,"%"PRGIDX,mat->bandwidth);
    ghost_line_string(str,"Avg. row band",NULL,"%.3f",mat->avgRowBand);
    ghost_line_string(str,"Avg. avg. row band",NULL,"%.3f",mat->avgAvgRowBand);
    ghost_line_string(str,"Smart row band",NULL,"%.3f",mat->smartRowBand);

    ghost_line_string(str,"Local number of rows",NULL,"%"PRLIDX,mat->nrows);
    ghost_line_string(str,"Local number of rows (padded)",NULL,"%"PRLIDX,mat->nrowsPadded);
    ghost_line_string(str,"Local number of nonzeros",NULL,"%"PRLIDX,mat->nnz);

    ghost_line_string(str,"Full   matrix format",NULL,"%s",mat->formatName(mat));
    if (mat->localPart) {
        ghost_line_string(str,"Local  matrix format",NULL,"%s",mat->localPart->formatName(mat->localPart));
        ghost_line_string(str,"Local  matrix symmetry",NULL,"%s",ghost_sparsemat_symmetry_string(mat->localPart->traits->symmetry));
        ghost_line_string(str,"Local  matrix size","MB","%u",mat->localPart->byteSize(mat->localPart)/(1024*1024));
    }
    if (mat->remotePart) {
        ghost_line_string(str,"Remote matrix format",NULL,"%s",mat->remotePart->formatName(mat->remotePart));
        ghost_line_string(str,"Remote matrix size","MB","%u",mat->remotePart->byteSize(mat->remotePart)/(1024*1024));
    }

    ghost_line_string(str,"Full   matrix size","MB","%u",mat->byteSize(mat)/(1024*1024));
    
    ghost_line_string(str,"Permuted",NULL,"%s",mat->traits->flags&GHOST_SPARSEMAT_PERMUTE?"Yes":"No");
    if ((mat->traits->flags & GHOST_SPARSEMAT_PERMUTE) && mat->context->permutation) {
        if (mat->traits->flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_line_string(str,"Permutation strategy",NULL,"Scotch%s",mat->traits->sortScope>1?"+Sorting":"");
            ghost_line_string(str,"Scotch ordering strategy",NULL,"%s",mat->traits->scotchStrat);
        } else {
            ghost_line_string(str,"Permutation strategy",NULL,"Sorting");
        }
        if (mat->traits->sortScope > 1) {
            ghost_line_string(str,"Sorting scope",NULL,"%d",mat->traits->sortScope);
        }
#ifdef GHOST_HAVE_MPI
        ghost_line_string(str,"Permutation scope",NULL,"%s",mat->context->permutation->scope==GHOST_PERMUTATION_GLOBAL?"Across processes":"Local to process");
#endif
        ghost_line_string(str,"Permuted column indices",NULL,"%s",mat->traits->flags&GHOST_SPARSEMAT_NOT_PERMUTE_COLS?"No":"Yes");
        ghost_line_string(str,"Ascending columns in row",NULL,"%s",mat->traits->flags&GHOST_SPARSEMAT_NOT_SORT_COLS?"No":"Yes");
    }
    ghost_line_string(str,"Max row length (# rows)",NULL,"%d (%d)",mat->maxRowLen,mat->nMaxRows);
    ghost_line_string(str,"Row length variance",NULL,"%f",mat->variance);
    ghost_line_string(str,"Row length standard deviation",NULL,"%f",mat->deviation);
    ghost_line_string(str,"Row length coefficient of variation",NULL,"%f",mat->cv);

    mat->auxString(mat,str);
    ghost_footer_string(str);

    return GHOST_SUCCESS;

}

ghost_error_t ghost_sparsemat_tofile_header(ghost_sparsemat_t *mat, char *path)
{
    ghost_gidx_t mnrows,mncols,mnnz;
    GHOST_CALL_RETURN(ghost_sparsemat_nrows(&mnrows,mat));
    mncols = mnrows;
    GHOST_CALL_RETURN(ghost_sparsemat_nnz(&mnnz,mat));
    
    int32_t endianess = ghost_machine_bigendian();
    int32_t version = 1;
    int32_t base = 0;
    int32_t symmetry = GHOST_BINCRS_SYMM_GENERAL;
    int32_t datatype = mat->traits->datatype;
    int64_t nrows = (int64_t)mnrows;
    int64_t ncols = (int64_t)mncols;
    int64_t nnz = (int64_t)mnnz;

    size_t ret;
    FILE *filed;

    if ((filed = fopen64(path, "w")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s: %s",path,strerror(errno));
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
    fclose(filed);

    return GHOST_SUCCESS;

}

bool ghost_sparsemat_symmetry_valid(ghost_sparsemat_symmetry_t symmetry)
{
    if ((symmetry & (ghost_sparsemat_symmetry_t)GHOST_SPARSEMAT_SYMM_GENERAL) &&
            (symmetry & ~(ghost_sparsemat_symmetry_t)GHOST_SPARSEMAT_SYMM_GENERAL))
        return 0;

    if ((symmetry & (ghost_sparsemat_symmetry_t)GHOST_SPARSEMAT_SYMM_SYMMETRIC) &&
            (symmetry & ~(ghost_sparsemat_symmetry_t)GHOST_SPARSEMAT_SYMM_SYMMETRIC))
        return 0;

    return 1;
}

char * ghost_sparsemat_symmetry_string(ghost_sparsemat_symmetry_t symmetry)
{
    if (symmetry & GHOST_SPARSEMAT_SYMM_GENERAL)
        return "General";

    if (symmetry & GHOST_SPARSEMAT_SYMM_SYMMETRIC)
        return "Symmetric";

    if (symmetry & GHOST_SPARSEMAT_SYMM_SKEW_SYMMETRIC) {
        if (symmetry & GHOST_SPARSEMAT_SYMM_HERMITIAN)
            return "Skew-hermitian";
        else
            return "Skew-symmetric";
    } else {
        if (symmetry & GHOST_SPARSEMAT_SYMM_HERMITIAN)
            return "Hermitian";
    }

    return "Invalid";
}

void ghost_sparsemat_destroy_common(ghost_sparsemat_t *mat)
{
    if (!mat) {
        return;
    }

    if (mat->context->permutation) {
        free(mat->context->permutation->perm); mat->context->permutation->perm = NULL;
        free(mat->context->permutation->invPerm); mat->context->permutation->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
        ghost_cu_free(mat->context->permutation->cu_perm); mat->context->permutation->cu_perm = NULL;
#endif
    }
    free(mat->context->permutation); mat->context->permutation = NULL;
    free(mat->data); mat->data = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
}

ghost_error_t ghost_sparsemat_from_mm(ghost_sparsemat_t *mat, char *path)
{
    PERFWARNING_LOG("The current implementation of Matrix Market read-in is "
            "unefficient!");
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_sparsemat_rowfunc_mm_initargs args;
    ghost_gidx_t dim[2];
    ghost_sparsemat_src_rowfunc_t src = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    
    src.func = &ghost_sparsemat_rowfunc_mm;
    args.filename = path;
    args.dt = mat->traits->datatype;
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_INIT,NULL,dim,&args)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    
    src.maxrowlen = dim[1];
    
    GHOST_CALL_GOTO(mat->fromRowFunc(mat,&src),err,ret);
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_FINALIZE,NULL,NULL,NULL)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;

}
