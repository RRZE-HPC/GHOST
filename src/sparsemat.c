#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/crs.h"
#include "ghost/sell.h"
#include "ghost/sparsemat.h"
#include "ghost/constants.h"
#include "ghost/context.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#include "ghost/io.h"

const ghost_sparsemat_traits_t GHOST_SPARSEMAT_TRAITS_INITIALIZER = {.flags = GHOST_SPARSEMAT_STORE_FULL|GHOST_SPARSEMAT_STORE_SPLIT, .aux = NULL, .nAux = 0, .datatype = GHOST_DT_DOUBLE|GHOST_DT_REAL, .format = GHOST_SPARSEMAT_CRS, .shift = NULL, .scale = NULL, .beta = NULL, .symmetry = GHOST_SPARSEMAT_SYMM_GENERAL};

ghost_error_t ghost_sparsemat_create(ghost_sparsemat_t ** mat, ghost_context_t *context, ghost_sparsemat_traits_t *traits, int nTraits)
{
    UNUSED(nTraits);
    ghost_error_t ret = GHOST_SUCCESS;

    int me;
    GHOST_CALL_GOTO(ghost_getRank(context->mpicomm,&me),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)mat,sizeof(ghost_sparsemat_t)),err,ret);
    
    (*mat)->traits = traits;
    (*mat)->context = context;
    (*mat)->localPart = NULL;
    (*mat)->remotePart = NULL;
    (*mat)->name = "Sparse matrix";
    (*mat)->data = NULL;
    (*mat)->nzDist = NULL;
    (*mat)->fromFile = NULL;
    (*mat)->toFile = NULL;
    (*mat)->fromRowFunc = NULL;
    (*mat)->fromCRS = NULL;
    (*mat)->printInfo = NULL;
    (*mat)->formatName = NULL;
    (*mat)->rowLen = NULL;
    (*mat)->byteSize = NULL;
    (*mat)->permute = NULL;
    (*mat)->destroy = NULL;
    (*mat)->stringify = NULL;
    (*mat)->upload = NULL;
    (*mat)->permute = NULL;
    (*mat)->spmv = NULL;
    (*mat)->destroy = NULL;
    (*mat)->split = NULL;
    (*mat)->bandwidth = 0;
    (*mat)->lowerBandwidth = 0;
    (*mat)->upperBandwidth = 0;
    (*mat)->nrows = context->lnrows[me];
    (*mat)->ncols = context->gncols;
    (*mat)->nrowsPadded = 0;
    (*mat)->nEnts = 0;
    (*mat)->nnz = 0;
   
    GHOST_CALL_GOTO(ghost_malloc((void **)&((*mat)->nzDist),sizeof(ghost_nnz_t)*(2*context->gnrows-1)),err,ret);
    GHOST_CALL_GOTO(ghost_sizeofDatatype(&(*mat)->traits->elSize,(*mat)->traits->datatype),err,ret);
    
    switch (traits->format) {
        case GHOST_SPARSEMAT_CRS:
            GHOST_CALL_GOTO(ghost_CRS_init(*mat),err,ret);
            break;
        case GHOST_SPARSEMAT_SELL:
            GHOST_CALL_GOTO(ghost_SELL_init(*mat),err,ret);
            break;
        default:
            WARNING_LOG("Invalid sparse matrix format. Falling back to CRS!");
            traits->format = GHOST_SPARSEMAT_CRS;
            GHOST_CALL_GOTO(ghost_CRS_init(*mat),err,ret);
    }

    goto out;
err:
    ERROR_LOG("Error. Free'ing resources");
    free(*mat); *mat = NULL;

out:
    return ret;    
}

ghost_error_t ghost_sparsemat_sortRow(ghost_idx_t *col, char *val, size_t valSize, ghost_idx_t rowlen, ghost_idx_t stride)
{
    ghost_idx_t n;
    ghost_idx_t c;
    ghost_idx_t swpcol;
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

ghost_error_t ghost_sparsemat_createPermutation(ghost_sparsemat_t *mat, char *matrixPath)
{
#if 0
    INFO_LOG("Reducing matrix bandwidth");
    ghost_error_t ret = GHOST_SUCCESS;
    if (mat->context->rowPerm || mat->context->invRowPerm) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }

    ghost_idx_t *rpt = NULL, *col = NULL, i;
    int me, nprocs;
    
    ghost_matfile_header_t header;
    ghost_readMatFileHeader(matrixPath,&header);
    MPI_Request *req = NULL;
    MPI_Status *stat = NULL;

    GHOST_CALL_GOTO(ghost_getRank(mat->context->mpicomm,&me),err,ret);
    GHOST_CALL_GOTO(ghost_getNumberOfRanks(mat->context->mpicomm,&nprocs),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&req,sizeof(MPI_Request)*nprocs),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&stat,sizeof(MPI_Status)*nprocs),err,ret);
        
    if (me == 0) {
        if (mat->context->flags & GHOST_CONTEXT_DIST_NZ) { // rpt has already been read
            rpt = mat->context->rpt;
        } else {
            GHOST_CALL_GOTO(ghost_malloc_align((void **)&rpt,(mat->context->gnrows+1) * sizeof(ghost_nnz_t), GHOST_DATA_ALIGNMENT),err,ret);
#pragma omp parallel for schedule(runtime) 
            for (i = 0; i < mat->context->gnrows+1; i++) {
                rpt[i] = 0;
            }
            GHOST_CALL_GOTO(ghost_readRpt(rpt, matrixPath, 0, header.nrows+1),err,ret);
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
    MPI_CALL_GOTO(MPI_Bcast(mat->context->lfEnt,  nprocs, ghost_mpi_dt_idx, 0, mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Bcast(mat->context->lnEnts, nprocs, ghost_mpi_dt_idx, 0, mat->context->mpicomm),err,ret);
        
    mat->nnz = mat->context->lnEnts[me];
    
    if (me != 0) {
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&rpt,(mat->context->lnrows[me]+1)*sizeof(ghost_idx_t),GHOST_DATA_ALIGNMENT),err,ret);
#pragma omp parallel for schedule(runtime)
        for (i = 0; i < mat->context->lnrows[me]+1; i++) {
            rpt[i] = 0;
        }
    }
    int msgcount = 0;

    for (i=0;i<nprocs;i++) 
        req[i] = MPI_REQUEST_NULL;

    if (me != 0) {
        MPI_CALL_GOTO(MPI_Irecv(rpt,mat->context->lnrows[me]+1,ghost_mpi_dt_idx,0,me,mat->context->mpicomm,&req[msgcount]),err,ret);
        msgcount++;
    } else {
        for (i=1;i<nprocs;i++) {
            MPI_CALL_GOTO(MPI_Isend(&rpt[mat->context->lfRow[i]],mat->context->lnrows[i]+1,ghost_mpi_dt_idx,i,i,mat->context->mpicomm,&req[msgcount]),err,ret);
            msgcount++;
        }
    }
    MPI_CALL_GOTO(MPI_Waitall(msgcount,req,stat),err,ret);

    DEBUG_LOG(1,"Adjusting row pointers");
    for (i=0;i<mat->context->lnrows[me]+1;i++) {
        rpt[i] -= mat->context->lfEnt[me]; 
    }

    rpt[mat->context->lnrows[me]] = mat->context->lnEnts[me];
        
    GHOST_CALL_GOTO(ghost_malloc_align((void **)&col,mat->nnz * sizeof(ghost_idx_t), GHOST_DATA_ALIGNMENT),err,ret);
    GHOST_CALL_GOTO(ghost_readCol(col, matrixPath, mat->context->lfEnt[me], mat->nnz),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->rowPerm,sizeof(ghost_idx_t)*mat->context->gnrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->invRowPerm,sizeof(ghost_idx_t)*mat->context->gnrows),err,ret);
    memset(mat->context->rowPerm,0,sizeof(ghost_idx_t)*mat->context->gnrows);
    memset(mat->context->rowPerm,0,sizeof(ghost_idx_t)*mat->context->gnrows);
    
    SCOTCH_Dgraph * dgraph = SCOTCH_dgraphAlloc();
    if (!dgraph) {
        ERROR_LOG("Could not alloc SCOTCH graph");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_dgraphInit(dgraph,mat->context->mpicomm),err,ret);
    SCOTCH_Strat * strat = SCOTCH_stratAlloc();
    if (!strat) {
        ERROR_LOG("Could not alloc SCOTCH strat");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_stratInit(strat),err,ret);
    SCOTCH_Dordering *dorder = SCOTCH_dorderAlloc();
    if (!dorder) {
        ERROR_LOG("Could not alloc SCOTCH order");
        ret = GHOST_ERR_SCOTCH;
        goto err;
    }
    SCOTCH_CALL_GOTO(SCOTCH_dgraphBuild(dgraph, 0, mat->nrows, mat->nrows, rpt, rpt+1, NULL, NULL, mat->nnz, mat->nnz, col, NULL, NULL),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphCheck(dgraph),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderInit(dgraph,dorder),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_stratDgraphOrder(strat,"n{sep=m{asc=b,low=b},ole=q{strat=g},ose=q{strat=g},osq=g}"),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderCompute(dgraph,dorder,strat),err,ret);
    SCOTCH_CALL_GOTO(SCOTCH_dgraphOrderPerm(dgraph,dorder,mat->context->rowPerm+mat->context->lfRow[me]),err,ret);

    // combine permutation vectors
    MPI_CALL_GOTO(MPI_Allreduce(MPI_IN_PLACE,mat->context->rowPerm,mat->context->gnrows,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm),err,ret);

    // assemble inverse permutation
    for (i=0; i<mat->context->gnrows; i++) {
        mat->context->invRowPerm[mat->context->rowPerm[i]] = i;
    }
    
 //   for (i=0; i<mat->context->gnrows; i++) {
 //       INFO_LOG("perm[%d] = %d",i,mat->context->rowPerm[i]);
 //   }

    goto out;
err:

out:
    return ret;
#endif
    ERROR_LOG("Not implemented");
    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_nrows(ghost_idx_t *nrows, ghost_sparsemat_t *mat)
{
    if (!nrows) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_nnz_t lnrows = mat->nrows;

    if (mat->context->flags & GHOST_CONTEXT_REDUNDANT) {
        *nrows = lnrows;
    } else {
#ifdef GHOST_HAVE_MPI
        MPI_CALL_RETURN(MPI_Allreduce(&lnrows,nrows,1,ghost_mpi_dt_idx,MPI_SUM,mat->context->mpicomm));
#else
        ERROR_LOG("Trying to get the number of matrix rows in a distributed context without MPI");
        return GHOST_ERR_UNKNOWN;
#endif
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_nnz(ghost_nnz_t *nnz, ghost_sparsemat_t *mat)
{
    if (!nnz) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_nnz_t lnnz = mat->nnz;

    if (mat->context->flags & GHOST_CONTEXT_REDUNDANT) {
        *nnz = lnnz;
    } else {
#ifdef GHOST_HAVE_MPI
        MPI_CALL_RETURN(MPI_Allreduce(&lnnz,nnz,1,ghost_mpi_dt_nnz,MPI_SUM,mat->context->mpicomm));
#else
        ERROR_LOG("Trying to get the number of matrix nonzeros in a distributed context without MPI");
        return GHOST_ERR_UNKNOWN;
#endif
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_string(char **str, ghost_sparsemat_t *mat)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    int myrank;
    ghost_idx_t nrows = 0;
    ghost_idx_t nnz = 0;
    
    GHOST_CALL_RETURN(ghost_sparsemat_nrows(&nrows,mat));
    GHOST_CALL_RETURN(ghost_sparsemat_nnz(&nnz,mat));
    GHOST_CALL_RETURN(ghost_getRank(mat->context->mpicomm,&myrank));


    char *matrixLocation;
    if (mat->traits->flags & GHOST_SPARSEMAT_DEVICE)
        matrixLocation = "Device";
    else if (mat->traits->flags & GHOST_SPARSEMAT_HOST)
        matrixLocation = "Host";
    else
        matrixLocation = "Default";


    ghost_printHeader(str,"%s @ rank %d",mat->name,myrank);
    ghost_printLine(str,"Data type",NULL,"%s",ghost_datatypeString(mat->traits->datatype));
    ghost_printLine(str,"Matrix location",NULL,"%s",matrixLocation);
    ghost_printLine(str,"Total number of rows",NULL,"%"PRIDX,nrows);
    ghost_printLine(str,"Total number of nonzeros",NULL,"%"PRNNZ,nnz);
    ghost_printLine(str,"Avg. nonzeros per row",NULL,"%.3f",(double)nnz/nrows);
    ghost_printLine(str,"Bandwidth",NULL,"%"PRIDX,mat->bandwidth);
    
    ghost_printLine(str,"Local number of rows",NULL,"%"PRIDX,mat->nrows);
    ghost_printLine(str,"Local number of rows (padded)",NULL,"%"PRIDX,mat->nrowsPadded);
    ghost_printLine(str,"Local number of nonzeros",NULL,"%"PRIDX,mat->nnz);

    ghost_printLine(str,"Full   matrix format",NULL,"%s",mat->formatName(mat));
    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED)
    {
        if (mat->localPart) {
            ghost_printLine(str,"Local  matrix format",NULL,"%s",mat->localPart->formatName(mat->localPart));
            ghost_printLine(str,"Local  matrix symmetry",NULL,"%s",ghost_sparsemat_symmetryString(mat->localPart->traits->symmetry));
            ghost_printLine(str,"Local  matrix size","MB","%u",mat->localPart->byteSize(mat->localPart)/(1024*1024));
        }
        if (mat->remotePart) {
            ghost_printLine(str,"Remote matrix format",NULL,"%s",mat->remotePart->formatName(mat->remotePart));
            ghost_printLine(str,"Remote matrix size","MB","%u",mat->remotePart->byteSize(mat->remotePart)/(1024*1024));
        }
    } else {
        ghost_printLine(str,"Full   matrix symmetry",NULL,"%s",ghost_sparsemat_symmetryString(mat->traits->symmetry));
    }

    ghost_printLine(str,"Full   matrix size","MB","%u",mat->byteSize(mat)/(1024*1024));

    mat->printInfo(str,mat);
    ghost_printFooter(str);

    return GHOST_SUCCESS;

}

char ghost_sparsemat_symmetryValid(int symmetry)
{
    if ((symmetry & GHOST_SPARSEMAT_SYMM_GENERAL) &&
            (symmetry & ~GHOST_SPARSEMAT_SYMM_GENERAL))
        return 0;

    if ((symmetry & GHOST_SPARSEMAT_SYMM_SYMMETRIC) &&
            (symmetry & ~GHOST_SPARSEMAT_SYMM_SYMMETRIC))
        return 0;

    return 1;
}

char * ghost_sparsemat_symmetryString(int symmetry)
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



