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

const ghost_sparsemat_traits_t GHOST_SPARSEMAT_TRAITS_INITIALIZER = {.flags = GHOST_SPARSEMAT_STORE_FULL|GHOST_SPARSEMAT_STORE_SPLIT, .aux = NULL, .nAux = 0, .datatype = GHOST_DT_DOUBLE|GHOST_DT_REAL, .format = GHOST_SPARSEMAT_CRS, .shift = NULL, .scale = NULL, .beta = NULL, .symmetry = GHOST_SPARSEMAT_SYMM_GENERAL};

ghost_error_t ghost_createMatrix(ghost_sparsemat_t ** mat, ghost_context_t *context, ghost_sparsemat_traits_t *traits, int nTraits)
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

ghost_error_t ghost_getMatNrows(ghost_midx_t *nrows, ghost_sparsemat_t *mat)
{
    if (!nrows) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_mnnz_t lnrows = mat->nrows;

    if (mat->context->flags & GHOST_CONTEXT_REDUNDANT) {
        *nrows = lnrows;
    } else {
#ifdef GHOST_HAVE_MPI
        MPI_CALL_RETURN(MPI_Allreduce(&lnrows,nrows,1,ghost_mpi_dt_midx,MPI_SUM,mat->context->mpicomm));
#else
        ERROR_LOG("Trying to get the number of matrix rows in a distributed context without MPI");
        return GHOST_ERR_UNKNOWN;
#endif
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_getMatNnz(ghost_mnnz_t *nnz, ghost_sparsemat_t *mat)
{
    if (!nnz) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_mnnz_t lnnz = mat->nnz;

    if (mat->context->flags & GHOST_CONTEXT_REDUNDANT) {
        *nnz = lnnz;
    } else {
#ifdef GHOST_HAVE_MPI
        MPI_CALL_RETURN(MPI_Allreduce(&lnnz,nnz,1,ghost_mpi_dt_mnnz,MPI_SUM,mat->context->mpicomm));
#else
        ERROR_LOG("Trying to get the number of matrix nonzeros in a distributed context without MPI");
        return GHOST_ERR_UNKNOWN;
#endif
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_printMatrixInfo(char **str, ghost_sparsemat_t *mat)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    int myrank;
    ghost_midx_t nrows = 0;
    ghost_midx_t nnz = 0;
    
    GHOST_CALL_RETURN(ghost_getMatNrows(&nrows,mat));
    GHOST_CALL_RETURN(ghost_getMatNnz(&nnz,mat));
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
    ghost_printLine(str,"Number of rows",NULL,"%"PRmatIDX,nrows);
    ghost_printLine(str,"Number of nonzeros",NULL,"%"PRmatNNZ,nnz);
    ghost_printLine(str,"Avg. nonzeros per row",NULL,"%.3f",(double)nnz/nrows);

    ghost_printLine(str,"Full   matrix format",NULL,"%s",mat->formatName(mat));
    if (mat->context->flags & GHOST_CONTEXT_DISTRIBUTED)
    {
        if (mat->localPart) {
            ghost_printLine(str,"Local  matrix format",NULL,"%s",mat->localPart->formatName(mat->localPart));
            ghost_printLine(str,"Local  matrix symmetry",NULL,"%s",ghost_symmetryName(mat->localPart->traits->symmetry));
            ghost_printLine(str,"Local  matrix size","MB","%u",mat->localPart->byteSize(mat->localPart)/(1024*1024));
        }
        if (mat->remotePart) {
            ghost_printLine(str,"Remote matrix format",NULL,"%s",mat->remotePart->formatName(mat->remotePart));
            ghost_printLine(str,"Remote matrix size","MB","%u",mat->remotePart->byteSize(mat->remotePart)/(1024*1024));
        }
    } else {
        ghost_printLine(str,"Full   matrix symmetry",NULL,"%s",ghost_symmetryName(mat->traits->symmetry));
    }

    ghost_printLine(str,"Full   matrix size","MB","%u",mat->byteSize(mat)/(1024*1024));

    mat->printInfo(str,mat);
    ghost_printFooter(str);

    return GHOST_SUCCESS;

}
