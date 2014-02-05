#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/crs.h"
#include "ghost/sell.h"
#include "ghost/mat.h"
#include "ghost/constants.h"
#include "ghost/context.h"
#include "ghost/util.h"
#include "ghost/log.h"

const ghost_mtraits_t GHOST_MTRAITS_INITIALIZER = {.flags = GHOST_SPM_DEFAULT, .aux = NULL, .nAux = 0, .datatype = GHOST_BINCRS_DT_DOUBLE|GHOST_BINCRS_DT_REAL, .format = GHOST_SPM_FORMAT_CRS, .shift = NULL, .scale = NULL };

ghost_error_t ghost_createMatrix(ghost_context_t *context, ghost_mtraits_t *traits, int nTraits, ghost_mat_t ** mat)
{
    UNUSED(nTraits);

    switch (traits->format) {
        case GHOST_SPM_FORMAT_CRS:
            return ghost_CRS_init(context,traits,mat);
        case GHOST_SPM_FORMAT_SELL:
            return ghost_SELL_init(context,traits,mat);
        default:
            WARNING_LOG("Invalid sparse matrix format. Falling back to CRS!");
            traits->format = GHOST_SPM_FORMAT_CRS;
            return ghost_CRS_init(context,traits,mat);
    }
    return GHOST_SUCCESS;    
}

ghost_mnnz_t ghost_getMatNrows(ghost_mat_t *mat)
{
    ghost_mnnz_t nrows;
    ghost_mnnz_t lnrows = mat->nrows;

    if (mat->context->flags & GHOST_CONTEXT_GLOBAL) {
        nrows = lnrows;
    } else {
#ifdef GHOST_HAVE_MPI
        MPI_safecall(MPI_Allreduce(&lnrows,&nrows,1,ghost_mpi_dt_midx,MPI_SUM,mat->context->mpicomm));
#else
        ABORT("Trying to get the number of matrix rows in a distributed context without MPI");
#endif
    }

    return nrows;
}

ghost_mnnz_t ghost_getMatNnz(ghost_mat_t *mat)
{
    ghost_mnnz_t nnz;
    ghost_mnnz_t lnnz = mat->nnz;

    if (mat->context->flags & GHOST_CONTEXT_GLOBAL) {
        nnz = lnnz;
    } else {
#ifdef GHOST_HAVE_MPI
        MPI_safecall(MPI_Allreduce(&lnnz,&nnz,1,ghost_mpi_dt_mnnz,MPI_SUM,mat->context->mpicomm));
#else
        ABORT("Trying to get the number of matrix nonzeros in a distributed context without MPI");
#endif
    }

    return nnz;
}

