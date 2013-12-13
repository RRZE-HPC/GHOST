#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_crs.h>
#include <ghost_sell.h>
#include <ghost_mat.h>
#include <ghost_constants.h>
#include <ghost_util.h>

ghost_mat_t *ghost_createMatrix(ghost_context_t *context, ghost_mtraits_t *traits, int nTraits)
{
    ghost_mat_t *mat;
    UNUSED(nTraits);

    switch (traits->format) {
        case GHOST_SPM_FORMAT_CRS:
            mat = ghost_CRS_init(context,traits);
            break;
        case GHOST_SPM_FORMAT_SELL:
            mat = ghost_SELL_init(context,traits);
            break;
        default:
            WARNING_LOG("Invalid sparse matrix format. Falling back to CRS!");
            traits->format = GHOST_SPM_FORMAT_CRS;
            mat = ghost_CRS_init(context,traits);
    }
    return mat;    
}
ghost_mnnz_t ghost_getMatNrows(ghost_mat_t *mat)
{
    ghost_mnnz_t nrows;
    ghost_mnnz_t lnrows = mat->nrows(mat);

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
    ghost_mnnz_t lnnz = mat->nnz(mat);

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
