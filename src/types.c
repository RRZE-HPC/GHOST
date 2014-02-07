#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"

static ghost_mpi_datatype_t GHOST_MPI_DT_C = MPI_DATATYPE_NULL;
static ghost_mpi_datatype_t GHOST_MPI_DT_Z = MPI_DATATYPE_NULL;

ghost_error_t ghost_mpi_datatype(ghost_mpi_datatype_t *dt, int datatype)
{
    if (!dt) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
#if GHOST_HAVE_MPI
    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_COMPLEX)
            *dt = GHOST_MPI_DT_C;
        else
            *dt = MPI_FLOAT;
    } else {
        if (datatype & GHOST_DT_COMPLEX)
            *dt = GHOST_MPI_DT_Z;
        else
            *dt = MPI_DOUBLE;
    }
#else
    UNUSED(datatype);
    *dt = MPI_DATATYPE_NULL;
#endif

    return GHOST_SUCCESS;
}


ghost_error_t ghost_mpi_createDatatypes()
{
#if GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Type_contiguous(2,MPI_FLOAT,&GHOST_MPI_DT_C));
    MPI_CALL_RETURN(MPI_Type_commit(&GHOST_MPI_DT_C));

    MPI_CALL_RETURN(MPI_Type_contiguous(2,MPI_DOUBLE,&GHOST_MPI_DT_Z));
    MPI_CALL_RETURN(MPI_Type_commit(&GHOST_MPI_DT_Z));
#else
    UNUSED(GHOST_MPI_DT_C);
    UNUSED(GHOST_MPI_DT_Z);
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_mpi_destroyDatatypes()
{
#if GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Type_free(&GHOST_MPI_DT_C));
    MPI_CALL_RETURN(MPI_Type_free(&GHOST_MPI_DT_Z));
#endif

    return GHOST_SUCCESS;
}
