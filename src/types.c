#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"

static ghost_mpi_datatype_t GHOST_MPI_DT_C = MPI_DATATYPE_NULL;
static ghost_mpi_datatype_t GHOST_MPI_DT_Z = MPI_DATATYPE_NULL;

ghost_error_t ghost_mpi_datatype(ghost_mpi_datatype_t *dt, ghost_datatype_t datatype)
{
    if (!dt) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
#ifdef GHOST_HAVE_MPI
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


ghost_error_t ghost_mpi_datatypes_create()
{
#ifdef GHOST_HAVE_MPI
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

ghost_error_t ghost_mpi_datatypes_destroy()
{
#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Type_free(&GHOST_MPI_DT_C));
    MPI_CALL_RETURN(MPI_Type_free(&GHOST_MPI_DT_Z));
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_datatype_size(size_t *size, ghost_datatype_t datatype)
{
    if (!ghost_datatype_valid(datatype)) {
        ERROR_LOG("Invalid data type %d",datatype);
        return GHOST_ERR_INVALID_ARG;
    }

    *size = 0;

    if (datatype & GHOST_DT_FLOAT) {
        *size = sizeof(float);
    } else {
        *size = sizeof(double);
    }

    if (datatype & GHOST_DT_COMPLEX) {
        *size *= 2;
    }

    return GHOST_SUCCESS;
}

bool ghost_datatype_valid(ghost_datatype_t datatype)
{
    if ((datatype & GHOST_DT_FLOAT) &&
            (datatype & GHOST_DT_DOUBLE))
        return 0;

    if (!(datatype & GHOST_DT_FLOAT) &&
            !(datatype & GHOST_DT_DOUBLE))
        return 0;

    if ((datatype & GHOST_DT_REAL) &&
            (datatype & GHOST_DT_COMPLEX))
        return 0;

    if (!(datatype & GHOST_DT_REAL) &&
            !(datatype & GHOST_DT_COMPLEX))
        return 0;

    return 1;
}

char * ghost_datatype_string(ghost_datatype_t datatype)
{
    if (!ghost_datatype_valid(datatype)) {
        return "Invalid";
    }

    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_REAL)
            return "Float";
        else
            return "Complex float";
    } else {
        if (datatype & GHOST_DT_REAL)
            return "Double";
        else
            return "Complex double";
    }

    return "Invalid";
}

ghost_error_t ghost_datatype_idx(ghost_datatype_idx_t *idx, ghost_datatype_t datatype)
{
    if (!ghost_datatype_valid(datatype)) {
        ERROR_LOG("Invalid data type");
        return GHOST_ERR_INVALID_ARG;
    }

    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_COMPLEX) {
            *idx = GHOST_DT_C_IDX;
        } else {
            *idx = GHOST_DT_S_IDX;
        }
    } else {
        if (datatype & GHOST_DT_COMPLEX) {
            *idx = GHOST_DT_Z_IDX;
        } else {
            *idx = GHOST_DT_D_IDX;
        }
    }

    return GHOST_SUCCESS;
}