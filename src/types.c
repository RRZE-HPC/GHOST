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
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
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
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
#else
    UNUSED(datatype);
    *dt = MPI_DATATYPE_NULL;
#endif

    return GHOST_SUCCESS;
}


ghost_error_t ghost_mpi_datatypes_create()
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_SETUP);
    MPI_CALL_RETURN(MPI_Type_contiguous(2,MPI_FLOAT,&GHOST_MPI_DT_C));
    MPI_CALL_RETURN(MPI_Type_commit(&GHOST_MPI_DT_C));

    MPI_CALL_RETURN(MPI_Type_contiguous(2,MPI_DOUBLE,&GHOST_MPI_DT_Z));
    MPI_CALL_RETURN(MPI_Type_commit(&GHOST_MPI_DT_Z));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_SETUP);
#else
    UNUSED(GHOST_MPI_DT_C);
    UNUSED(GHOST_MPI_DT_Z);
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_mpi_datatypes_destroy()
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TEARDOWN);
    MPI_CALL_RETURN(MPI_Type_free(&GHOST_MPI_DT_C));
    MPI_CALL_RETURN(MPI_Type_free(&GHOST_MPI_DT_Z));
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_TEARDOWN);
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_datatype_size(size_t *size, ghost_datatype_t datatype)
{
    if (!ghost_datatype_valid(datatype)) {
        ERROR_LOG("Invalid data type %d",(int)datatype);
        return GHOST_ERR_INVALID_ARG;
    }

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    *size = 0;

    if (datatype & GHOST_DT_FLOAT) {
        *size = sizeof(float);
    } else {
        *size = sizeof(double);
    }

    if (datatype & GHOST_DT_COMPLEX) {
        *size *= 2;
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

bool ghost_datatype_valid(ghost_datatype_t datatype)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    if (datatype == GHOST_DT_ANY) {
        return 1;
    }

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

const char * ghost_datatype_string(ghost_datatype_t datatype)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    if (!ghost_datatype_valid(datatype)) {
        return "Invalid";
    }

    if (datatype == GHOST_DT_ANY) {
        return "any";
    }

    if (datatype & GHOST_DT_FLOAT) {
        if (datatype & GHOST_DT_REAL) {
            return "Float";
        } else {
            return "Complex float";
        }
    } else {
        if (datatype & GHOST_DT_REAL) {
            return "Double";
        } else {
            return "Complex double";
        }
    }
}

ghost_error_t ghost_datatype_idx(ghost_datatype_idx_t *idx, ghost_datatype_t datatype)
{
    if (!ghost_datatype_valid(datatype)) {
        ERROR_LOG("Invalid data type");
        return GHOST_ERR_INVALID_ARG;
    }
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);

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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error_t ghost_idx2datatype(ghost_datatype_t *datatype, ghost_datatype_idx_t idx)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
   
    switch(idx) {
        case (ghost_datatype_idx_t)0: 
            *datatype = (ghost_datatype_t)(GHOST_DT_REAL|GHOST_DT_FLOAT);
            break;
        case (ghost_datatype_idx_t)1: 
            *datatype = (ghost_datatype_t)(GHOST_DT_REAL|GHOST_DT_DOUBLE);
            break;
        case (ghost_datatype_idx_t)2: 
            *datatype = (ghost_datatype_t)(GHOST_DT_COMPLEX|GHOST_DT_FLOAT);
            break;
        case (ghost_datatype_idx_t)3: 
            *datatype = (ghost_datatype_t)(GHOST_DT_COMPLEX|GHOST_DT_DOUBLE);
            break;
        default:
            ERROR_LOG("Invalid datatype index!");
            return GHOST_ERR_INVALID_ARG;
    }
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}
   
const char * ghost_location_string(ghost_location_t location)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    
    if (location & GHOST_LOCATION_HOST) {
        if (location & GHOST_LOCATION_DEVICE) {
            return "Host&Device";
        } else {
            return "Host";
        }
    } else if (location & GHOST_LOCATION_DEVICE) {
        return "Device";
    } else {
        return "Invalid";
    }
}
    
const char * ghost_implementation_string(ghost_implementation_t implementation)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    
    switch(implementation) {
        case GHOST_IMPLEMENTATION_PLAIN:
            return "vanilla";
        case GHOST_IMPLEMENTATION_SSE:
            return "SSE";
        case GHOST_IMPLEMENTATION_AVX:
            return "AVX";
        case GHOST_IMPLEMENTATION_AVX2:
            return "AVX2";
        case GHOST_IMPLEMENTATION_MIC:
            return "MIC";
        case GHOST_IMPLEMENTATION_CUDA:
            return "CUDA";
        default:
            return "unknown";
    }
}
