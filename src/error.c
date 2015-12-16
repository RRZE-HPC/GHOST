#include "ghost/error.h"
#include "ghost/func_util.h"

char * ghost_error_string(ghost_error_t e)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    char *ret;
    switch (e) {
        case GHOST_ERR_INVALID_ARG:
            ret = "Invalid argument";
            break;
        case GHOST_ERR_MPI:
            ret = "Error in MPI";
            break;
        case GHOST_ERR_CUDA:
            ret = "Error in CUDA";
            break;
        case GHOST_ERR_CUBLAS:
            ret = "Error in cuBLAS";
            break;
        case GHOST_ERR_CURAND:
            ret = "Error in cuRAND";
            break;
        case GHOST_ERR_HWLOC:
            ret = "Error in hwloc";
            break;
        case GHOST_ERR_SCOTCH:
            ret = "Error in Scotch";
            break;
        case GHOST_ERR_NOT_IMPLEMENTED:
            ret = "Not implemented";
            break;
        case GHOST_ERR_IO:
            ret = "I/O error";
            break;
        case GHOST_ERR_DATATYPE:
            ret = "Error with data types";
            break;
        case GHOST_ERR_COLPACK:
            ret = "Error in ColPack";
            break;
        case GHOST_ERR_LAPACK:
            ret = "Error in LAPACK";
            break;
        case GHOST_ERR_BLAS:
            ret = "Error in BLAS";
            break;
        case GHOST_ERR_UNKNOWN:
            ret = "Unknown error";
            break;
        default:
            ret = "Invalid";
            break;
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return ret;
}
