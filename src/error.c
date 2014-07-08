#include "ghost/error.h"

char * ghost_error_string(ghost_error_t e)
{
    switch (e) {
        case GHOST_ERR_INVALID_ARG:
            return "Invalid argument";
            break;
        case GHOST_ERR_MPI:
            return "Error in MPI";
            break;
        case GHOST_ERR_CUDA:
            return "Error in CUDA";
            break;
        case GHOST_ERR_CUBLAS:
            return "Error in cuBLAS";
            break;
        case GHOST_ERR_CURAND:
            return "Error in cuRAND";
            break;
        case GHOST_ERR_HWLOC:
            return "Error in hwloc";
            break;
        case GHOST_ERR_SCOTCH:
            return "Error in Scotch";
            break;
        case GHOST_ERR_NOT_IMPLEMENTED:
            return "Not implemented";
            break;
        case GHOST_ERR_IO:
            return "I/O error";
            break;
        case GHOST_ERR_DATATYPE:
            return "Error with data types";
            break;
        case GHOST_ERR_UNKNOWN:
            return "Unknown error";
            break;
        default:
            return "Invalid";
            break;
    }

}
