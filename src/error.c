#include "ghost/error.h"

char * ghost_errorString(ghost_error_t e)
{
    switch (e) {
        case GHOST_ERR_INVALID_ARG:
            return "Invalid argument";
            break;
        case GHOST_ERR_IO:
            return "I/O error";
            break;
        default:
            return "Unknown error";
            break;
    }

}
