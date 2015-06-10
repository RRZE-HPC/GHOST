#include "ghost/errorhandler.h"
#include <map>

using namespace std;

static map<ghost_error_t,ghost_errorhandler_t> errorhandlers;


ghost_errorhandler_t ghost_errorhandler_get(ghost_error_t e)
{
    return errorhandlers[e];
}

ghost_error_t ghost_errorhandler_set(ghost_error_t e, ghost_errorhandler_t h)
{
    errorhandlers[e] = h;

    return GHOST_SUCCESS;
}
