#include "ghost/errorhandler.h"
#include "ghost/func_util.h"
#include <map>

using namespace std;

static map<ghost_error_t,ghost_errorhandler_t> errorhandlers;

ghost_errorhandler_t ghost_errorhandler_get(ghost_error_t e)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
   
    return errorhandlers[e];
}

ghost_error_t ghost_errorhandler_set(ghost_error_t e, ghost_errorhandler_t h)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    
    errorhandlers[e] = h;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}
