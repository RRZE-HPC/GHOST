#include "ghost/errorhandler.h"
#include "ghost/func_util.h"
#include <map>

using namespace std;

static map<ghost_error,ghost_errorhandler> errorhandlers;

ghost_errorhandler ghost_errorhandler_get(ghost_error e)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
   
    return errorhandlers[e];
}

ghost_error ghost_errorhandler_set(ghost_error e, ghost_errorhandler h)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    
    errorhandlers[e] = h;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}
