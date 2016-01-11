#include "ghost/densemat_rm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include <complex>

template<typename T>
static ghost_error_t ghost_densemat_rm_averagehalo_tmpl(ghost_densemat_t *vec)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    
    UNUSED(vec);
    ERROR_LOG("Not implemented!");
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return GHOST_ERR_NOT_IMPLEMENTED;
}

ghost_error_t ghost_densemat_rm_averagehalo_selector(ghost_densemat_t *vec)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,std::complex,ret,ghost_densemat_rm_averagehalo_tmpl,vec);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
}
