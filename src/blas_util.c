#include "ghost/blas_util.h"
#include "ghost/log.h"
#include "ghost/util.h"

static int ghost_blas_err = 0;

#ifdef GHOST_HAVE_MKL
void cblas_xerbla(const char *name, const int num)
{
    GHOST_ERROR_LOG("Error in BLAS call %s in parameter %d",name,num);
    ghost_blas_err = 1;
}
#endif

#ifdef GHOST_HAVE_GSL
void cblas_xerbla (int p, const char *rout, const char *form, ...)
{
    GHOST_ERROR_LOG("Error in BLAS call %s in parameter %d",rout,p);
    ghost_blas_err = 1;
    UNUSED(form);
}
#endif



int ghost_blas_err_pop()
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    
    int ret = ghost_blas_err;
    ghost_blas_err = 0;
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return ret;
}

