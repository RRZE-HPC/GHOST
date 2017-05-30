#define _XOPEN_SOURCE 500 
#include "ghost/densemat.h"
#include "ghost/densemat_cm.h"
#include "ghost/densemat_rm.h"
#include "ghost/util.h"

#include <string>
#include <sstream>

#define PASTER(x,y) x ## _ ## y
#define EVALUATOR(x,y) PASTER(x,y)
#define CM_FUNCNAME(fun) EVALUATOR(ghost_densemat_cm,fun)
#define RM_FUNCNAME(fun) EVALUATOR(ghost_densemat_rm,fun)

#define CALL_DENSEMAT_FUNC_NORET(ret,vec,func,...) \
    if (vec->traits.storage == GHOST_DENSEMAT_COLMAJOR) {\
        ret = CM_FUNCNAME(func)(__VA_ARGS__);\
    } else {\
        ret = RM_FUNCNAME(func)(__VA_ARGS__);\
    }

ghost_error ghost_densemat_string(char **str, ghost_densemat *x)
{
    ghost_error ret = GHOST_SUCCESS;
    std::string finalstr;

    for (ghost_lidx s=0; s<x->nsub; s++) {
        char *substr;
        CALL_DENSEMAT_FUNC_NORET(ret,x->subdm[s],string_selector,x->subdm[s],&substr);

        std::istringstream subiss(substr);
        std::istringstream finaliss(finalstr);
        finalstr = "";
        for (std::string subline; std::getline(subiss,subline); ) {
            std::string finalline;
            std::getline(finaliss,finalline);
            finalstr += finalline + (finalline.length()?"\t":"") + subline + "\n";
        }
        free(substr);
    }
    
    GHOST_CALL_RETURN(ghost_malloc((void **)str,finalstr.length()+1));
    strcpy(*str,finalstr.c_str());

    return ret;
}

