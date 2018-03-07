#include <iostream>
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/sparsemat.h"

#include "ghost/tsmtspmtsm_var2_cuda.h"

using namespace std;

ghost_error tsmtspmtsm_var2_cuda(ghost_densemat *v, ghost_densemat *w, ghost_densemat *x,
    ghost_sparsemat *A, void *pAlpha, void *pBeta)
{
    cout << "tsmtspmtsm_var2_cuda reporting\n";
    return GHOST_SUCCESS;
}
