#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmm_inplace.h"
#include "ghost/tsmm_inplace_gen.h"

#include <map>

using namespace std;

    
bool operator<(const ghost_tsmm_inplace_parameters_t &a, const ghost_tsmm_inplace_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz,0) < ghost_hash(b.dt,b.blocksz,0); 
}


static map<ghost_tsmm_inplace_parameters_t, ghost_tsmm_inplace_kernel_t> ghost_tsmm_inplace_kernels;

void ghost_tsmm_inplace_kernelmap_generate() 
{
#include "tsmm_inplace.def"
}

ghost_tsmm_inplace_kernel_t ghost_tsmm_inplace_kernel(ghost_tsmm_inplace_parameters_t p, ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, int reduce)
{
    if (x != v) {
        return NULL;
    }
    if (x->traits.datatype != w->traits.datatype) {
        return NULL;
    }
    if (w->traits.storage != GHOST_DENSEMAT_COLMAJOR) {
        return NULL;
    }
    if (x->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        return NULL;
    }
    if ((x->traits.flags & GHOST_DENSEMAT_SCATTERED) || (w->traits.flags & GHOST_DENSEMAT_SCATTERED) || !ghost_bitmap_iscompact(x->ldmask) || !ghost_bitmap_iscompact(w->ldmask)) {
        return NULL;
    }
    if (reduce != GHOST_GEMM_NO_REDUCE) {
        return NULL;
    }

    ghost_tsmm_inplace_kernel_t kernel = ghost_tsmm_inplace_kernels[p];
    
    return kernel;
}
