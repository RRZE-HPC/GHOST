#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmm.h"
#include "ghost/tsmm_gen.h"

#include <map>

using namespace std;

    
bool operator<(const ghost_tsmm_parameters_t &a, const ghost_tsmm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz1,a.blocksz2) < ghost_hash(b.dt,b.blocksz1,b.blocksz2); 
}


static map<ghost_tsmm_parameters_t, ghost_tsmm_kernel_t> ghost_tsmm_kernels;

void ghost_tsmm_kernelmap_generate() 
{
#include "tsmm.def"
}

ghost_tsmm_kernel_t ghost_tsmm_kernel(ghost_tsmm_parameters_t p, ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, int reduce)
{
    if (x->traits.datatype != v->traits.datatype || x->traits.datatype != w->traits.datatype) {
        return NULL;
    }
    if (w->traits.storage != GHOST_DENSEMAT_COLMAJOR) {
        return NULL;
    }
    if (x->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        return NULL;
    }
    if (v->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
        return NULL;
    }
    if (v->traits.datatype != (GHOST_DT_DOUBLE|GHOST_DT_REAL)) {
        return NULL;
    }
    if (v->traits.datatype != w->traits.datatype || v->traits.datatype != x->traits.datatype) {
        return NULL;
    }
    if ((x->traits.flags & GHOST_DENSEMAT_SCATTERED) || (v->traits.flags & GHOST_DENSEMAT_SCATTERED) || (w->traits.flags & GHOST_DENSEMAT_SCATTERED) || !ghost_bitmap_iscompact(x->ldmask) || !ghost_bitmap_iscompact(v->ldmask) || !ghost_bitmap_iscompact(w->ldmask)) {
        return NULL;
    }
    if (reduce != GHOST_GEMM_NO_REDUCE) {
        return NULL;
    }

    ghost_tsmm_kernel_t kernel = ghost_tsmm_kernels[p];
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary blocksz2");
        p.blocksz2 = -1;
    }
    kernel = ghost_tsmm_kernels[p];
    
    return kernel;
}
