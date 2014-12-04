#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/math.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmttsm_gen.h"

#include <map>

using namespace std;

bool operator<(const ghost_tsmttsm_parameters_t &a, const ghost_tsmttsm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz1,a.blocksz2) < ghost_hash(b.dt,b.blocksz1,b.blocksz2); 
}

static map<ghost_tsmttsm_parameters_t, ghost_tsmttsm_kernel_t> ghost_tsmttsm_kernels;

void ghost_tsmttsm_kernelmap_generate() {
#include "tsmttsm.def"
}

ghost_tsmttsm_kernel_t ghost_tsmttsm_kernel(ghost_tsmttsm_parameters_t p, ghost_densemat_t *x, ghost_densemat_t *v, ghost_densemat_t *w, int reduce)
{
    if (w->traits.storage != GHOST_DENSEMAT_ROWMAJOR) {
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
    if (reduce != GHOST_GEMM_ALL_REDUCE) {
        return NULL;
    }
    
    ghost_tsmttsm_kernel_t kernel = ghost_tsmttsm_kernels[p];
    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary blocksz2");
        p.blocksz2 = -1;
    }
    kernel = ghost_tsmttsm_kernels[p];
    
    return kernel;
}
