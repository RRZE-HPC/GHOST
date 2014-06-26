#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/tsmttsm.h"

#include <map>

using namespace std;

struct ghost_tsmttsm_parameters_t
{
    ghost_datatype_t dt;
    int blocksz;

    ghost_tsmttsm_parameters_t(ghost_datatype_t dt, ghost_idx_t blocksz) : dt(dt), blocksz(blocksz) {}
};
    
bool operator<(const ghost_tsmttsm_parameters_t &a, const ghost_tsmttsm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz,0) - ghost_hash(b.dt,b.blocksz,0); 
}

typedef ghost_error_t (*tsmttsm_kernel)(ghost_densemat_t *, ghost_densemat_t *, ghost_densemat_t *, void *, void *);

static map<ghost_tsmttsm_parameters_t, tsmttsm_kernel> ghost_tsmttsm_kernels;

void ghost_tsmttsm_kernelmap_generate() {
#include "tsmttsm.def"
}
