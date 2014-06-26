#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/tsmttsm.h"
#include "ghost/tsmttsm_gen.h"

#include <map>

using namespace std;

bool operator<(const ghost_tsmttsm_parameters_t &a, const ghost_tsmttsm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz,0) < ghost_hash(b.dt,b.blocksz,0); 
}

static map<ghost_tsmttsm_parameters_t, tsmttsm_kernel> ghost_tsmttsm_kernels;

void ghost_tsmttsm_kernelmap_generate() {
#include "tsmttsm.def"
}

tsmttsm_kernel ghost_tsmttsm_kernel(ghost_tsmttsm_parameters_t p)
{
    tsmttsm_kernel kernel = ghost_tsmttsm_kernels[p];
    
    return kernel;
}
