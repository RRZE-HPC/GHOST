#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/tsmm.h"
#include "ghost/tsmm_gen.h"

#include <map>

using namespace std;

    
bool operator<(const ghost_tsmm_parameters_t &a, const ghost_tsmm_parameters_t &b) 
{ 
    return ghost_hash(a.dt,a.blocksz1,a.blocksz2) < ghost_hash(b.dt,b.blocksz1,b.blocksz2); 
}


static map<ghost_tsmm_parameters_t, tsmm_kernel> ghost_tsmm_kernels;

void ghost_tsmm_kernelmap_generate() 
{
#include "tsmm.def"
}

tsmm_kernel ghost_tsmm_kernel(ghost_tsmm_parameters_t p)
{
    tsmm_kernel kernel = ghost_tsmm_kernels[p];
    if (!kernel) {
        INFO_LOG("Try kernel with arbitrary blocksz2");
        p.blocksz2 = -1;
    }
    kernel = ghost_tsmm_kernels[p];
    
    return kernel;
}
