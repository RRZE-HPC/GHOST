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
    return ghost_hash(a.dt,a.blocksz,0) < ghost_hash(b.dt,b.blocksz,0); 
}


static map<ghost_tsmm_parameters_t, tsmm_kernel> ghost_tsmm_kernels;

void ghost_tsmm_kernelmap_generate() 
{
#include "tsmm.def"
}

tsmm_kernel ghost_tsmm_kernel(ghost_tsmm_parameters_t p)
{
    tsmm_kernel kernel = ghost_tsmm_kernels[p];
    
    return kernel;
}
