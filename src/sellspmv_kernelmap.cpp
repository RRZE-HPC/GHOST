#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/sell.h"
#include "ghost/sell_kernel_avx_gen.h"
#include "ghost/sell_kernel_sse_gen.h"

#include <map>

using namespace std;
    
bool operator<(const ghost_sellspmv_parameters_t &a, const ghost_sellspmv_parameters_t &b) 
{ 
    return ghost_hash(ghost_hash(a.mdt,a.blocksz,a.storage),ghost_hash(a.vdt,a.impl,a.chunkheight),0) < ghost_hash(ghost_hash(b.mdt,b.blocksz,b.storage),ghost_hash(b.vdt,b.impl,b.chunkheight),0); 
}

static ghost_error_t (*SELL_kernels_plain[4][4]) (ghost_sparsemat_t *, ghost_densemat_t *, ghost_densemat_t *, ghost_spmv_flags_t, va_list argp) = 
{{&ss_SELL_kernel_plain,&sd_SELL_kernel_plain,&sc_SELL_kernel_plain,&sz_SELL_kernel_plain},
    {&ds_SELL_kernel_plain,&dd_SELL_kernel_plain,&dc_SELL_kernel_plain,&dz_SELL_kernel_plain},
    {&cs_SELL_kernel_plain,&cd_SELL_kernel_plain,&cc_SELL_kernel_plain,&cz_SELL_kernel_plain},
    {&zs_SELL_kernel_plain,&zd_SELL_kernel_plain,&zc_SELL_kernel_plain,&zz_SELL_kernel_plain}};


static map<ghost_sellspmv_parameters_t, sellspmv_kernel> ghost_sellspmv_kernels;

void ghost_sellspmv_kernelmap_generate() {
#include "sell_kernel_avx.def"
#include "sell_kernel_sse.def"
    
    int mdt, vdt;
    for (mdt = 0; mdt<4; mdt++) {
        for (vdt = 0; vdt<4; vdt++) {
            ghost_datatype_t mdt_t, vdt_t;
            ghost_idx2datatype(&mdt_t,(ghost_datatype_idx_t)mdt);
            ghost_idx2datatype(&vdt_t,(ghost_datatype_idx_t)vdt);
            ghost_sellspmv_parameters_t p = {.mdt = mdt_t, .vdt = vdt_t, .blocksz = -1, .storage = GHOST_DENSEMAT_STORAGE_ANY, .impl = GHOST_IMPLEMENTATION_PLAIN, .chunkheight = -1};
            ghost_sellspmv_kernels[p] =  SELL_kernels_plain[mdt][vdt];
        }
    }
}

sellspmv_kernel ghost_sellspmv_kernel(ghost_sellspmv_parameters_t p)
{

    if (p.impl == GHOST_IMPLEMENTATION_AVX && p.blocksz <= 2) {
        INFO_LOG("Chose SSE over AVX for blocksz=2");
        p.impl = GHOST_IMPLEMENTATION_SSE;
    }
    sellspmv_kernel kernel = ghost_sellspmv_kernels[p];

    if (!kernel) {
        INFO_LOG("Try kernel with arbitrary blocksz");
        p.blocksz = -1;
    }
    kernel = ghost_sellspmv_kernels[p];
    if (!kernel) {
        INFO_LOG("Try plain C kernel");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
        p.storage = GHOST_DENSEMAT_STORAGE_ANY;
        p.chunkheight = -1;
    }
    kernel = ghost_sellspmv_kernels[p];
    
    return kernel;

}
