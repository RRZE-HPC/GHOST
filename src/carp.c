#include "ghost/carp.h"
#include "ghost/crs_kacz.h"
#include "ghost/util.h"
#include "ghost/locality.h"

ghost_error_t ghost_carp(ghost_sparsemat_t *mat, ghost_densemat_t *x, ghost_densemat_t *b, void *omega)
{

    // 1. communicate local x entries to remote halo
    // 2. perform kacz on full matrix
    // 3. communicate halo x entries to remote local
    //    if column is present on more procs: average!

    ghost_spmv_haloexchange_initiate(x,NULL,false);
    ghost_spmv_haloexchange_finalize(x);
    dd_crs_kacz(mat,x,b,omega);

    return GHOST_SUCCESS;

}
