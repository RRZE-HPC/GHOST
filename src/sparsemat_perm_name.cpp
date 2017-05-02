#include "ghost/sparsemat.h"
#include "NAME/interface.h"
#include "ghost/omp.h"
//Just for checking

extern "C" {
    ghost_error ghost_sparsemat_perm_name(ghost_context *ctx, ghost_sparsemat *mat)
    {
        int nthread;

#ifdef GHOST_HAVE_OPENMP
#pragma omp parallel
        {
#pragma omp master
            nthread = ghost_omp_nthread();
        }
#else
        nthread = 1;
#endif
        NAMEInterface *bmc = new NAMEInterface(mat->context->row_map->dim, nthread, TWO, mat->chunkStart, mat->col, ctx->row_map->loc_perm_inv, ctx->row_map->loc_perm);
        bmc->NAMEColor();

        int permLen;
        //perm and invPerm have been switched in library
        bmc->getInvPerm(&ctx->row_map->loc_perm, &permLen);
        bmc->getPerm(&ctx->row_map->loc_perm_inv, &permLen);

        mat->context->coloringEngine = (void *) (bmc);

        return GHOST_SUCCESS;

   }
}
