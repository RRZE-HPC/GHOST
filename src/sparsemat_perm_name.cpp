#include "ghost/sparsemat.h"
#include "ghost/omp.h"
#include "ghost/locality.h"
#include "ghost/util.h"
#ifdef GHOST_HAVE_NAME
#include "NAME/interface.h"
#endif
//Just for checking

extern "C" {
    ghost_error ghost_sparsemat_perm_name(ghost_context *ctx, ghost_sparsemat *mat)
    {
#ifdef GHOST_HAVE_NAME
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
        ghost_hwconfig hwconfig;
        ghost_hwconfig_get(&hwconfig);
        int smt=std::max(hwconfig.nsmt,1);
        printf("smt = %d\n",smt);
        NAMEInterface *bmc = new NAMEInterface(mat->context->row_map->dim, nthread, TWO, mat->chunkStart, mat->col, smt, FILL, ctx->row_map->loc_perm_inv, ctx->row_map->loc_perm);
        bmc->NAMEColor();

        int permLen;
        //perm and invPerm have been switched in library
        bmc->getInvPerm(&ctx->row_map->loc_perm, &permLen);
        bmc->getPerm(&ctx->row_map->loc_perm_inv, &permLen);
/*
        FILE *file;
        file = fopen("perm.txt", "w");
        printf("PermLen = %d\n",permLen);
        for(int i=0; i<permLen; ++i)
        {
            fprintf(file, "%d\n", ctx->row_map->loc_perm[i]);
        }
        fclose(file);
*/

        mat->context->coloringEngine = (void *) (bmc);
#else
        UNUSED(ctx);
        UNUSED(mat);
#endif
        return GHOST_SUCCESS;

   }
}

extern "C" {
    ghost_error destroy_name(ghost_context *ctx)
    {
#ifdef GHOST_HAVE_NAME
        NAMEInterface *ce = (NAMEInterface*) ctx->coloringEngine;
        delete ce;
#else
        UNUSED(ctx);
#endif
        return GHOST_SUCCESS;
    }
}
