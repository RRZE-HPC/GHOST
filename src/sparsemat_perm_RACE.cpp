#include "ghost/sparsemat.h"
#include "ghost/omp.h"
#include "ghost/locality.h"
#include "ghost/util.h"
#include "ghost/machine.h"
#ifdef GHOST_HAVE_RACE
#include "RACE/interface.h"
#include "RACE/simdify.h"
#endif
//Just for checking

extern "C" {
    ghost_error ghost_sparsemat_perm_RACE(ghost_context *ctx, ghost_sparsemat *mat)
    {
#ifdef GHOST_HAVE_RACE
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

        //convert from SELL-C-sigma to CRS
        RACE::Interface *ce = new RACE::Interface(mat->context->row_map->dim, nthread, RACE::TWO, mat->chunkStart, mat->col, smt, RACE::FILL, ctx->row_map->loc_perm_inv, ctx->row_map->loc_perm);
        ce->RACEColor();

        int permLen;
        //perm and invPerm have been switched in library
        ce->getInvPerm(&ctx->row_map->loc_perm, &permLen);
        ce->getPerm(&ctx->row_map->loc_perm_inv, &permLen);

        ctx->col_map->loc_perm = ctx->row_map->loc_perm;
        ctx->col_map->loc_perm_inv = ctx->row_map->loc_perm_inv;


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

        ctx->coloringEngine = (void *) (ce);
#else
        UNUSED(ctx);
        UNUSED(mat);
#endif
        return GHOST_SUCCESS;

   }
}

extern "C" {
    ghost_error ghost_destroy_RACE(ghost_context *ctx)
    {
#ifdef GHOST_HAVE_RACE
        RACE::Interface *ce = (RACE::Interface*) ctx->coloringEngine;
        delete ce;
#else
        UNUSED(ctx);
#endif
        return GHOST_SUCCESS;
    }
}

extern "C" {
    ghost_error ghost_simdify_RACE(ghost_sparsemat* mat)
    {
        int simdWidth =  ghost_machine_simd_width()/8;
        printf("simdWidth set to %d\n", simdWidth);
#ifdef GHOST_HAVE_RACE
        GHOST_WARNING_LOG("Dist-1 dependent kernel will not work currently with C>1");

        if( mat->traits.flags & GHOST_SPARSEMAT_RACE )
        {
            RACE::Interface *ce = (RACE::Interface*) mat->context->coloringEngine;

            bool ret = ce->simdify(simdWidth, mat->traits.C, mat->context->row_map->dim, mat->col, mat->chunkStart, mat->rowLen, mat->chunkLenPadded, ((double*) mat->val));
            if(!ret)
            {
                GHOST_ERROR_LOG("ERROR while simdifying");
                return GHOST_ERR_INVALID_ARG;
            }
        }

        return GHOST_SUCCESS;
#else
    UNUSED(mat);
    GHOST_ERROR_LOG("Sidify cannot be performed: RACE not installed");
    return GHOST_ERR_INVALID_ARG;
#endif
   }
}

extern "C" {
    void ghost_sleep_RACE(ghost_context* ctx)
    {
#ifdef GHOST_HAVE_RACE
        RACE::Interface *ce = (RACE::Interface*) ctx->coloringEngine;
        ce->sleep();
#else
        UNUSED(ctx);
#endif
    }
}
