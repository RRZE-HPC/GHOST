#include "ghost/sparsemat.h"
#include "ghost/omp.h"
#include "ghost/locality.h"
#include "ghost/util.h"
#include "ghost/machine.h"
#ifdef GHOST_HAVE_NAME
#include "NAME/interface.h"
#include "NAME/simdify.h"
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

        //convert from SELL-C-sigma to CRS
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

extern "C" {
    ghost_error simdify(ghost_sparsemat* mat)
    {
        int simdWidth =  ghost_machine_simd_width()/8;
        printf("simdWidth set to %d\n", simdWidth);
#ifdef GHOST_HAVE_NAME
        if( mat->traits.flags & GHOST_SPARSEMAT_NAME )
        {
            NAMEInterface *ce = (NAMEInterface*) mat->context->coloringEngine;

            bool ret = ce->simdify(simdWidth, mat->traits.C, mat->context->row_map->dim, mat->col, mat->chunkStart, mat->rowLen, mat->chunkLenPadded, ((double*) mat->val));
            if(!ret)
            {
                ERROR_LOG("ERROR while simdifying");
                return GHOST_ERR_INVALID_ARG;
            }
        }

        return GHOST_SUCCESS;
#else
    ERROR_LOG("Sidify cannot be performed: RACE not installed");
    return GHOST_ERR_INVALID_ARG;
#endif
   }
}

extern "C" {
    void ghost_ce_sleep(ghost_context* ctx)
    {
        NAMEInterface *ce = (NAMEInterface*) ctx->coloringEngine;
        ce->sleep();
    }
}
