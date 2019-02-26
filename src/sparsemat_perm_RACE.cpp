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

        char *color_dist_env = getenv("GHOST_COLOR_DISTANCE");
        int color_dist = 2;

        if(color_dist_env)
        {
            color_dist = atoi(color_dist_env);
        }

        if(color_dist != 1 && color_dist != 2)
        {
            printf("Dist %d not supported, falling back to dist-2\n",color_dist);
            color_dist = 2;
        }

        RACE::dist dist_ = RACE::TWO;

        if(color_dist == 1)
            dist_ = RACE::ONE;

        //convert from SELL-C-sigma to CRS
        RACE::Interface *ce = new RACE::Interface(mat->context->row_map->dim, nthread, dist_, mat->chunkStart, mat->col, smt, RACE::FILL, ctx->row_map->loc_perm_inv, ctx->row_map->loc_perm);
        ce->RACEColor();

        printf("coloring efficiency = %f\n", ce->getEfficiency());
        printf("max. stage depth = %d\n", ce->getMaxStageDepth());
        int permLen;

        ghost_lidx* finalPerm;
        ghost_lidx* finalInvPerm;
        //perm and invPerm have been switched in library
        ce->getInvPerm(&finalPerm, &permLen);
        ce->getPerm(&finalInvPerm, &permLen);

        if(ctx->row_map->loc_perm)
        {
            free(ctx->row_map->loc_perm);
            free(ctx->row_map->loc_perm_inv);
        }

        ctx->row_map->loc_perm = finalPerm;
        ctx->row_map->loc_perm_inv = finalInvPerm;


        ctx->col_map->loc_perm = ctx->row_map->loc_perm;
        ctx->col_map->loc_perm_inv = ctx->row_map->loc_perm_inv;

//#if 0
        //pin Thread according to RACE
        omp_set_dynamic(0);    //  Explicitly disable dynamic teams
        int availableThreads = ce->getNumThreads();
        omp_set_num_threads(availableThreads);

#pragma omp parallel
        {
            int pinOrder = omp_get_thread_num();
            ce->pinThread(pinOrder);
        }
//#endif
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

        if( mat->traits.flags & GHOST_SPARSEMAT_RACE )
        {
            RACE::Interface *ce = (RACE::Interface*) mat->context->coloringEngine;

            bool diag_first = false;

            if(mat->traits.flags & GHOST_SPARSEMAT_DIAG_FIRST)
            {
                diag_first = true;
            }

            bool ret = ce->simdify(simdWidth, mat->traits.C, mat->context->row_map->dim, mat->col, mat->chunkStart, mat->rowLen, mat->chunkLenPadded, ((double*) mat->val), diag_first);
            if(!ret)
            {
                GHOST_ERROR_LOG("ERROR while simdifying");
                return GHOST_ERR_INVALID_ARG;
            }
        }

        return GHOST_SUCCESS;
#else
    UNUSED(mat);
    GHOST_ERROR_LOG("Simdify cannot be performed: RACE not installed");
    return GHOST_ERR_INVALID_ARG;
#endif
   }
}

extern "C" {
    ghost_error ghost_simdifyD1_RACE(ghost_sparsemat* mat)
    {
        int simdWidth =  ghost_machine_simd_width()/8;
        printf("simdWidth set to %d\n", simdWidth);
#ifdef GHOST_HAVE_RACE

        if( mat->traits.flags & GHOST_SPARSEMAT_RACE )
        {
            RACE::Interface *ce = (RACE::Interface*) mat->context->coloringEngine;

            bool ret = ce->simdifyD1(simdWidth, mat->traits.C, mat->context->row_map->dim, mat->col, mat->chunkStart, mat->rowLen, mat->chunkLenPadded, ((double*) mat->val), false);
            if(!ret)
            {
                GHOST_ERROR_LOG("ERROR while simdifying");
                return GHOST_ERR_INVALID_ARG;
            }
        }

        return GHOST_SUCCESS;
#else
    UNUSED(mat);
    GHOST_ERROR_LOG("SimdifyD1 cannot be performed: RACE not installed");
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

