#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/instr.h"
#include "ghost/omp.h"
#include "ghost/machine.h"
#include "ghost/math.h"
#include "ghost/sparsemat.h"
#include "ghost/densemat.h"
#include "ghost/locality.h"
#include <complex>
#include <complex.h>
#include "ghost/timing.h"
#include "ghost/sparsemat.h"
#ifdef GHOST_HAVE_NAME
#include <NAME/interface.h>
#endif


inline void sell_c_sigmize_Kernel(int start, int end, void *args)
{
    ghost_sparsemat *mat = (ghost_sparsemat *)(args);

    ghost_lidx C = mat->traits.C;
    for(int row=start; row<end; ++row)
    {
        ghost_lidx chunk = row/C;
        ghost_lidx rowinchunk = row%C;
        ghost_lidx base_idx = mat->chunkStart[chunk] + rowinchunk;
        for(int j=mat->rowLen[row]; j<mat->chunkLen[chunk]; ++j)
        {
            ghost_lidx idx = base_idx + j*C;
            mat->col[idx] = start;
        }
    }
}

extern "C" {
void ghost_sell_c_sigmize(ghost_sparsemat *mat)
{
#ifdef GHOST_HAVE_NAME
    NAMEInterface *ce = (NAMEInterface*) (mat->context->coloringEngine);
    void* argPtr = (void*) (mat);
    int sell_c_sigma_id = ce->registerFunction(&sell_c_sigmize_Kernel, argPtr);
    ce->executeFunction(sell_c_sigma_id);
#else
    UNUSED(mat);
    ERROR_LOG("Enable RACE library");
#endif
}
}
