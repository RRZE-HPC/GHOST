#include <vector>
#include <algorithm>
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/sparsemat.h"
#include "ghost/util.h"

#include "ghost/tsmtspmtsm.h"
#include "ghost/tsmtspmtsm_var2_plain.h"

#ifdef GHOST_HAVE_CUDA
#include "ghost/tsmtspmtsm_var2_cuda.h"
#endif

using namespace std;

typedef ghost_error (*ghost_tsmtspmtsm_kernel)(
    ghost_densemat *, ghost_densemat *, ghost_densemat *, ghost_sparsemat *, void *, void *);

struct ghost_tsmtspmtsm_impl {
    bool cuda;
    ghost_tsmtspmtsm_kernel pimpl;
};

ghost_error ghost_tsmtspmtsm(ghost_densemat *x, ghost_densemat *v, ghost_densemat *w,
    ghost_sparsemat *A, void *pAlpha, void *pBeta)
{
    ghost_error ret;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);


    if (DM_NROWS(v) != DM_NROWS(w) || w->traits.ncols != x->traits.ncols
        || v->traits.ncols != DM_NROWS(x) || A->context->row_map->gdim != DM_NROWS(v)) {
        GHOST_ERROR_LOG("Input vector/matrix sizes mismatch, %dx%d = %dx%d * %dx%d * %dx%d",
            x->traits.ncols, DM_NROWS(x), v->traits.ncols, DM_NROWS(v), A->context->row_map->gdim,
            A->context->row_map->gdim, DM_NROWS(w), w->traits.ncols);
        return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.datatype != w->traits.datatype || w->traits.datatype != A->traits.datatype) {
        GHOST_ERROR_LOG("Invalid pairing of data types\n");
        return GHOST_ERR_DATATYPE;
    }


    std::vector<ghost_tsmtspmtsm_impl> impls;
    impls.push_back(ghost_tsmtspmtsm_impl{false, tsmtspmtsm_var2_plain});

#ifdef GHOST_HAVE_CUDA
    impls.push_back(ghost_tsmtspmtsm_impl{true, tsmtspmtsm_var2_cuda});
    if (x->traits.location & GHOST_LOCATION_DEVICE && x->traits.compute_at != GHOST_LOCATION_HOST) {
        impls.erase(std::remove_if(begin(impls), end(impls),
                        [](ghost_tsmtspmtsm_impl &impl) { return !impl.cuda; }),
            end(impls));
    } else {
        impls.erase(std::remove_if(begin(impls), end(impls),
                        [](ghost_tsmtspmtsm_impl &impl) { return impl.cuda; }),
            end(impls));
    }
#endif

    ret = impls[0].pimpl(x, v, w, A, pAlpha, pBeta);

    //  ghost_densemat_reduce(x, reduce);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}
