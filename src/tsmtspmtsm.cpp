#include <vector>
#include <algorithm>
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/densemat.h"
#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
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
    ghost_sparsemat *A, void *pAlpha, void *pBeta, int reduce)
{
    ghost_error ret;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);


    GHOST_INFO_LOG("TSMTSPMTSM %dx%d = %dx%d * %dx%d * %dx%d", x->traits.ncols, DM_NROWS(x),
        v->traits.ncols, DM_NROWS(v), SPM_NROWS(A), SPM_NROWS(A), DM_NROWS(w), w->traits.ncols);
    if (DM_NROWS(v) != DM_NROWS(w) || w->traits.ncols != x->traits.ncols
        || v->traits.ncols != DM_NROWS(x) || SPM_NROWS(A) != DM_NROWS(v)) {
        GHOST_ERROR_LOG("Input vector/matrix sizes mismatch, %dx%d = %dx%d * %dx%d * %dx%d",
            x->traits.ncols, DM_NROWS(x), v->traits.ncols, DM_NROWS(v), SPM_NROWS(A), SPM_NROWS(A),
            DM_NROWS(w), w->traits.ncols);
        return GHOST_ERR_INVALID_ARG;
    }
    if (v->traits.datatype != w->traits.datatype || w->traits.datatype != A->traits.datatype) {
        GHOST_ERROR_LOG("Invalid pairing of data types\n");
        return GHOST_ERR_DATATYPE;
    }


#ifdef GHOST_HAVE_MPI
    GHOST_INSTR_START("comm");
    int me = 0;
    ghost_rank(&me, v->map->mpicomm);
    vector<char> zero_beta(x->elSize, 0);
    // make sure that the initial x only gets added up once
    if (me != 0 && (reduce != GHOST_GEMM_NO_REDUCE)) {
        pBeta = zero_beta.data();
    }
    ghost_densemat_halo_comm comm = GHOST_DENSEMAT_HALO_COMM_INITIALIZER;
    GHOST_CALL(ghost_densemat_halocomm_init(w, A->context, &comm), ret);
    GHOST_CALL(ghost_densemat_halocomm_start(w, A->context, &comm), ret);
    GHOST_CALL(ghost_densemat_halocomm_finalize(w, A->context, &comm), ret);
    GHOST_INSTR_STOP("comm");
#endif

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

    GHOST_INSTR_START("comp");
    ret = impls[0].pimpl(x, v, w, A, pAlpha, pBeta);
    GHOST_INSTR_STOP("comp");

    if (reduce != GHOST_GEMM_NO_REDUCE) {
        ghost_densemat_reduce(x, reduce);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}
