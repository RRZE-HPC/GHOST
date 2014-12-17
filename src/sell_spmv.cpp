#include "ghost/types.h"
#include "ghost/omp.h"

#include "ghost/complex.h"
#include "ghost/util.h"
#include "ghost/sell.h"
#include "ghost/densemat.h"
#include "ghost/math.h"
#include "ghost/log.h"
#include "ghost/machine.h"

#include "ghost/sell_spmv_avx_gen.h"
#include "ghost/sell_spmv_sse_gen.h"

#include <map>

using namespace std;
    
    template<typename m_t, typename v_t, bool scatteredrows> 
static ghost_error_t ghost_sell_spmv_plain_rm(ghost_sparsemat_t *mat, 
        ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    ghost_sell_t *sell = (ghost_sell_t *)(mat->data);
    v_t *local_dot_product = NULL, *partsums = NULL;
    ghost_lidx_t i,j,c,rcol,lcol,cidx;
    ghost_lidx_t v;
    int nthreads = 1;
    int ch = sell->chunkHeight;

    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    int pad = (int) clsize/sizeof(v_t);

    v_t scale = 1., beta = 1.;
    v_t *shift = NULL;
    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,v_t,
            v_t);

    if (options & GHOST_SPMV_DOT_ANY) {
#pragma omp parallel
        nthreads = ghost_omp_nthread();

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,
                    (3*lhs->traits.ncols+pad)*nthreads*sizeof(v_t))); 
        for (i=0; i<(3*lhs->traits.ncols+pad)*nthreads; i++) {
            partsums[i] = 0.;
        }
    }
#pragma omp parallel private(c,j,rcol,lcol,cidx,i,v) shared(partsums)
    {
        v_t **tmp;
        ghost_malloc((void **)&tmp,sizeof(v_t *)*ch);
        ghost_malloc((void **)&tmp[0],
                sizeof(v_t)*ch*rhs->traits.ncols);
        for (i=1; i<ch; i++) {
            tmp[i] = tmp[i-1] + rhs->traits.ncols;
        }
        v_t **lhsv = NULL;
        int tid = ghost_omp_threadnum();
        v_t * rhsrow;
        v_t matrixval;

#pragma omp for schedule(runtime) 
        for (c=0; c<mat->nrowsPadded/ch; c++) { // loop over chunks
            lhsv = (v_t **)&(lhs->val[c*ch]);

            for (i=0; i<ch; i++) {
                for (rcol=0; rcol<rhs->traits.ncols; rcol++) {
                    tmp[i][rcol] = (v_t)0;
                }
            }

            for (j=0; j<sell->chunkLen[c]; j++) { // loop inside chunk
                for (i=0; i<ch; i++) {
                    matrixval = (v_t)(((m_t*)(sell->val))
                            [sell->chunkStart[c]+j*ch+i]);
                    rhsrow = (v_t *)rhs->val
                        [sell->col[sell->chunkStart[c]+j*ch+i]];
                    rcol = ghost_bitmap_first(rhs->ldmask)-1;
                    for (cidx = 0; cidx<rhs->traits.ncols; cidx++) {
                        if (scatteredrows) {
                            rcol = ghost_bitmap_next(rhs->ldmask,rcol);
                        } else {
                            rcol++;
                        }
                        tmp[i][cidx] +=  matrixval * rhsrow[rcol];
                    }

                }
            }

            for (i=0; (i<ch) && (c*ch+i < mat->nrows); i++) {
                rhsrow = (v_t *)rhs->val[c*ch+i];
                rcol = ghost_bitmap_first(rhs->ldmask)-1;
                lcol = ghost_bitmap_first(lhs->ldmask)-1;
                for (cidx = 0; cidx<lhs->traits.ncols; cidx++) {
                    if (scatteredrows) {
                        rcol = ghost_bitmap_next(rhs->ldmask,rcol);
                        lcol = ghost_bitmap_next(lhs->ldmask,lcol);
                    } else {
                        rcol++;
                        lcol++;
                    }
                    if ((options & GHOST_SPMV_SHIFT) && shift) {
                        tmp[i][cidx] = tmp[i][cidx]-shift[0]*rhsrow[rcol];
                    }
                    if ((options & GHOST_SPMV_VSHIFT) && shift) {
                        tmp[i][cidx] = tmp[i][cidx]-shift[cidx]*rhsrow[rcol];
                    }
                    if (options & GHOST_SPMV_SCALE) {
                        tmp[i][cidx] = tmp[i][cidx]*scale;
                    }
                    if (options & GHOST_SPMV_AXPY) {
                        lhsv[i][lcol] += tmp[i][cidx];
                    } else if (options & GHOST_SPMV_AXPBY) {
                        lhsv[i][lcol] = beta*lhsv[i][lcol] + tmp[i][cidx];
                    } else {
                        lhsv[i][lcol] = tmp[i][cidx];
                    }

                    if (options & GHOST_SPMV_DOT_ANY) {
                        partsums[((pad+3*lhs->traits.ncols)*tid)+3*cidx+0] += 
                            conjugate(&lhsv[i][lcol])*lhsv[i][rcol];
                        partsums[((pad+3*lhs->traits.ncols)*tid)+3*cidx+1] += 
                            conjugate(&lhsv[i][lcol])*rhsrow[rcol];
                        partsums[((pad+3*lhs->traits.ncols)*tid)+3*cidx+2] += 
                            conjugate(&rhsrow[rcol])*rhsrow[rcol];
                    }
                }
            }

        }
    }
    if (options & GHOST_SPMV_DOT_ANY) {
        if (!local_dot_product) {
            ERROR_LOG("The location of the local dot products is NULL!");
            return GHOST_ERR_INVALID_ARG;
        }
        for (v=0; v<lhs->traits.ncols; v++) {
            local_dot_product[v                       ] = 0.; 
            local_dot_product[v  +   lhs->traits.ncols] = 0.;
            local_dot_product[v  + 2*lhs->traits.ncols] = 0.;
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                      ] += 
                    partsums[(pad+3*lhs->traits.ncols)*i + 3*v + 0];
                local_dot_product[v +   lhs->traits.ncols] += 
                    partsums[(pad+3*lhs->traits.ncols)*i + 3*v + 1];
                local_dot_product[v + 2*lhs->traits.ncols] += 
                    partsums[(pad+3*lhs->traits.ncols)*i + 3*v + 2];
            }
        }
        free(partsums);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}
    
    template<typename m_t, typename v_t> 
static ghost_error_t ghost_sell_spmv_plain_rm_selector(ghost_sparsemat_t *mat, 
        ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    if (lhs->traits.ncolsorig != lhs->traits.ncols || 
            rhs->traits.ncolsorig != rhs->traits.ncols) {
        return ghost_sell_spmv_plain_rm<m_t,v_t,true>(mat,lhs,rhs,options,argp);
    } else {
        return ghost_sell_spmv_plain_rm<m_t,v_t,false>(mat,lhs,rhs,options,argp);
    }
}

    template<typename m_t, typename v_t> 
static ghost_error_t ghost_sell_spmv_plain_cm(ghost_sparsemat_t *mat, 
        ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    ghost_sell_t *sell = (ghost_sell_t *)(mat->data);
    v_t *rhsv = NULL;
    v_t *local_dot_product = NULL, *partsums = NULL;
    ghost_lidx_t i,j,c;
    ghost_lidx_t v;
    int nthreads = 1;
    int ch = sell->chunkHeight;

    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    int pad = (int) clsize/sizeof(v_t);

    v_t scale = 1., beta = 1.;
    v_t *shift = NULL;
    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,v_t,
            v_t);

    if (options & GHOST_SPMV_DOT_ANY) {
#pragma omp parallel
        nthreads = ghost_omp_nthread();

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,
                    (3*lhs->traits.ncols+pad)*nthreads*sizeof(v_t))); 
        for (i=0; i<(3*lhs->traits.ncols+pad)*nthreads; i++) {
            partsums[i] = 0.;
        }
    }
#pragma omp parallel private(c,j,i,v) shared(partsums)
    {
        v_t *tmp = NULL;
        ghost_malloc((void **)&tmp,ch*sizeof(v_t));
        v_t *lhsv = NULL;
        int tid = ghost_omp_threadnum();


#pragma omp for schedule(runtime) 
        for (c=0; c<mat->nrowsPadded/ch; c++) { // loop over chunks
            for (v=0; v<lhs->traits.ncols; v++)
            {
                rhsv = (v_t *)rhs->val[v];
                lhsv = (v_t *)lhs->val[v];

                for (i=0; i<ch; i++) {
                    tmp[i] = (v_t)0;
                }

                for (j=0; j<sell->chunkLen[c]; j++) { // loop inside chunk
                    for (i=0; i<ch; i++) {
                        tmp[i] += (v_t)(((m_t*)(sell->val))[sell->chunkStart[c]+
                                j*ch+i]) * rhsv[sell->col[sell->chunkStart[c]+
                            j*ch+i]];
                    }
                }
                for (i=0; i<ch; i++) {
                    if (c*ch+i < mat->nrows) {
                        if ((options & GHOST_SPMV_SHIFT) && shift) {
                            tmp[i] = tmp[i]-shift[0]*rhsv[c*ch+i];
                        }
                        if ((options & GHOST_SPMV_VSHIFT) && shift) {
                            tmp[i] = tmp[i]-shift[v]*rhsv[c*ch+i];
                        }
                        if (options & GHOST_SPMV_SCALE) {
                            tmp[i] = tmp[i]*scale;
                        }
                        if (options & GHOST_SPMV_AXPY) {
                            lhsv[c*ch+i] += tmp[i];
                        } else if (options & GHOST_SPMV_AXPBY) {
                            lhsv[c*ch+i] = beta*lhsv[c*ch+i] + tmp[i];
                        } else {
                            lhsv[c*ch+i] = tmp[i];
                        }

                        if (options & GHOST_SPMV_DOT_ANY) {
                            partsums[((pad+3*lhs->traits.ncols)*tid)+3*v+0] += 
                                conjugate(&lhsv[c*ch+i])*
                                lhsv[c*ch+i];
                            partsums[((pad+3*lhs->traits.ncols)*tid)+3*v+1] += 
                                conjugate(&lhsv[c*ch+i])*
                                rhsv[c*ch+i];
                            partsums[((pad+3*lhs->traits.ncols)*tid)+3*v+2] += 
                                conjugate(&rhsv[c*ch+i])*
                                rhsv[c*ch+i];
                        }
                    }

                }

            }
        }
        free(tmp);
    }
    if (options & GHOST_SPMV_DOT_ANY) {
        if (!local_dot_product) {
            ERROR_LOG("The location of the local dot products is NULL!");
            return GHOST_ERR_INVALID_ARG;
        }
        for (v=0; v<lhs->traits.ncols; v++) {
            local_dot_product[v                       ] = 0.; 
            local_dot_product[v  +   lhs->traits.ncols] = 0.;
            local_dot_product[v  + 2*lhs->traits.ncols] = 0.;
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                      ] += partsums[(pad+
                        3*lhs->traits.ncols)*i + 3*v + 0];
                local_dot_product[v +   lhs->traits.ncols] += partsums[(pad+
                        3*lhs->traits.ncols)*i + 3*v + 1];
                local_dot_product[v + 2*lhs->traits.ncols] += partsums[(pad+
                        3*lhs->traits.ncols)*i + 3*v + 2];
            }
        }
        free(partsums);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

    template<typename m_t, typename v_t> 
static ghost_error_t ghost_sell_spmv_kernel_ellpack_cm(
        ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    v_t *rhsv = NULL;
    v_t *lhsv = NULL;
    v_t *local_dot_product = NULL, *partsums = NULL;
    int nthreads = 1;
    ghost_lidx_t i,j;
    ghost_lidx_t v;
    v_t tmp;
    ghost_sell_t *sell = (ghost_sell_t *)(mat->data);
    m_t *sellv = (m_t*)(sell->val);

    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    int pad = (int) clsize/sizeof(v_t);

    v_t scale = 1., beta = 1.;
    v_t *shift = NULL;
    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,v_t,
            v_t);

    if (options & GHOST_SPMV_DOT_ANY) {
#pragma omp parallel
        nthreads = ghost_omp_nthread();

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,
                    (3*lhs->traits.ncols+pad)*nthreads*sizeof(v_t))); 
        for (i=0; i<(3*lhs->traits.ncols+pad)*nthreads; i++) {
            partsums[i] = 0.;
        }
    }

#pragma omp parallel private(i,j,tmp,v)
    {
        int tid = ghost_omp_threadnum();
#pragma omp for schedule(runtime)
        for (i=0; i<mat->nrows; i++) 
        {
            for (v=0; v<lhs->traits.ncols; v++)
            {
                rhsv = (v_t *)rhs->val[v];
                lhsv = (v_t *)lhs->val[v];
                tmp = (v_t)0;

                for (j=0; j<sell->rowLen[i]; j++) 
                {
                    tmp += (v_t)sellv[mat->nrowsPadded*j+i] * 
                        rhsv[sell->col[mat->nrowsPadded*j+i]];
                }

                if ((options & GHOST_SPMV_SHIFT) && shift) {
                    tmp = tmp-shift[0]*rhsv[i];
                }
                if ((options & GHOST_SPMV_VSHIFT) && shift) {
                    tmp = tmp-shift[v]*rhsv[i];
                }
                if (options & GHOST_SPMV_SCALE) {
                    tmp = tmp*scale;
                }
                if (options & GHOST_SPMV_AXPY) {
                    lhsv[i] += tmp;
                } else if (options & GHOST_SPMV_AXPBY) {
                    lhsv[i] = beta*lhsv[i] + tmp;
                } else {
                    lhsv[i] = tmp;
                }
                if (options & GHOST_SPMV_DOT_ANY) {
                    partsums[(v+tid*lhs->traits.ncols)*16 + 0] += 
                        conjugate(&lhsv[i])*lhsv[i];
                    partsums[(v+tid*lhs->traits.ncols)*16 + 1] += 
                        conjugate(&lhsv[i])*rhsv[i];
                    partsums[(v+tid*lhs->traits.ncols)*16 + 2] += 
                        conjugate(&rhsv[i])*rhsv[i];
                }
            }
        }
    }
    if (options & GHOST_SPMV_DOT_ANY) {
        for (v=0; v<lhs->traits.ncols; v++) {
            local_dot_product[v                       ] = 0.; 
            local_dot_product[v  +   lhs->traits.ncols] = 0.;
            local_dot_product[v  + 2*lhs->traits.ncols] = 0.;
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                      ] += 
                    partsums[(v+i*lhs->traits.ncols)*16 + 0];
                local_dot_product[v +   lhs->traits.ncols] += 
                    partsums[(v+i*lhs->traits.ncols)*16 + 1];
                local_dot_product[v + 2*lhs->traits.ncols] += 
                    partsums[(v+i*lhs->traits.ncols)*16 + 2];
            }
        }
        free(partsums);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

static bool operator<(const ghost_sellspmv_parameters_t &a, 
        const ghost_sellspmv_parameters_t &b) 
{ 
    return ghost_hash(ghost_hash(a.mdt,a.blocksz,a.storage),
            ghost_hash(a.vdt,a.impl,a.chunkheight),0) <
        ghost_hash(ghost_hash(b.mdt,b.blocksz,b.storage),
                ghost_hash(b.vdt,b.impl,b.chunkheight),0); 
}

static map<ghost_sellspmv_parameters_t, ghost_spmv_kernel_t> 
ghost_sellspmv_kernels = map<ghost_sellspmv_parameters_t,ghost_spmv_kernel_t>();

extern "C" ghost_error_t ghost_sell_spmv_selector(ghost_sparsemat_t *mat, 
        ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    if (rhs->traits.storage != lhs->traits.storage) {
        ERROR_LOG("Different storage layout for in- and output densemats!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (rhs->traits.ncols != lhs->traits.ncols) {
        ERROR_LOG("The number of columns for the densemats does not match!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (rhs->traits.nrows != lhs->traits.nrows) {
        ERROR_LOG("The number of rows for the densemats does not match!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (rhs->traits.nrows != mat->nrows) {
        ERROR_LOG("Different number of rows for the densemats and matrix!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (((rhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) && 
                (rhs->traits.nrowsorig != rhs->traits.nrows)) || 
            ((lhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) && 
            (lhs->traits.nrowsorig != lhs->traits.nrows))) {
        ERROR_LOG("Col-major densemats with masked out rows currently not "
                "supported!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }


    // if map is empty include generated code for map construction
    if (ghost_sellspmv_kernels.empty()) {
#include "sell_spmv_avx.def"
#include "sell_spmv_sse.def"
    }

    ghost_error_t ret = GHOST_SUCCESS;

    ghost_implementation_t impl = GHOST_IMPLEMENTATION_PLAIN;

    if (!(rhs->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
#ifdef GHOST_HAVE_MIC
        impl = GHOST_IMPLEMENTATION_MIC;
#elif defined(GHOST_HAVE_AVX)
        impl = GHOST_IMPLEMENTATION_AVX;
#elif defined(GHOST_HAVE_SSE)
        impl = GHOST_IMPLEMENTATION_SSE;
#endif
    }

    ghost_sellspmv_parameters_t p;
    p.impl = impl;
    p.vdt = rhs->traits.datatype;
    p.mdt = mat->traits->datatype;
    p.blocksz = rhs->traits.ncols;
    p.storage = rhs->traits.storage;
    p.chunkheight = SELL(mat)->chunkHeight;


    if (p.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz == 1 && 
            rhs->traits.ncolsorig == 1 && lhs->traits.ncolsorig== 1) {
        INFO_LOG("Chose col-major kernel for row-major densemat with 1 column");
        p.storage = GHOST_DENSEMAT_COLMAJOR;
    }

    if (p.impl == GHOST_IMPLEMENTATION_AVX && 
            p.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz <= 2 && 
            !(rhs->traits.datatype & GHOST_DT_COMPLEX)) {
        PERFWARNING_LOG("Chose SSE over AVX for blocksz=2");
        p.impl = GHOST_IMPLEMENTATION_SSE;
    }

    if (p.impl == GHOST_IMPLEMENTATION_AVX && 
            p.storage == GHOST_DENSEMAT_COLMAJOR && p.chunkheight < 4 
            && !(rhs->traits.datatype & GHOST_DT_COMPLEX)) {
        if (p.chunkheight < 2) {
            PERFWARNING_LOG("Chose plain kernel for col-major densemats and C<2");
            p.impl = GHOST_IMPLEMENTATION_PLAIN;
        } else {
            PERFWARNING_LOG("Chose SSE for col-major densemats and C<4");
            p.impl = GHOST_IMPLEMENTATION_SSE;
        }
    }

    if ((lhs->traits.flags & GHOST_DENSEMAT_SCATTERED) || 
            (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        PERFWARNING_LOG("Use plain implementation for scattered views");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }

    if (((lhs->traits.flags & GHOST_DENSEMAT_VIEW) 
            || (rhs->traits.flags & GHOST_DENSEMAT_VIEW)) &&
            p.impl == GHOST_IMPLEMENTATION_SSE) {
        WARNING_LOG("Not sure whether aligned load intrinsics on potentially "
                "unaligned addresses work with SSE.");
    }

    ghost_spmv_kernel_t kernel = ghost_sellspmv_kernels[p];

    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary blocksz");
        p.blocksz = -1;
    }
    kernel = ghost_sellspmv_kernels[p];

    if (kernel) {
        kernel(mat,lhs,rhs,options,argp);
    } else { // execute plain kernel as fallback
        PERFWARNING_LOG("Execute fallback SELL SpMV kernel which is potentially"
                " slow!");
        if (lhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
            if (SELL(mat)->chunkHeight == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
                SELECT_TMPL_2DATATYPES(mat->traits->datatype,
                        rhs->traits.datatype,ghost_complex,ret,
                        ghost_sell_spmv_kernel_ellpack_cm,mat,lhs,rhs,options,
                        argp);
            } else {
                SELECT_TMPL_2DATATYPES(mat->traits->datatype,
                        rhs->traits.datatype,ghost_complex,ret,
                        ghost_sell_spmv_plain_cm,mat,lhs,rhs,options,argp);
            }

        } else {
            SELECT_TMPL_2DATATYPES(mat->traits->datatype,
                    rhs->traits.datatype,ghost_complex,ret,
                    ghost_sell_spmv_plain_rm_selector,mat,lhs,rhs,options,argp);
        }
    } 

    return ret;
}

