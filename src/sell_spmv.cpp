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
#include "ghost/sell_spmv_plain_gen.h"

#include <map>

using namespace std;
    
    template<typename m_t, typename v_t, bool scatteredvecs> 
static ghost_error_t ghost_sell_spmv_plain_rm(ghost_sparsemat_t *mat, 
        ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    PERFWARNING_LOG("In plain row-major SEL SpMV with scatteredvecs=%d, blocksz=%d",scatteredvecs,rhs->traits.ncols);
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
    v_t delta = 0., eta = 0.;
    ghost_densemat_t *z = NULL;
    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,z,delta,eta,v_t,
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
        int tid = ghost_omp_threadnum();
        v_t * rhsrow, *lhsrow;
        v_t matrixval;

#pragma omp for schedule(runtime) 
        for (c=0; c<mat->nrowsPadded/ch; c++) { // loop over chunks

            for (i=0; i<ch; i++) {
                for (rcol=0; rcol<rhs->traits.ncols; rcol++) {
                    tmp[i][rcol] = (v_t)0;
                }
            }

            for (j=0; j<sell->chunkLen[c]; j++) { // loop inside chunk
                for (i=0; i<ch; i++) {
                    matrixval = (v_t)(((m_t*)(sell->val))
                            [sell->chunkStart[c]+j*ch+i]);
                    rhsrow = ((v_t *)rhs->val)+rhs->stride*sell->col[sell->chunkStart[c]+j*ch+i];
                    rcol = 0;
                    for (cidx = 0; cidx<rhs->traits.ncols; cidx++) {
                        tmp[i][cidx] +=  matrixval * rhsrow[rcol];
                        if (scatteredvecs) {
                            rcol = ghost_bitmap_next(rhs->colmask,rcol);
                        } else {
                            rcol++;
                        }
                    }

                }
            }

            for (i=0; (i<ch) && (c*ch+i < mat->nrows); i++) {
                lhsrow = ((v_t *)lhs->val)+lhs->stride*(c*ch+i);
                rhsrow = ((v_t *)rhs->val)+rhs->stride*(c*ch+i);
                rcol = 0;
                lcol = 0;
                for (cidx = 0; cidx<lhs->traits.ncols; cidx++) {
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
                        lhsrow[lcol] += tmp[i][cidx];
                    } else if (options & GHOST_SPMV_AXPBY) {
                        lhsrow[lcol] = beta*lhsrow[lcol] + tmp[i][cidx];
                    } else {
                        lhsrow[lcol] = tmp[i][cidx];
                    }

                    if (options & GHOST_SPMV_DOT_ANY) {
                        partsums[((pad+3*lhs->traits.ncols)*tid)+3*cidx+0] += 
                            conjugate(&lhsrow[lcol])*lhsrow[rcol];
                        partsums[((pad+3*lhs->traits.ncols)*tid)+3*cidx+1] += 
                            conjugate(&lhsrow[lcol])*rhsrow[rcol];
                        partsums[((pad+3*lhs->traits.ncols)*tid)+3*cidx+2] += 
                            conjugate(&rhsrow[rcol])*rhsrow[rcol];
                    }
                    if (scatteredvecs) {
                        rcol = ghost_bitmap_next(rhs->colmask,rcol);
                        lcol = ghost_bitmap_next(lhs->colmask,lcol);
                    } else {
                        rcol++;
                        lcol++;
                    }
                }
            }

        }
        free(tmp[0]);
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
    if ((lhs->traits.flags & GHOST_DENSEMAT_SCATTERED) || (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        return ghost_sell_spmv_plain_rm<m_t,v_t,true>(mat,lhs,rhs,options,argp);
    } else {
        return ghost_sell_spmv_plain_rm<m_t,v_t,false>(mat,lhs,rhs,options,argp);
    }
}

    template<typename m_t, typename v_t, bool scatteredvecs> 
static ghost_error_t ghost_sell_spmv_plain_cm(ghost_sparsemat_t *mat, 
        ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    ghost_sell_t *sell = (ghost_sell_t *)(mat->data);
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
    v_t delta = 0., eta = 0.;
    ghost_densemat_t *z = NULL;
    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,z,delta,eta,v_t,
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
        v_t *rhsv = NULL;
        int tid = ghost_omp_threadnum();


#pragma omp for schedule(runtime) 
        for (c=0; c<mat->nrowsPadded/ch; c++) { // loop over chunks
                    
            ghost_lidx_t rcol = 0, lcol = 0;

            for (v=0; v<lhs->traits.ncols; v++)
            {

                rhsv = (v_t *)rhs->val+rcol*rhs->stride;
                lhsv = (v_t *)lhs->val+lcol*rhs->stride;

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

                if (scatteredvecs) {
                    rcol = ghost_bitmap_next(rhs->colmask,rcol);
                    lcol = ghost_bitmap_next(lhs->colmask,lcol);
                } else {
                    rcol++;
                    lcol++;
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
static ghost_error_t ghost_sell_spmv_plain_cm_selector(ghost_sparsemat_t *mat, 
        ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    if ((lhs->traits.flags & GHOST_DENSEMAT_SCATTERED) || (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        return ghost_sell_spmv_plain_cm<m_t,v_t,true>(mat,lhs,rhs,options,argp);
    } else {
        return ghost_sell_spmv_plain_cm<m_t,v_t,false>(mat,lhs,rhs,options,argp);
    }
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
    v_t delta = 0., eta = 0.;
    ghost_densemat_t *z = NULL;
    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,z,delta,eta,v_t,
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
                rhsv = (v_t *)rhs->val+v*rhs->stride;
                lhsv = (v_t *)lhs->val+v*rhs->stride;
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
            ghost_hash(a.vdt,a.impl,a.chunkheight),a.alignment) <
        ghost_hash(ghost_hash(b.mdt,b.blocksz,b.storage),
                ghost_hash(b.vdt,b.impl,b.chunkheight),b.alignment); 
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
#include "sell_spmv_plain.def"
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
#else
        impl = GHOST_IMPLEMENTATION_PLAIN;
#endif
    }
        //impl = GHOST_IMPLEMENTATION_PLAIN;

    ghost_sellspmv_parameters_t p;
    p.alignment = GHOST_ALIGNED;
    p.impl = impl;
    p.vdt = rhs->traits.datatype;
    p.mdt = mat->traits->datatype;
    p.storage = rhs->traits.storage;
    p.chunkheight = SELL(mat)->chunkHeight;
    
    if (!(lhs->traits.flags & GHOST_DENSEMAT_VIEW)) {
        p.blocksz = rhs->traits.ncols;
    } else {
        p.blocksz = rhs->traits.ncolspadded;
    }

    if (p.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz == 1 && 
            rhs->stride == 1 && lhs->stride == 1) {
        INFO_LOG("Chose col-major kernel for row-major densemat with 1 column");
        p.storage = GHOST_DENSEMAT_COLMAJOR;
    }

    if (p.impl == GHOST_IMPLEMENTATION_AVX && 
            p.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz == 2 && 
            !(rhs->traits.datatype & GHOST_DT_COMPLEX)) {
        PERFWARNING_LOG("Chose SSE over AVX for blocksz=2");
        p.impl = GHOST_IMPLEMENTATION_SSE;
    }
    
    if (p.impl == GHOST_IMPLEMENTATION_AVX && 
            p.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz == 1) {
        if (rhs->traits.datatype & GHOST_DT_COMPLEX) {
            PERFWARNING_LOG("Chose SSE over AVX for blocksz=1 and complex densemat");
            p.impl = GHOST_IMPLEMENTATION_SSE;
        } else {
            PERFWARNING_LOG("Chose plain over AVX for blocksz=1");
            p.impl = GHOST_IMPLEMENTATION_PLAIN;
        }
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
    
    if (p.impl == GHOST_IMPLEMENTATION_AVX && 
            p.storage == GHOST_DENSEMAT_COLMAJOR && p.chunkheight < 2 
            && rhs->traits.datatype & GHOST_DT_COMPLEX) {
        PERFWARNING_LOG("Chose SSE for col-major densemats, complex vector and C<2");
        p.impl = GHOST_IMPLEMENTATION_SSE;
    }
    
    if (p.impl == GHOST_IMPLEMENTATION_SSE && 
            p.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz == 1 && 
            !(rhs->traits.datatype & GHOST_DT_COMPLEX)) {
        PERFWARNING_LOG("Chose plain over SSE for blocksz=1");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }
    
    if (p.impl == GHOST_IMPLEMENTATION_SSE && 
            p.storage == GHOST_DENSEMAT_COLMAJOR && p.chunkheight < 2 
            && !(rhs->traits.datatype & GHOST_DT_COMPLEX)) {
        PERFWARNING_LOG("Chose plain kernel for col-major densemats and C<2");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }

    if ((lhs->traits.flags & GHOST_DENSEMAT_SCATTERED) || 
            (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        PERFWARNING_LOG("Use plain implementation for scattered views");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }

    /*if (((lhs->traits.flags & GHOST_DENSEMAT_VIEW) 
            || (rhs->traits.flags & GHOST_DENSEMAT_VIEW)) &&
            p.impl == GHOST_IMPLEMENTATION_SSE) {
        WARNING_LOG("Not sure whether aligned load intrinsics on potentially "
                "unaligned addresses work with SSE.");
        p.alignment = GHOST_UNALIGNED;
    }*/
    if (p.impl == GHOST_IMPLEMENTATION_AVX) {
        if (!(IS_ALIGNED(lhs->val,32)) || !(IS_ALIGNED(rhs->val,32))) {
            PERFWARNING_LOG("Using unaligned version for because (one of) the vectors are not aligned to 32 bytes.");
            p.alignment = GHOST_UNALIGNED;
        }
    }
    if (p.impl == GHOST_IMPLEMENTATION_SSE) {
        if (!(IS_ALIGNED(lhs->val,16)) || !(IS_ALIGNED(rhs->val,16))) {
            PERFWARNING_LOG("Using unaligned version for because (one of) the vectors are not aligned to 32 bytes.");
            p.alignment = GHOST_UNALIGNED;
        }
    }
    if (lhs->traits.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz > 1 && p.blocksz % 4) {
        if (p.impl == GHOST_IMPLEMENTATION_AVX) {
            PERFWARNING_LOG("Use SSE implementation non-multiples of four!");
            p.impl = GHOST_IMPLEMENTATION_SSE;
        }
        if (p.blocksz % 2 && p.impl >= GHOST_IMPLEMENTATION_SSE) {
            PERFWARNING_LOG("Use plain implementation non-multiples of two!");
            p.impl = GHOST_IMPLEMENTATION_PLAIN;
        }
    }


    ghost_spmv_kernel_t kernel = ghost_sellspmv_kernels[p];

    if (!kernel) {
        PERFWARNING_LOG("Try kernel with arbitrary blocksz because blocksz %d is not available.",p.blocksz);
        /*if (p.storage == GHOST_DENSEMAT_ROWMAJOR) {
            PERFWARNING_LOG("The vectorized version is broken so I will fall back to the plain implementation!");
            p.impl = GHOST_IMPLEMENTATION_PLAIN;
        }*/
        p.blocksz = -1;
    }
    kernel = ghost_sellspmv_kernels[p];
/*
    char *str;
    lhs->string(lhs,&str);
    printf("%s\n",str);
    rhs->string(rhs,&str);
    printf("%s\n",str);*/
    if (kernel) {
        ret = kernel(mat,lhs,rhs,options,argp);
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
                        ghost_sell_spmv_plain_cm_selector,mat,lhs,rhs,options,argp);
            }

        } else {
            SELECT_TMPL_2DATATYPES(mat->traits->datatype,
                    rhs->traits.datatype,ghost_complex,ret,
                    ghost_sell_spmv_plain_rm_selector,mat,lhs,rhs,options,argp);
        }
    } 

    return ret;
}

