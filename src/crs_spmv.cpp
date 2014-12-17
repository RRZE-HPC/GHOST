#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/omp.h"

#include "ghost/locality.h"
#include "ghost/complex.h"
#include "ghost/math.h"
#include "ghost/util.h"
#include "ghost/crs.h"
#include "ghost/machine.h"

#include <sstream>
#include <iostream>
#include <cstdarg>

    template<typename m_t, typename v_t, bool scatteredrows> 
static ghost_error_t ghost_crs_spmv_plain_rm(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp)
{
    ghost_crs_t *cr = CR(mat);
    v_t *lhsv = NULL;
    v_t *local_dot_product = NULL, *partsums = NULL;
    m_t *mval = (m_t *)(cr->val);
    ghost_lidx_t i, j;
    ghost_lidx_t rcol,lcol;
    ghost_lidx_t cidx;
    int nthreads = 1;

    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    int padding = (int)clsize/sizeof(v_t);

    v_t scale = 1., beta = 1.;
    v_t *shift = NULL;

    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,v_t,v_t);
    
        
    if (options & GHOST_SPMV_DOT_ANY) {

#pragma omp parallel 
        {
#pragma omp single
            nthreads = ghost_omp_nthread();
        }

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,(3*lhs->traits.ncols+padding)*nthreads*sizeof(v_t))); 
        for (i=0; i<(3*lhs->traits.ncols+padding)*nthreads; i++) {
            partsums[i] = 0.;
        }
    }
#pragma omp parallel private (i, j, lhsv,rcol,lcol,cidx) shared (partsums)
    {
        v_t matrixval;
        v_t * rhsrow;
        v_t *tmp;
        ghost_malloc((void **)&tmp,rhs->traits.ncols*sizeof(v_t));
        int tid = ghost_omp_threadnum();
#pragma omp for schedule(runtime) 
        for (i=0; i<mat->nrows; i++) {
            lhsv = (v_t *)lhs->val[i];

            for (cidx=0; cidx<rhs->traits.ncols; cidx++) {
                tmp[cidx] = 0.;
            }
            for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
                matrixval = ((v_t)(mval[j]));
                rhsrow = (v_t *)rhs->val[cr->col[j]];
                rcol = ghost_bitmap_first(rhs->ldmask)-1;
                for (cidx = 0; cidx<rhs->traits.ncols; cidx++) {
                    if (scatteredrows) {
                        rcol = ghost_bitmap_next(rhs->ldmask,rcol);
                    } else {
                        rcol++;
                    }
                    tmp[cidx] += matrixval * rhsrow[rcol];
                }
            }

            rhsrow = (v_t *)rhs->val[i];
            rcol = ghost_bitmap_first(rhs->ldmask)-1;
            lcol = ghost_bitmap_first(lhs->ldmask)-1;
            for (cidx = 0; cidx<rhs->traits.ncols; cidx++) {
                if (scatteredrows) {
                    rcol = ghost_bitmap_next(rhs->ldmask,rcol);
                    lcol = ghost_bitmap_next(lhs->ldmask,lcol);
                } else {
                    rcol++;
                    lcol++;
                }
                if ((options & GHOST_SPMV_SHIFT) && shift) {
                    tmp[cidx] = tmp[cidx]-shift[0]*rhsrow[rcol];
                }
                if ((options & GHOST_SPMV_VSHIFT) && shift) {
                    tmp[cidx] = tmp[cidx]-shift[cidx]*rhsrow[rcol];
                }
                if (options & GHOST_SPMV_SCALE) {
                    tmp[cidx] = tmp[cidx]*scale;
                }
                if (options & GHOST_SPMV_AXPY) {
                    lhsv[lcol] += tmp[cidx];
                } else if (options & GHOST_SPMV_AXPBY) {
                    lhsv[lcol] = beta*lhsv[lcol] + tmp[cidx];
                } else {
                    lhsv[lcol] = tmp[cidx];
                }
                if (options & GHOST_SPMV_DOT_ANY) {
                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*cidx+0] += conjugate(&lhsv[lcol])*lhsv[lcol];
                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*cidx+1] += conjugate(&lhsv[lcol])*rhsrow[rcol];
                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*cidx+2] += conjugate(&rhsrow[rcol])*rhsrow[rcol];
                }
            }
        }
        free(tmp);
    }
    if (options & GHOST_SPMV_DOT_ANY) {
        for (cidx=0; cidx<lhs->traits.ncols; cidx++) {
            local_dot_product[cidx                       ] = 0.; 
            local_dot_product[cidx  +   lhs->traits.ncols] = 0.;
            local_dot_product[cidx  + 2*lhs->traits.ncols] = 0.;
            for (i=0; i<nthreads; i++) {
                local_dot_product[cidx                       ] += partsums[(padding+3*lhs->traits.ncols)*i + 3*cidx + 0];
                local_dot_product[cidx  +   lhs->traits.ncols] += partsums[(padding+3*lhs->traits.ncols)*i + 3*cidx + 1];
                local_dot_product[cidx  + 2*lhs->traits.ncols] += partsums[(padding+3*lhs->traits.ncols)*i + 3*cidx + 2];
            }
        }
        free(partsums);
    }
    return GHOST_SUCCESS;
}
    
    template<typename m_t, typename v_t> 
static ghost_error_t ghost_crs_spmv_plain_rm_selector(ghost_sparsemat_t *mat, 
        ghost_densemat_t *lhs, ghost_densemat_t *rhs, 
        ghost_spmv_flags_t options, va_list argp)
{
    if (lhs->traits.ncolsorig != lhs->traits.ncols || 
            rhs->traits.ncolsorig != rhs->traits.ncols) {
        return ghost_crs_spmv_plain_rm<m_t,v_t,true>(mat,lhs,rhs,options,argp);
    } else {
        return ghost_crs_spmv_plain_rm<m_t,v_t,false>(mat,lhs,rhs,options,argp);
    }
}

template<typename m_t, typename v_t> 
static ghost_error_t ghost_crs_spmv_plain_cm(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp)
{
    ghost_crs_t *cr = CR(mat);
    v_t *rhsv = NULL;
    v_t *lhsv = NULL;
    v_t *local_dot_product = NULL, *partsums = NULL;
    m_t *mval = (m_t *)(cr->val);
    ghost_lidx_t i, j;
    ghost_lidx_t v;
    int nthreads = 1;

    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    int padding = (int)clsize/sizeof(v_t);

    v_t hlp1 = 0.;
    v_t scale = 1., beta = 1.;
    v_t *shift = NULL;

    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,v_t,v_t);
    
        
    if (options & GHOST_SPMV_DOT_ANY) {

#pragma omp parallel 
        {
#pragma omp single
            nthreads = ghost_omp_nthread();
        }

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,(3*lhs->traits.ncols+padding)*nthreads*sizeof(v_t))); 
        for (i=0; i<(3*lhs->traits.ncols+padding)*nthreads; i++) {
            partsums[i] = 0.;
        }
    }

#pragma omp parallel private (i,hlp1, j, rhsv, lhsv,v) shared (partsums)
    {
        int tid = ghost_omp_threadnum();
#pragma omp for schedule(runtime) 
        for (i=0; i<mat->nrows; i++){
            for (v=0; v<lhs->traits.ncols; v++)
            {
                rhsv = (v_t *)rhs->val[v];
                lhsv = (v_t *)lhs->val[v];
                hlp1 = (v_t)0.0;
                for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
                    hlp1 += ((v_t)(mval[j])) * rhsv[cr->col[j]];
                }

                if ((options & GHOST_SPMV_SHIFT) && shift) {
                    hlp1 = hlp1-shift[0]*rhsv[i];
                }
                if ((options & GHOST_SPMV_VSHIFT) && shift) {
                    hlp1 = hlp1-shift[v]*rhsv[i];
                }
                if (options & GHOST_SPMV_SCALE) {
                    hlp1 = hlp1*scale;
                }
                if (options & GHOST_SPMV_AXPY) {
                    lhsv[i] += (hlp1);
                } else if (options & GHOST_SPMV_AXPBY) {
                    lhsv[i] = beta*lhsv[i] + hlp1;
                } else {
                    lhsv[i] = (hlp1);
                }

                if (options & GHOST_SPMV_DOT_ANY) {
                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+0] += conjugate(&lhsv[i])*lhsv[i];
                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+1] += conjugate(&lhsv[i])*rhsv[i];
                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+2] += conjugate(&rhsv[i])*rhsv[i];
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
                local_dot_product[v                       ] += partsums[(padding+3*lhs->traits.ncols)*i + 3*v + 0];
                local_dot_product[v  +   lhs->traits.ncols] += partsums[(padding+3*lhs->traits.ncols)*i + 3*v + 1];
                local_dot_product[v  + 2*lhs->traits.ncols] += partsums[(padding+3*lhs->traits.ncols)*i + 3*v + 2];
            }
        }
        free(partsums);
    }
    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_crs_spmv_selector(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp)
{
    ghost_error_t ret;
   
    GHOST_DENSEMAT_CHECK_SIMILARITY(rhs,lhs);
    
    if (((rhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) && 
                (rhs->traits.nrowsorig != rhs->traits.nrows)) || 
            ((lhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) && 
            (lhs->traits.nrowsorig != lhs->traits.nrows))) {
        ERROR_LOG("Col-major densemats with masked out rows currently not "
                "supported!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    PERFWARNING_LOG("Using the CRS SpMV kernel which is potentially slower than "
            "auto-generated SELL SpMV kernels!");
        
    if (lhs->traits.storage == GHOST_DENSEMAT_COLMAJOR) {
        SELECT_TMPL_2DATATYPES(mat->traits->datatype,rhs->traits.datatype,ghost_complex,ret,ghost_crs_spmv_plain_cm,mat,lhs,rhs,options,argp);
    } else {
        SELECT_TMPL_2DATATYPES(mat->traits->datatype,rhs->traits.datatype,ghost_complex,ret,ghost_crs_spmv_plain_rm_selector,mat,lhs,rhs,options,argp);
    }


    return ret;
}

