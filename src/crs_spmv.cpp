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

template<typename m_t, typename v_t> 
static ghost_error_t CRS_kernel_plain_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp)
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
    if (rhs->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {
#pragma omp parallel private (i, j, lhsv,v) shared (partsums)
        {
            v_t matrixval;
            v_t * rhsrow;
            ghost_lidx_t colidx;
            v_t *tmp;
            ghost_malloc((void **)&tmp,rhs->traits.ncols*sizeof(v_t));
            int tid = ghost_omp_threadnum();
#pragma omp for schedule(runtime) 
            for (i=0; i<mat->nrows; i++) {
                //for (v=0; v<MIN(lhs->traits.ncols,rhs->traits.ncols); v++)
                {
                    lhsv = (v_t *)lhs->val[i];

                    for (v=0; v<rhs->traits.ncols; v++) {
                        tmp[v] = 0.;
                    }
                    if (rhs->traits.flags & GHOST_DENSEMAT_SCATTERED) {
                        for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
                            matrixval = ((v_t)(mval[j]));
                            rhsrow = (v_t *)rhs->val[cr->col[j]];
                            for (colidx=0, v=0; v<rhs->traits.ncolsorig; v++) {
                                if (ghost_bitmap_isset(rhs->ldmask,v)) {
                                    tmp[colidx] += matrixval * rhsrow[v];
                                    colidx++;
                                }
                            }
                        }
                    } else {
                        for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
                            matrixval = ((v_t)(mval[j]));
                            rhsrow = (v_t *)rhs->val[cr->col[j]];
                            for (v=0; v<rhs->traits.ncols; v++) {
                                tmp[v] += matrixval * rhsrow[ghost_bitmap_first(rhs->ldmask)+v];
                            }
                        }
                    }

                    rhsrow = (v_t *)rhs->val[i];
                    for (colidx=0, v=0; v<lhs->traits.ncolsorig; v++) {
                        if (ghost_bitmap_isset(lhs->ldmask,v)) {
                            if ((options & GHOST_SPMV_SHIFT) && shift) {
                                tmp[colidx] = tmp[colidx]-shift[0]*rhsrow[v];
                            }
                            if ((options & GHOST_SPMV_VSHIFT) && shift) {
                                tmp[colidx] = tmp[colidx]-shift[v]*rhsrow[v];
                            }
                            if (options & GHOST_SPMV_SCALE) {
                                tmp[colidx] = tmp[colidx]*scale;
                            }
                            if (options & GHOST_SPMV_AXPY) {
                                lhsv[v] += tmp[colidx];
                            } else if (options & GHOST_SPMV_AXPBY) {
                                lhsv[v] = beta*lhsv[v] + tmp[colidx];
                            } else {
                                lhsv[v] = tmp[colidx];
                            }
                            if (options & GHOST_SPMV_DOT_ANY) {
                                partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+0] += conjugate(&lhsv[v])*lhsv[v];
                                partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+1] += conjugate(&lhsv[v])*rhsrow[v];
                                partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+2] += conjugate(&rhsrow[v])*rhsrow[v];
                            }
                            colidx++;
                        }
                    }
                }
            }
            free(tmp);
        }
    } else {

#pragma omp parallel private (i,hlp1, j, rhsv, lhsv,v) shared (partsums)
        {
            int tid = ghost_omp_threadnum();
#pragma omp for schedule(runtime) 
            for (i=0; i<mat->nrows; i++){
                for (v=0; v<MIN(lhs->traits.ncols,rhs->traits.ncols); v++)
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
    }
    if (options & GHOST_SPMV_DOT_ANY) {
        if (!local_dot_product) {
            WARNING_LOG("The location of the local dot products is NULL. Will not compute them!");
            return GHOST_SUCCESS;
        }
        for (v=0; v<MIN(lhs->traits.ncols,rhs->traits.ncols); v++) {
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

    SELECT_TMPL_2DATATYPES(mat->traits->datatype,rhs->traits.datatype,ghost_complex,ret,CRS_kernel_plain_tmpl,mat,lhs,rhs,options,argp);

    return ret;
}

