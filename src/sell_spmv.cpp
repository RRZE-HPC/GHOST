#include "ghost/types.h"
#include "ghost/omp.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h> //mpi.h has to be included before stdio.h
#endif
#include <stdio.h>

#include "ghost/complex.h"
#include "ghost/util.h"
#include "ghost/crs.h"
#include "ghost/sell.h"
#include "ghost/densemat.h"
#include "ghost/math.h"
#include "ghost/log.h"
#include "ghost/machine.h"

#include "ghost/sell_kernel_avx_gen.h"
#include "ghost/sell_kernel_sse_gen.h"

#include <sstream>
#include <cstdlib>
#include <map>
#include <iostream>
#include <map>

using namespace std;

    template<typename m_t, typename v_t> 
static ghost_error_t SELL_kernel_plain_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    ghost_sell_t *sell = (ghost_sell_t *)(mat->data);
    v_t *rhsv = NULL;
    v_t *local_dot_product = NULL, *partsums = NULL;
    ghost_lidx_t i,j,c,col,colidx;
    ghost_lidx_t v;
    int nthreads = 1;
    int chunkHeight = sell->chunkHeight;

    unsigned clsize;
    ghost_machine_cacheline_size(&clsize);
    int padding = (int) clsize/sizeof(v_t);

    v_t scale = 1., beta = 1.;
    v_t *shift = NULL;
    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,v_t,v_t);

    if (options & GHOST_SPMV_DOT_ANY) {
#pragma omp parallel
        nthreads = ghost_omp_nthread();

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,(3*lhs->traits.ncols+padding)*nthreads*sizeof(v_t))); 
        for (i=0; i<(3*lhs->traits.ncols+padding)*nthreads; i++) {
            partsums[i] = 0.;
        }
    }
    if (rhs->traits.storage == GHOST_DENSEMAT_ROWMAJOR) {

#pragma omp parallel private(c,j,col,colidx,i,v) shared(partsums)
        {
            v_t **tmp;
            ghost_malloc((void **)&tmp,sizeof(v_t *)*chunkHeight);
            ghost_malloc((void **)&tmp[0],sizeof(v_t)*chunkHeight*rhs->traits.ncols);
            for (i=1; i<chunkHeight; i++) {
                tmp[i] = tmp[i-1] + rhs->traits.ncols;
            }
            v_t **lhsv = NULL;
            int tid = ghost_omp_threadnum();
            v_t * rhsrow;
            v_t matrixval;

#pragma omp for schedule(runtime) 
            for (c=0; c<mat->nrowsPadded/chunkHeight; c++) 
            { // loop over chunks
//                    rhsv = (v_t *)rhs->val[v];
                lhsv = (v_t **)&(lhs->val[c*chunkHeight]);

                for (i=0; i<chunkHeight; i++) {
                    for (col=0; col<rhs->traits.ncols; col++) {
                        tmp[i][col] = (v_t)0;
                    }
                }

                for (j=0; j<(sell->chunkStart[c+1]-sell->chunkStart[c])/chunkHeight; j++) 
                { // loop inside chunk
                    for (i=0; i<chunkHeight; i++) {
                        // INFO_LOG("%d: %f * %f",i,(v_t)(((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]), rhsv[sell->col[sell->chunkStart[c]+j*chunkHeight+i]]);
                        matrixval = (v_t)(((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]);
                        rhsrow = (v_t *)rhs->val[sell->col[sell->chunkStart[c]+j*chunkHeight+i]];

                        for (colidx=0, col=0; col<rhs->traits.ncolsorig; col++) {
                            if (ghost_bitmap_isset(rhs->ldmask,col)) {
                                tmp[i][colidx] +=  matrixval * rhsrow[col];
                                colidx++;
                            } 
                        }
                    }
                }
                
                for (i=0; i<chunkHeight; i++) {
                    rhsrow = (v_t *)rhs->val[c*chunkHeight+i];
                    if (c*chunkHeight+i < mat->nrows) {
                        for (colidx = 0, col=0; col<lhs->traits.ncolsorig; col++) {
                            if (ghost_bitmap_isset(lhs->ldmask,col)) {
                                if ((options & GHOST_SPMV_SHIFT) && shift) {
                                    tmp[i][colidx] = tmp[i][colidx]-shift[0]*rhsrow[col];
                                }
                                if ((options & GHOST_SPMV_VSHIFT) && shift) {
                                    tmp[i][colidx] = tmp[i][colidx]-shift[colidx]*rhsrow[col];
                                }
                                if (options & GHOST_SPMV_SCALE) {
                                    tmp[i][colidx] = tmp[i][colidx]*scale;
                                }
                                if (options & GHOST_SPMV_AXPY) {
                                    lhsv[i][col] += tmp[i][colidx];
                                } else if (options & GHOST_SPMV_AXPBY) {
                                    lhsv[i][col] = beta*lhsv[i][col] + tmp[i][colidx];
                                } else {
                                    lhsv[i][col] = tmp[i][colidx];
                                }

                                if (options & GHOST_SPMV_DOT_ANY) {
                                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*col+0] += conjugate(&lhsv[i][col])*lhsv[i][col];
                                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*col+1] += conjugate(&lhsv[i][col])*rhsrow[col];
                                    partsums[((padding+3*lhs->traits.ncols)*tid)+3*col+2] += conjugate(&rhsrow[col])*rhsrow[col];
                                }
                                colidx++;
                            }
                        }
                    }
                }

            }
        }
    } else {
#pragma omp parallel private(c,j,i,v) shared(partsums)
        {
            v_t *tmp = new v_t[chunkHeight];
            v_t *lhsv = NULL;
            int tid = ghost_omp_threadnum();


#pragma omp for schedule(runtime) 
            for (c=0; c<mat->nrowsPadded/chunkHeight; c++) 
            { // loop over chunks
                for (v=0; v<MIN(lhs->traits.ncols,rhs->traits.ncols); v++)
                {
                    rhsv = (v_t *)rhs->val[v];
                    lhsv = (v_t *)lhs->val[v];

                    for (i=0; i<chunkHeight; i++) {
                        tmp[i] = (v_t)0;
                    }

                    for (j=0; j<(sell->chunkStart[c+1]-sell->chunkStart[c])/chunkHeight; j++) 
                    { // loop inside chunk
                        for (i=0; i<chunkHeight; i++) {
                            // INFO_LOG("%d: %f * %f",i,(v_t)(((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]), rhsv[sell->col[sell->chunkStart[c]+j*chunkHeight+i]]);
                            tmp[i] += (v_t)(((m_t*)(sell->val))[sell->chunkStart[c]+j*chunkHeight+i]) * 
                                rhsv[sell->col[sell->chunkStart[c]+j*chunkHeight+i]];
                        }
                    }
                    for (i=0; i<chunkHeight; i++) {
                        if (c*chunkHeight+i < mat->nrows) {
                            if ((options & GHOST_SPMV_SHIFT) && shift) {
                                tmp[i] = tmp[i]-shift[0]*rhsv[c*chunkHeight+i];
                            }
                            if ((options & GHOST_SPMV_VSHIFT) && shift) {
                                tmp[i] = tmp[i]-shift[v]*rhsv[c*chunkHeight+i];
                            }
                            if (options & GHOST_SPMV_SCALE) {
                                tmp[i] = tmp[i]*scale;
                            }
                            if (options & GHOST_SPMV_AXPY) {
                                lhsv[c*chunkHeight+i] += tmp[i];
                            } else if (options & GHOST_SPMV_AXPBY) {
                                lhsv[c*chunkHeight+i] = beta*lhsv[c*chunkHeight+i] + tmp[i];
                            } else {
                                lhsv[c*chunkHeight+i] = tmp[i];
                            }

                            if (options & GHOST_SPMV_DOT_ANY) {
                                partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+0] += conjugate(&lhsv[c*chunkHeight+i])*lhsv[c*chunkHeight+i];
                                partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+1] += conjugate(&lhsv[c*chunkHeight+i])*rhsv[c*chunkHeight+i];
                                partsums[((padding+3*lhs->traits.ncols)*tid)+3*v+2] += conjugate(&rhsv[c*chunkHeight+i])*rhsv[c*chunkHeight+i];
                            }
                        }

                    }

                }
            }
            free(tmp);
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

template<typename m_t, typename v_t> 
static ghost_error_t SELL_kernel_plain_ELLPACK_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    DEBUG_LOG(2,"In plain ELLPACK (SELL) kernel");
    v_t *rhsv = NULL;
    v_t *lhsv = NULL;
    v_t *local_dot_product = NULL, *partsums = NULL;
    int nthreads = 1;
    ghost_lidx_t i,j;
    ghost_lidx_t v;
    v_t tmp;
    ghost_sell_t *sell = (ghost_sell_t *)(mat->data);
    m_t *sellv = (m_t*)(sell->val);

    v_t scale = 1., beta = 1.;
    v_t *shift = NULL;
    GHOST_SPMV_PARSE_ARGS(options,argp,scale,beta,shift,local_dot_product,v_t,v_t);

    if (options & GHOST_SPMV_DOT_ANY) {
#pragma omp parallel
        nthreads = ghost_omp_nthread();

        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,16*nthreads*sizeof(v_t)));

        for (i=0; i<16*lhs->traits.ncols*nthreads; i++) {
            partsums[i] = 0.;
        }
    }


#pragma omp parallel private(i,j,tmp,v)
    {
        int tid = ghost_omp_threadnum();
#pragma omp for schedule(runtime)
        for (i=0; i<mat->nrows; i++) 
        {
            for (v=0; v<MIN(lhs->traits.ncols,rhs->traits.ncols); v++)
            {
                rhsv = (v_t *)rhs->val[v];
                lhsv = (v_t *)lhs->val[v];
                tmp = (v_t)0;

                for (j=0; j<sell->rowLen[i]; j++) 
                {
                    //                INFO_LOG("%d: %f * %f",i,(v_t)sellv[mat->nrowsPadded*j+i], rhsv[sell->col[mat->nrowsPadded*j+i]]);
                    tmp += (v_t)sellv[mat->nrowsPadded*j+i] * rhsv[sell->col[mat->nrowsPadded*j+i]];
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
                    partsums[(v+tid*lhs->traits.ncols)*16 + 0] += conjugate(&lhsv[i])*lhsv[i];
                    partsums[(v+tid*lhs->traits.ncols)*16 + 1] += conjugate(&lhsv[i])*rhsv[i];
                    partsums[(v+tid*lhs->traits.ncols)*16 + 2] += conjugate(&rhsv[i])*rhsv[i];
                }
            }
        }
    }
    if (options & GHOST_SPMV_DOT_ANY) {
        for (v=0; v<MIN(lhs->traits.ncols,rhs->traits.ncols); v++) {
            local_dot_product[v                       ] = 0.; 
            local_dot_product[v  +   lhs->traits.ncols] = 0.;
            local_dot_product[v  + 2*lhs->traits.ncols] = 0.;
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                       ] += partsums[(v+i*lhs->traits.ncols)*16 + 0];
                local_dot_product[v +   lhs->traits.ncols] += partsums[(v+i*lhs->traits.ncols)*16 + 1];
                local_dot_product[v + 2*lhs->traits.ncols] += partsums[(v+i*lhs->traits.ncols)*16 + 2];
            }
        }
        free(partsums);
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

static bool operator<(const ghost_sellspmv_parameters_t &a, const ghost_sellspmv_parameters_t &b) 
{ 
    return ghost_hash(ghost_hash(a.mdt,a.blocksz,a.storage),ghost_hash(a.vdt,a.impl,a.chunkheight),0) < ghost_hash(ghost_hash(b.mdt,b.blocksz,b.storage),ghost_hash(b.vdt,b.impl,b.chunkheight),0); 
}

static map<ghost_sellspmv_parameters_t, sellspmv_kernel> ghost_sellspmv_kernels = map<ghost_sellspmv_parameters_t, sellspmv_kernel>();

extern "C" ghost_error_t ghost_sell_spmv_selector(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options, va_list argp)
{
    // if map is empty include generated code for map construction
    if (ghost_sellspmv_kernels.empty()) {
#include "sell_kernel_avx.def"
#include "sell_kernel_sse.def"
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


    if (p.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz == 1 && rhs->traits.ncolsorig == 1 && lhs->traits.ncolsorig== 1) {
        INFO_LOG("Chose col-major kernel for row-major densemat with 1 column");
        p.storage = GHOST_DENSEMAT_COLMAJOR;
    }

    if (p.impl == GHOST_IMPLEMENTATION_AVX && p.storage == GHOST_DENSEMAT_ROWMAJOR && p.blocksz <= 2 && !(rhs->traits.datatype & GHOST_DT_COMPLEX)) {
        INFO_LOG("Chose SSE over AVX for blocksz=2");
        p.impl = GHOST_IMPLEMENTATION_SSE;
    }
    
    if (p.impl == GHOST_IMPLEMENTATION_AVX && p.storage == GHOST_DENSEMAT_COLMAJOR && p.chunkheight < 4 && !(rhs->traits.datatype & GHOST_DT_COMPLEX)) {
        if (p.chunkheight < 2) {
            INFO_LOG("Chose plain kernel for col-major densemats and C<2");
            p.impl = GHOST_IMPLEMENTATION_PLAIN;
        } else {
            INFO_LOG("Chose SSE for col-major densemats and C<4");
            p.impl = GHOST_IMPLEMENTATION_SSE;
        }
    }
    
    if (lhs->traits.flags & GHOST_DENSEMAT_SCATTERED || rhs->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        INFO_LOG("Use plain implementation for scattered views");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }
    
    if (lhs->traits.flags & GHOST_DENSEMAT_VIEW || rhs->traits.flags & GHOST_DENSEMAT_VIEW) {
        INFO_LOG("Use plain implementation for views. This is subject to be fixed, i.e., the vectorized kernels should work with dense views.");
        p.impl = GHOST_IMPLEMENTATION_PLAIN;
    }

    sellspmv_kernel kernel = ghost_sellspmv_kernels[p];

    if (!kernel) {
        INFO_LOG("Try kernel with arbitrary blocksz");
        p.blocksz = -1;
    }
    kernel = ghost_sellspmv_kernels[p];

    if (kernel) {
        kernel(mat,lhs,rhs,options,argp);
    } else { // fallback
        if (SELL(mat)->chunkHeight == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
            SELECT_TMPL_2DATATYPES(mat->traits->datatype,rhs->traits.datatype,ghost_complex,ret,SELL_kernel_plain_ELLPACK_tmpl,mat,lhs,rhs,options,argp);
        } else {
            SELECT_TMPL_2DATATYPES(mat->traits->datatype,rhs->traits.datatype,ghost_complex,ret,SELL_kernel_plain_tmpl,mat,lhs,rhs,options,argp);
        }
    } 


    return ret;
}

