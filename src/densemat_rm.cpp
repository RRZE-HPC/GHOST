#include "ghost/config.h"
#include "ghost/omp.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h> //mpi.h has to be included before stdio.h
#endif
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <complex>
#include <stdio.h>

#include "ghost/complex.h"
#include "ghost/rand.h"
#include "ghost/util.h"
#include "ghost/densemat_rm.h"
#include "ghost/math.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include "ghost/machine.h"


using namespace std;


template <typename v_t> 
static ghost_error_t ghost_densemat_rm_normalize_tmpl(ghost_densemat_t *vec)
{
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_lidx_t v;
    v_t *s = NULL;

    GHOST_CALL_GOTO(ghost_malloc((void **)&s,vec->traits.ncols*sizeof(v_t)),err,ret);
    GHOST_CALL_GOTO(ghost_dot(s,vec,vec),err,ret);

    for (v=0; v<vec->traits.ncols; v++)
    {
        s[v] = (v_t)sqrt(s[v]);
        s[v] = (v_t)(((v_t)1.)/s[v]);
    }
    GHOST_CALL_GOTO(vec->vscale(vec,s),err,ret);

    goto out;
err:

out:
    free(s);

    return ret;

}

template <typename v_t> 
static ghost_error_t ghost_densemat_rm_dotprod_tmpl(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
{ // the parallelization is done manually because reduction does not work with ghost_complex numbers
    if (vec->traits.nrows != vec2->traits.nrows) {
        WARNING_LOG("The input vectors of the dot product have different numbers of rows");
    }
    if (vec->traits.ncols != vec2->traits.ncols) {
        WARNING_LOG("The input vectors of the dot product have different numbers of columns");
    }
    ghost_lidx_t i,v,vidx;
    ghost_lidx_t nr = MIN(vec->traits.nrows,vec2->traits.nrows);

    int nthreads;
#pragma omp parallel
#pragma omp single
    nthreads = ghost_omp_nthread();


    if (!ghost_bitmap_iscompact(vec->ldmask) || !ghost_bitmap_iscompact(vec2->ldmask)) {
        WARNING_LOG("Potentially slow DOT operation because some rows are masked out!");
        v_t *partsums;
        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,16*nthreads*sizeof(v_t)));
        ITER_COLS_BEGIN(vec,v,vidx)
            v_t sum = 0;
            for (i=0; i<nthreads*16; i++) partsums[i] = (v_t)0.;

#pragma omp parallel 
            {
                int tid = ghost_omp_threadnum();
#pragma omp for schedule(runtime)
                for (i=0; i<nr; i++) {
                    partsums[tid*16] += 
                        *(v_t *)VECVAL_RM(vec2,vec2->val,i,v)*
                        conjugate((v_t *)(VECVAL_RM(vec,vec->val,i,v)));
                }
            }

            for (i=0; i<nthreads; i++) {
                sum += partsums[i*16];
            }
            ((v_t *)res)[vidx] = sum;
        ITER_COLS_END(vidx)
        free(partsums);
    } else {
        unsigned clsize;
        ghost_machine_cacheline_size(&clsize);
        int padding = (int)clsize/sizeof(v_t);
        v_t *partsums;
        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,nthreads*(vec->traits.ncols+padding)*sizeof(v_t)));
        memset(partsums,0,nthreads*(vec->traits.ncols+padding)*sizeof(v_t));

#pragma omp parallel 
        {
            int tid = ghost_omp_threadnum();
            ghost_lidx_t col1,col2,row,colidx;
            ITER2_COMPACT_BEGIN_RM_INPAR(vec,vec2,col1,col2,row,colidx)
                partsums[(padding+vec->traits.ncols)*tid+colidx] += *(v_t *)VECVAL_RM(vec2,vec2->val,row,col2)*
                    conjugate((v_t *)(VECVAL_RM(vec,vec->val,row,col1)));

            ITER2_COMPACT_END_RM()

        }
        ghost_lidx_t col;
        for (col=0; col<vec->traits.ncols; col++) {
            ((v_t *)res)[col] = 0.;

            for (i=0; i<nthreads; i++) {
                ((v_t *)res)[col] += partsums[i*(vec->traits.ncols+padding)+col];
            }
        }

        free(partsums);
    }
    
    return GHOST_SUCCESS;
}

template <typename v_t> 
static ghost_error_t ghost_densemat_rm_vaxpy_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale)
{
    if (vec->traits.storage != vec2->traits.storage) {
        ERROR_LOG("Cannot VAXPY densemats with different storage order");
        return GHOST_ERR_INVALID_ARG;
    }
    if (vec->traits.ncols != vec2->traits.ncols) {
        ERROR_LOG("Cannot VAXPY densemats with different number of columns");
        return GHOST_ERR_INVALID_ARG;
    }
    if (vec->traits.nrows != vec2->traits.nrows) {
        ERROR_LOG("Cannot VAXPY densemats with different number of rows");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_lidx_t col1,col2,row,colidx;
    v_t *s = (v_t *)scale;
    
    if (!ghost_bitmap_iscompact(vec->ldmask) || !ghost_bitmap_iscompact(vec2->ldmask)) {
        WARNING_LOG("Potentially slow VAXPY operation because some rows are masked out!");
        ITER2_BEGIN_RM(vec,vec2,col1,col2,row,colidx)
            *(v_t *)VECVAL_RM(vec,vec->val,row,col1) += *(v_t *)VECVAL_RM(vec2,vec2->val,row,col2) * s[colidx];
        ITER2_END_RM(colidx)
    } else {
        ITER2_COMPACT_BEGIN_RM(vec,vec2,col1,col2,row,colidx)
            *(v_t *)VECVAL_RM(vec,vec->val,row,col1) += *(v_t *)VECVAL_RM(vec2,vec2->val,row,col2) * s[colidx];
        ITER2_COMPACT_END_RM()
    }


    return GHOST_SUCCESS;
}

template <typename v_t> 
static ghost_error_t ghost_densemat_rm_vaxpby_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b_)
{
    if (vec->traits.storage != vec2->traits.storage) {
        ERROR_LOG("Cannot VAXPBY densemats with different storage order");
        return GHOST_ERR_INVALID_ARG;
    }
    if (vec->traits.ncols != vec2->traits.ncols) {
        ERROR_LOG("Cannot VAXPBY densemats with different number of columns");
        return GHOST_ERR_INVALID_ARG;
    }
    if (vec->traits.nrows != vec2->traits.nrows) {
        ERROR_LOG("Cannot VAXPBY densemats with different number of rows");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_lidx_t col1,col2,row,colidx;
    v_t *s = (v_t *)scale;
    v_t *b = (v_t *)b_;
    
    if (!ghost_bitmap_iscompact(vec->ldmask) || !ghost_bitmap_iscompact(vec2->ldmask)) {
        WARNING_LOG("Potentially slow VAXPBY operation because some rows are masked out!");
        ITER2_BEGIN_RM(vec,vec2,col1,col2,row,colidx)
            *(v_t *)VECVAL_RM(vec,vec->val,row,col1) = *(v_t *)VECVAL_RM(vec2,vec2->val,row,col2) * s[colidx] + 
                *(v_t *)VECVAL_RM(vec,vec->val,row,col1) * b[colidx];
        ITER2_END_RM(colidx)
    } else {
        ITER2_COMPACT_BEGIN_RM(vec,vec2,col1,col2,row,colidx)
            *(v_t *)VECVAL_RM(vec,vec->val,row,col1) = *(v_t *)VECVAL_RM(vec2,vec2->val,row,col2) * s[colidx] + 
                *(v_t *)VECVAL_RM(vec,vec->val,row,col1) * b[colidx];
        ITER2_COMPACT_END_RM()
    }

    return GHOST_SUCCESS;
}

template<typename v_t> 
static ghost_error_t ghost_densemat_rm_vscale_tmpl(ghost_densemat_t *vec, void *scale)
{
    ghost_lidx_t col,row,colidx;
    v_t *s = (v_t *)scale;

    if (!ghost_bitmap_iscompact(vec->ldmask)) {
        WARNING_LOG("Potentially slow SCAL operation because some rows are masked out!");
        ITER_BEGIN_RM(vec,col,row,colidx)
                *(v_t *)VECVAL_RM(vec,vec->val,row,col) *= s[colidx];
        ITER_END_RM(colidx)
    } else {
        ITER_COMPACT_BEGIN_RM(vec,col,row,colidx)
                *(v_t *)VECVAL_RM(vec,vec->val,row,col) *= s[colidx];
        ITER_COMPACT_END_RM()
    }

    return GHOST_SUCCESS;
}

// thread-safe type generic random function, returns pseudo-random numbers between -1 and 1.
template <typename v_t>
static inline void my_rand(unsigned int* state, v_t* result)
{
    // default implementation
    static const v_t scal = (v_t)2.0/(v_t)RAND_MAX;
    static const v_t shift=(v_t)(-1.0);
    *result=(v_t)rand_r(state)*scal+shift;
}

template <typename float_type>
static void my_rand(unsigned int* state, std::complex<float_type>* result)
{
    float_type* ft_res = (float_type*)result;
    my_rand(state,&ft_res[0]);
    my_rand(state,&ft_res[1]);
}

template <typename float_type>
static inline void my_rand(unsigned int* state, ghost_complex<float_type>* result)
{
    my_rand<float_type>(state,(float_type *)result);
    my_rand<float_type>(state,((float_type *)result)+1);
}



template <typename v_t> 
static ghost_error_t ghost_densemat_rm_fromRand_tmpl(ghost_densemat_t *vec)
{
    ghost_densemat_rm_malloc(vec);
    DEBUG_LOG(1,"Filling vector with random values");

    if (ghost_bitmap_weight(vec->ldmask) != vec->traits.ncolsorig || 
            ghost_bitmap_weight(vec->trmask) != vec->traits.nrowsorig) {
        WARNING_LOG("Potentially slow fromRand operation because some rows or columns are masked out!");
#pragma omp parallel
        {
            ghost_lidx_t col,row,colidx;
            unsigned int *state;
            ghost_rand_get(&state);
            ITER_BEGIN_RM_INPAR(vec,col,row,colidx)
            my_rand(state,(v_t *)VECVAL_RM(vec,vec->val,row,col));
            ITER_END_RM(colidx)
        }
    } else {
#pragma omp parallel
        {
            unsigned int *state;
            ghost_rand_get(&state);
            ghost_lidx_t i;
#pragma omp for
            for (i=0; i<vec->traits.ncolspadded*vec->traits.nrows; i++) {
                my_rand(state,&((v_t *)(vec->val[0]))[i]);
            }
        }
    }

    vec->upload(vec);

    return GHOST_SUCCESS;
}


template <typename v_t> 
static ghost_error_t ghost_densemat_rm_string_tmpl(char **str, ghost_densemat_t *vec)
{
    stringstream buffer;
    buffer << std::setprecision(6)
           << std::right
           << std::scientific
           << std::setw(10);

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t i,v,r,j;
        for (i=0,r=0; i<vec->traits.nrowsorig; i++) {
            if (ghost_bitmap_isset(vec->trmask,i)) {
                for (j=0,v=0; j<vec->traits.ncolsorig; j++) {
                    if (ghost_bitmap_isset(vec->ldmask,j)) {
                        v_t val = 0.;
                        GHOST_CALL_RETURN(ghost_cu_download(&val,&(((v_t *)vec->cu_val)[i*vec->traits.ncolspadded+j]),sizeof(v_t)));
                        buffer << val;
                        v++;
                    }
                }
                if (r<vec->traits.nrows-1) {
                    buffer << std::endl;
                }
                r++;
            }
        }
#endif
    } else {
        ghost_lidx_t i,v,r;
        for (r=0; r<vec->traits.nrows; r++) {
            for (i=0, v=0; v<vec->traits.ncolsorig; v++) {
                if (ghost_bitmap_isset(vec->ldmask,v)) {
                    if (vec->traits.datatype & GHOST_DT_COMPLEX) {
                        if (vec->traits.datatype & GHOST_DT_DOUBLE) {
                            double *val;
                            val = (double *)VECVAL_RM(vec,vec->val,r,v);
                            buffer << "(" << *val << ", " << *(val+1) << ")\t";
                        } else {
                            float *val;
                            val = (float *)VECVAL_RM(vec,vec->val,r,v);
                            buffer << "(" << *val << ", " << *(val+1) << ")\t";
                        }
                    } else {
                        v_t val = *(v_t *)VECVAL_RM(vec,vec->val,r,v);
                        buffer << val << "\t";
                    }
                    i++;
                }
            }
            if (r<vec->traits.nrows-1) {
                buffer << std::endl;
            }
        }
    }

#if 0
    ghost_lidx_t i,r;
    for (r=0; r<vec->traits.nrows; r++) {
        for (i=0; i<vec->traits.ncolsorig; i++) {
            if (ghost_bitmap_isset(vec->ldmask,i)) {
                v_t val = 0.;
                if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
                {
#ifdef GHOST_HAVE_CUDA
                    GHOST_CALL_RETURN(ghost_cu_download(&val,&(((v_t *)vec->cu_val)[r*vec->traits.ncolspadded+i]),sizeof(v_t)));
#endif
                }
                else if (vec->traits.flags & GHOST_DENSEMAT_HOST)
                {
                    val = *(v_t *)VECVAL_RM(vec,vec->val,r,i);
                }
                buffer << val << "\t";
            }
        }
        if (r<vec->traits.nrows-1) {
            buffer << std::endl;
        }
    }
#endif

    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    return GHOST_SUCCESS;
}


extern "C" ghost_error_t ghost_densemat_rm_string_selector(ghost_densemat_t *vec, char **str) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_rm_string_tmpl,str,vec);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_normalize_selector(ghost_densemat_t *vec) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_rm_normalize_tmpl,vec);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_dotprod_selector(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2) 
{
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_rm_dotprod_tmpl,vec,res,vec2);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_vscale_selector(ghost_densemat_t *vec, void *scale) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_rm_vscale_tmpl,vec,scale);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_vaxpy_selector(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_rm_vaxpy_tmpl,vec,vec2,scale);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_vaxpby_selector(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_rm_vaxpby_tmpl,vec,vec2,scale,b);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_rm_fromRand_selector(ghost_densemat_t *vec) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_rm_fromRand_tmpl,vec);

    return ret;
}

