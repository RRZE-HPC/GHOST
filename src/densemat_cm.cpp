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
#include "ghost/densemat_cm.h"
#include "ghost/math.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#define COLMAJOR
#include "ghost/densemat_iter_macros.h"


using namespace std;


template <typename v_t> 
static ghost_error_t ghost_densemat_cm_normalize_tmpl(ghost_densemat_t *vec)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;

}


template <typename v_t> 
static ghost_error_t ghost_densemat_cm_dotprod_tmpl(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
{ 
    // the parallelization is done manually because reduction does not work with ghost_complex numbers
   
    GHOST_DENSEMAT_CHECK_SIMILARITY(vec,vec2);
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);

    ghost_error_t ret = GHOST_SUCCESS;
    int nthreads, i;
    v_t *partsums;
    unsigned clsize;

#pragma omp parallel
#pragma omp single
    nthreads = ghost_omp_nthread();
   
    ghost_machine_cacheline_size(&clsize);
    int padding = (int)clsize/sizeof(v_t);
        
    GHOST_CALL_GOTO(ghost_malloc((void **)&partsums,nthreads*(vec->traits.ncols+padding)*sizeof(v_t)),err,ret);
    memset(partsums,0,nthreads*(vec->traits.ncols+padding)*sizeof(v_t));
    
#pragma omp parallel shared(partsums) 
    {
        int tid = ghost_omp_threadnum();
        DENSEMAT_ITER2(vec,vec2,
            partsums[(padding+vec->traits.ncols)*tid+col] += *(v_t *)DENSEMAT_VAL(vec2,memrow2,col)*
                conjugate((v_t *)(DENSEMAT_VAL(vec,memrow1,col))));
    }
    ghost_lidx_t col;
    for (col=0; col<vec->traits.ncols; col++) {
        ((v_t *)res)[col] = 0.;

        for (i=0; i<nthreads; i++) {
            ((v_t *)res)[col] += partsums[i*(vec->traits.ncols+padding)+col];
        }
    }

    goto out;
err:

out:
    free(partsums);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return ret;
}

template <typename v_t> 
static ghost_error_t ghost_densemat_cm_vaxpy_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale)
{
    GHOST_DENSEMAT_CHECK_SIMILARITY(vec,vec2);
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    
    v_t *s = (v_t *)scale;
    
    DENSEMAT_ITER2(vec,vec2,
            *(v_t *)DENSEMAT_VAL(vec,memrow1,col) += 
            *(v_t *)DENSEMAT_VAL(vec2,memrow2,col) * s[col]);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

template <typename v_t> 
static ghost_error_t ghost_densemat_cm_vaxpby_tmpl(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b_)
{
    GHOST_DENSEMAT_CHECK_SIMILARITY(vec,vec2);
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);

    v_t *s = (v_t *)scale;
    v_t *b = (v_t *)b_;

    DENSEMAT_ITER2(vec,vec2,
            *(v_t *)DENSEMAT_VAL(vec,memrow1,col) = 
            *(v_t *)DENSEMAT_VAL(vec2,memrow2,col) * s[col] + 
                *(v_t *)DENSEMAT_VAL(vec,row,memcol1) * b[col]);
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

template<typename v_t> 
static ghost_error_t ghost_densemat_cm_vscale_tmpl(ghost_densemat_t *vec, void *scale)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
        
    DENSEMAT_ITER(vec,*(v_t *)DENSEMAT_VAL(vec,memrow,col) *= ((v_t *)scale)[col]);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH|GHOST_FUNCTYPE_KERNEL);
    return GHOST_SUCCESS;
}

// thread-safe type generic random function, returns pseudo-random numbers between -1 and 1.
template <typename v_t>
static void my_rand(unsigned int* state, v_t* result)
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
static void my_rand(unsigned int* state, ghost_complex<float_type>* result)
{
    my_rand<float_type>(state,(float_type *)result);
    my_rand<float_type>(state,((float_type *)result)+1);
}



template <typename v_t> 
static ghost_error_t ghost_densemat_cm_fromRand_tmpl(ghost_densemat_t *vec)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    
    ghost_densemat_cm_malloc(vec);

#pragma omp parallel
    {
        unsigned int *state;
        ghost_rand_get(&state);
        DENSEMAT_ITER(vec,my_rand(state,(v_t *)DENSEMAT_VAL(vec,memrow,col)));
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return GHOST_SUCCESS;
}

template <typename v_t> 
static ghost_error_t ghost_densemat_cm_string_tmpl(char **str, ghost_densemat_t *vec)
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
            if (ghost_bitmap_isset(vec->ldmask,i)) {
                for (j=0,v=0; j<vec->traits.ncolsorig; j++) {
                    if (ghost_bitmap_isset(vec->trmask,j)) {
                        v_t val = 0.;
                        //printf("dl col %d row %d idx %d cu_val %p val %p\n",j,i,j*vec->traits.nrowspadded+i,&(((v_t *)vec->cu_val)[j*vec->traits.nrowspadded+i]),&val);
                        GHOST_CALL_RETURN(ghost_cu_download(&val,&(((v_t *)vec->cu_val)[j*vec->traits.nrowspadded+i]),sizeof(v_t)));
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
        for (i=0,r=0; i<vec->traits.nrowsorig; i++) {
            if (ghost_bitmap_isset(vec->ldmask,i)) {
                for (v=0; v<vec->traits.ncols; v++) {
                    if (vec->traits.datatype & GHOST_DT_COMPLEX) {
                        if (vec->traits.datatype & GHOST_DT_DOUBLE) {
                            double *val;
                            val = (double *)DENSEMAT_VAL(vec,i,v);
                            buffer << "(" << *val << ", " << *(val+1) << ")\t";
                        } else {
                            float *val;
                            val = (float *)DENSEMAT_VAL(vec,i,v);
                            buffer << "(" << *val << ", " << *(val+1) << ")\t";
                        }
                    } else {
                        v_t val = *(v_t *)DENSEMAT_VAL(vec,i,v);
                        buffer << val << "\t";
                    }

                }
                if (r<vec->traits.nrows-1) {
                    buffer << std::endl;
                }
                r++;
            }
        }
    }
 
    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str, buffer.str().c_str());

    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_densemat_cm_string_selector(ghost_densemat_t *vec, char **str) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_cm_string_tmpl,str,vec);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_normalize_selector(ghost_densemat_t *vec) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_cm_normalize_tmpl,vec);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_dotprod_selector(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2) 
{
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_cm_dotprod_tmpl,vec,res,vec2);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_vscale_selector(ghost_densemat_t *vec, void *scale) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_cm_vscale_tmpl,vec,scale);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_vaxpy_selector(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_cm_vaxpy_tmpl,vec,vec2,scale);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_vaxpby_selector(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_cm_vaxpby_tmpl,vec,vec2,scale,b);

    return ret;
}

extern "C" ghost_error_t ghost_densemat_cm_fromRand_selector(ghost_densemat_t *vec) 
{ 
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,ghost_complex,ret,ghost_densemat_cm_fromRand_tmpl,vec);

    return ret;
}

