#define _XOPEN_SOURCE 500 
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/vec.h"
#include "ghost/util.h"
#include "ghost/constants.h"
#include "ghost/affinity.h"
#include "ghost/context.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/io.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#ifdef GHOST_HAVE_CUDA
#include <cuda_runtime.h> // TODO in cu_util
#include <cublas_v2.h>
extern cublasHandle_t ghost_cublas_handle;
#endif

const ghost_vtraits_t GHOST_VTRAITS_INITIALIZER = {.flags = GHOST_VEC_DEFAULT, .aux = NULL, .datatype = GHOST_DT_DOUBLE|GHOST_DT_REAL, .nrows = 0, .nrowshalo = 0, .nrowspadded = 0, .nvecs = 1 };

ghost_error_t (*ghost_normalizeVector_funcs[4]) (ghost_vec_t *) = 
{&s_ghost_normalizeVector, &d_ghost_normalizeVector, &c_ghost_normalizeVector, &z_ghost_normalizeVector};

ghost_error_t (*ghost_vec_dotprod_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*) = 
{&s_ghost_vec_dotprod, &d_ghost_vec_dotprod, &c_ghost_vec_dotprod, &z_ghost_vec_dotprod};

ghost_error_t (*ghost_vec_vscale_funcs[4]) (ghost_vec_t *, void*) = 
{&s_ghost_vec_vscale, &d_ghost_vec_vscale, &c_ghost_vec_vscale, &z_ghost_vec_vscale};

ghost_error_t (*ghost_vec_vaxpy_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*) = 
{&s_ghost_vec_vaxpy, &d_ghost_vec_vaxpy, &c_ghost_vec_vaxpy, &z_ghost_vec_vaxpy};

ghost_error_t (*ghost_vec_vaxpby_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*, void*) = 
{&s_ghost_vec_vaxpby, &d_ghost_vec_vaxpby, &c_ghost_vec_vaxpby, &z_ghost_vec_vaxpby};

ghost_error_t (*ghost_vec_fromRand_funcs[4]) (ghost_vec_t *) = 
{&s_ghost_vec_fromRand, &d_ghost_vec_fromRand, &c_ghost_vec_fromRand, &z_ghost_vec_fromRand};

ghost_error_t (*ghost_vec_print_funcs[4]) (ghost_vec_t *) = 
{&s_ghost_printVector, &d_ghost_printVector, &c_ghost_printVector, &z_ghost_printVector};

static ghost_error_t vec_print(ghost_vec_t *vec);
static ghost_error_t vec_scale(ghost_vec_t *vec, void *scale);
static ghost_error_t vec_vscale(ghost_vec_t *vec, void *scale);
static ghost_error_t vec_vaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale);
static ghost_error_t vec_vaxpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b);
static ghost_error_t vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale);
static ghost_error_t vec_axpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b);
static ghost_error_t vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res);
static ghost_error_t vec_fromFunc(ghost_vec_t *vec, void (*fp)(int,int,void *));
static ghost_error_t vec_fromVec(ghost_vec_t *vec, ghost_vec_t *vec2, ghost_vidx_t coffs);
static ghost_error_t vec_fromRand(ghost_vec_t *vec);
static ghost_error_t vec_fromScalar(ghost_vec_t *vec, void *val);
static ghost_error_t vec_fromFile(ghost_vec_t *vec, char *path);
static ghost_error_t vec_toFile(ghost_vec_t *vec, char *path);
static ghost_error_t ghost_zeroVector(ghost_vec_t *vec);
static ghost_error_t ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2);
static ghost_error_t ghost_normalizeVector( ghost_vec_t *vec);
static ghost_error_t ghost_distributeVector(ghost_vec_t *vec, ghost_vec_t *nodeVec);
static ghost_error_t ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec); 
static void ghost_freeVector( ghost_vec_t* const vec );
static ghost_error_t ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm); 
static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src, ghost_vidx_t, ghost_vidx_t);
static ghost_error_t vec_entry(ghost_vec_t *, ghost_vidx_t, ghost_vidx_t, void *);
static ghost_vec_t * vec_view (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t coffs);
static ghost_vec_t * vec_viewScatteredVec (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t *coffs);
static ghost_error_t vec_viewPlain (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda);
static ghost_error_t vec_compress(ghost_vec_t *vec);
static ghost_error_t vec_upload(ghost_vec_t *vec);
static ghost_error_t vec_download(ghost_vec_t *vec);
static ghost_error_t vec_uploadHalo(ghost_vec_t *vec);
static ghost_error_t vec_downloadHalo(ghost_vec_t *vec);
static ghost_error_t vec_uploadNonHalo(ghost_vec_t *vec);
static ghost_error_t vec_downloadNonHalo(ghost_vec_t *vec);
static ghost_error_t getNrowsFromContext(ghost_vec_t *vec);
ghost_error_t ghost_vec_malloc(ghost_vec_t *vec);

ghost_error_t ghost_createVector(ghost_vec_t **vec, ghost_context_t *ctx, ghost_vtraits_t *traits)
{
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_vidx_t v;
    GHOST_CALL_GOTO(ghost_malloc((void **)vec,sizeof(ghost_vec_t)),err,ret);
    (*vec)->context = ctx;
    (*vec)->traits = traits;
    getNrowsFromContext((*vec));
    GHOST_CALL_GOTO(ghost_sizeofDatatype(&(*vec)->traits->elSize,(*vec)->traits->datatype),err,ret);

    DEBUG_LOG(1,"Initializing vector");

    if (!((*vec)->traits->flags & (GHOST_VEC_HOST | GHOST_VEC_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting vector placement");
        (*vec)->traits->flags |= GHOST_VEC_HOST;
        ghost_type_t ghost_type;
        GHOST_CALL_RETURN(ghost_getType(&ghost_type));
        if (ghost_type == GHOST_TYPE_CUDAMGMT) {
            (*vec)->traits->flags |= GHOST_VEC_DEVICE;
        }
    }

    if ((*vec)->traits->flags & GHOST_VEC_DEVICE)
    {
#if GHOST_HAVE_CUDA
        (*vec)->dotProduct = &ghost_vec_cu_dotprod;
        (*vec)->vaxpy = &ghost_vec_cu_vaxpy;
        (*vec)->vaxpby = &ghost_vec_cu_vaxpby;
        (*vec)->axpy = &ghost_vec_cu_axpy;
        (*vec)->axpby = &ghost_vec_cu_axpby;
        (*vec)->scale = &ghost_vec_cu_scale;
        (*vec)->vscale = &ghost_vec_cu_vscale;
        (*vec)->fromScalar = &ghost_vec_cu_fromScalar;
        (*vec)->fromRand = &ghost_vec_cu_fromRand;
#endif
    }
    else if ((*vec)->traits->flags & GHOST_VEC_HOST)
    {
        (*vec)->dotProduct = &vec_dotprod;
        (*vec)->vaxpy = &vec_vaxpy;
        (*vec)->vaxpby = &vec_vaxpby;
        (*vec)->axpy = &vec_axpy;
        (*vec)->axpby = &vec_axpby;
        (*vec)->scale = &vec_scale;
        (*vec)->vscale = &vec_vscale;
        (*vec)->fromScalar = &vec_fromScalar;
        (*vec)->fromRand = &vec_fromRand;
    }

    (*vec)->compress = &vec_compress;
    (*vec)->print = &vec_print;
    (*vec)->fromFunc = &vec_fromFunc;
    (*vec)->fromVec = &vec_fromVec;
    (*vec)->fromFile = &vec_fromFile;
    (*vec)->toFile = &vec_toFile;
    (*vec)->zero = &ghost_zeroVector;
    (*vec)->distribute = &ghost_distributeVector;
    (*vec)->collect = &ghost_collectVectors;
    (*vec)->swap = &ghost_swapVectors;
    (*vec)->normalize = &ghost_normalizeVector;
    (*vec)->destroy = &ghost_freeVector;
    (*vec)->permute = &ghost_permuteVector;
    (*vec)->clone = &ghost_cloneVector;
    (*vec)->entry = &vec_entry;
    (*vec)->viewVec = &vec_view;
    (*vec)->viewPlain = &vec_viewPlain;
    (*vec)->viewScatteredVec = &vec_viewScatteredVec;

    (*vec)->upload = &vec_upload;
    (*vec)->download = &vec_download;
    (*vec)->uploadHalo = &vec_uploadHalo;
    (*vec)->downloadHalo = &vec_downloadHalo;
    (*vec)->uploadNonHalo = &vec_uploadNonHalo;
    (*vec)->downloadNonHalo = &vec_downloadNonHalo;
#ifdef GHOST_HAVE_CUDA
    if ((*vec)->traits->flags & GHOST_VEC_DEVICE) {
        (*vec)->cu_val = NULL;
    }
#endif

    // TODO free val of vec only if scattered (but do not free val[0] of course!)
    GHOST_CALL_GOTO(ghost_malloc((void **)&(*vec)->val,(*vec)->traits->nvecs*sizeof(char *)),err,ret);

    for (v=0; v<(*vec)->traits->nvecs; v++) {
        (*vec)->val[v] = NULL;
    }

    goto out;
err:
    free(*vec); *vec = NULL;

out:
    return ret;
}

static ghost_error_t vec_uploadHalo(ghost_vec_t *vec)
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Uploading halo elements of vector");
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            ghost_cu_upload(CUVECVAL(vec,vec->cu_val,v,vec->traits->nrows),
                    VECVAL(vec,vec->val,v,vec->traits->nrows), 
                    vec->context->halo_elements*vec->traits->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_downloadHalo(ghost_vec_t *vec)
{

    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Downloading halo elements of vector");
        WARNING_LOG("Not yet implemented!");
    }
    return GHOST_SUCCESS;
}
static ghost_error_t vec_uploadNonHalo(ghost_vec_t *vec)
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Uploading %"PRvecIDX" rows of vector",vec->traits->nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            ghost_cu_upload(&vec->cu_val[vec->traits->nrowspadded*v*vec->traits->elSize],VECVAL(vec,vec->val,v,0), vec->traits->nrows*vec->traits->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_downloadNonHalo(ghost_vec_t *vec)
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Downloading vector");
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            ghost_cu_download(VECVAL(vec,vec->val,v,0),&vec->cu_val[vec->traits->nrowspadded*v*vec->traits->elSize],vec->traits->nrows*vec->traits->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_upload(ghost_vec_t *vec) 
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Uploading %"PRvecIDX" rows of vector",vec->traits->nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            ghost_cu_upload(&vec->cu_val[vec->traits->nrowspadded*v*vec->traits->elSize],VECVAL(vec,vec->val,v,0), vec->traits->nrowshalo*vec->traits->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_download(ghost_vec_t *vec)
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Downloading vector");
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            ghost_cu_download(VECVAL(vec,vec->val,v,0),&vec->cu_val[vec->traits->nrowspadded*v*vec->traits->elSize)],vec->traits->nrowshalo*vec->traits->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_vec_t * vec_view (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRvecIDX"x%"PRvecIDX" dense matrix with col offset %"PRvecIDX,src->traits->nrows,nc,coffs);
    ghost_vec_t *new;
    ghost_vtraits_t *newTraits;
    ghost_cloneVtraits(src->traits,&newTraits);
    newTraits->nvecs = nc;

    ghost_createVector(&new,src->context,newTraits);
    ghost_vidx_t v;

    for (v=0; v<new->traits->nvecs; v++) {
        new->val[v] = VECVAL(src,src->val,coffs+v,0);
    }

    new->traits->flags |= GHOST_VEC_VIEW;
    return new;
}

static ghost_error_t vec_viewPlain (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda)
{
    DEBUG_LOG(1,"Viewing a %"PRvecIDX"x%"PRvecIDX" dense matrix from plain data with offset %"PRvecIDX"x%"PRvecIDX,nr,nc,roffs,coffs);

    ghost_vidx_t v;

    for (v=0; v<vec->traits->nvecs; v++) {
        vec->val[v] = &((char *)data)[(lda*(coffs+v)+roffs)*vec->traits->elSize];
    }
    vec->traits->flags |= GHOST_VEC_VIEW;

    return GHOST_SUCCESS;
}

static ghost_vec_t* vec_viewScatteredVec (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t *coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRvecIDX"x%"PRvecIDX" scattered dense matrix",src->traits->nrows,nc);
    ghost_vec_t *new;
    ghost_vidx_t v;
    ghost_vtraits_t *newTraits;
    ghost_cloneVtraits(src->traits,&newTraits);
    newTraits->nvecs = nc;

    ghost_createVector(&new,src->context,newTraits);

    for (v=0; v<nc; v++) {
        new->val[v] = VECVAL(src,src->val,coffs[v],0);
    }    

    new->traits->flags |= GHOST_VEC_VIEW;
    new->traits->flags |= GHOST_VEC_SCATTERED;
    return new;
}

static ghost_error_t ghost_normalizeVector( ghost_vec_t *vec)
{
    ghost_normalizeVector_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_print(ghost_vec_t *vec)
{
    return ghost_vec_print_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec);

}

ghost_error_t ghost_vec_malloc(ghost_vec_t *vec)
{

    ghost_vidx_t v;
    if (vec->traits->flags & GHOST_VEC_HOST) {
        if (vec->val[0] == NULL) {
            DEBUG_LOG(2,"Allocating host side of vector");
            GHOST_CALL_RETURN(ghost_malloc_align((void **)&vec->val[0],vec->traits->nvecs*vec->traits->nrowspadded*vec->traits->elSize,GHOST_DATA_ALIGNMENT));
            for (v=1; v<vec->traits->nvecs; v++) {
                vec->val[v] = vec->val[0]+v*vec->traits->nrowspadded*vec->traits->elSize;
            }
        }
    }

    if (vec->traits->flags & GHOST_VEC_DEVICE) {
        DEBUG_LOG(2,"Allocating device side of vector");
#ifdef GHOST_HAVE_CUDA
        if (vec->cu_val == NULL) {
#ifdef GHOST_HAVE_CUDA_PINNEDMEM
            WARNING_LOG("CUDA pinned memory is disabled");
            //ghost_cu_safecall(cudaHostGetDevicePointer((void **)&vec->cu_val,vec->val,0));
            GHOST_CALL_RETURN(ghost_cu_malloc(&vec->cu_val,vec->traits->nrowspadded*vec->traits->nvecs*vec->traits->elSize));
#else
            //ghost_cu_safecall(cudaMallocPitch(&(void *)vec->cu_val,&vec->traits->nrowspadded,vec->traits->nrowshalo*sizeofdt,vec->traits->nvecs));
            GHOST_CALL_RETURN(ghost_cu_malloc(&vec->cu_val,vec->traits->nrowspadded*vec->traits->nvecs*vec->traits->elSize));
#endif
        }
#endif
    }   

    return GHOST_SUCCESS; 
}

static ghost_error_t getNrowsFromContext(ghost_vec_t *vec)
{
    DEBUG_LOG(1,"Computing the number of vector rows from the context");

    if (vec->context != NULL) {
        if (vec->traits->nrows == 0) {
            DEBUG_LOG(2,"nrows for vector not given. determining it from the context");
            if (vec->traits->flags & GHOST_VEC_DUMMY) {
                vec->traits->nrows = 0;
            } else if ((vec->context->flags & GHOST_CONTEXT_REDUNDANT) || (vec->traits->flags & GHOST_VEC_GLOBAL))
            {
                if (vec->traits->flags & GHOST_VEC_LHS) {
                    vec->traits->nrows = vec->context->gnrows;
                } else if (vec->traits->flags & GHOST_VEC_RHS) {
                    vec->traits->nrows = vec->context->gncols;
                }
            } 
            else 
            {
                int rank;
                GHOST_CALL_RETURN(ghost_getRank(vec->context->mpicomm,&rank));
                vec->traits->nrows = vec->context->lnrows[rank];
            }
        }
        if (vec->traits->nrowshalo == 0) {
            DEBUG_LOG(2,"nrowshalo for vector not given. determining it from the context");
            if (vec->traits->flags & GHOST_VEC_DUMMY) {
                vec->traits->nrowshalo = 0;
            } else if ((vec->context->flags & GHOST_CONTEXT_REDUNDANT) || (vec->traits->flags & GHOST_VEC_GLOBAL))
            {
                vec->traits->nrowshalo = vec->traits->nrows;
            } 
            else 
            {
                if (!(vec->traits->flags & GHOST_VEC_GLOBAL) && vec->traits->flags & GHOST_VEC_RHS) {
                    if (vec->context->halo_elements == -1) {
                        ERROR_LOG("You have to make sure to read in the matrix _before_ creating the right hand side vector in a distributed context! This is because we have to know the number of halo elements of the vector.");
                        return GHOST_ERR_UNKNOWN;
                    }
                    vec->traits->nrowshalo = vec->traits->nrows+vec->context->halo_elements;
                } else {
                    vec->traits->nrowshalo = vec->traits->nrows;
                }
            }    
        }
    } else {
        // the case context==NULL is allowed - the vector is local.
        DEBUG_LOG(1,"The vector's context is NULL.");
    }


    if (vec->traits->nrowspadded == 0) {
        DEBUG_LOG(2,"nrowspadded for vector not given. determining it from the context");
        vec->traits->nrowspadded = ghost_pad(MAX(vec->traits->nrowshalo,vec->traits->nrows),GHOST_PAD_MAX); // TODO needed?
    }
    DEBUG_LOG(1,"The vector has %"PRvecIDX" w/ %"PRvecIDX" halo elements (padded: %"PRvecIDX") rows",
            vec->traits->nrows,vec->traits->nrowshalo-vec->traits->nrows,vec->traits->nrowspadded);
    return GHOST_SUCCESS; 
}


static ghost_error_t vec_fromVec(ghost_vec_t *vec, ghost_vec_t *vec2, ghost_vidx_t coffs)
{
    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from vector w/ col offset %"PRvecIDX,coffs);
    ghost_vidx_t v;

    for (v=0; v<vec->traits->nvecs; v++) {
        if (vec->traits->flags & GHOST_VEC_DEVICE)
        {
            if (vec2->traits->flags & GHOST_VEC_DEVICE)
            {
#if GHOST_HAVE_CUDA
                ghost_cu_memcpy(CUVECVAL(vec,vec->cu_val,v,0),CUVECVAL(vec2,vec2->cu_val,coffs+v,0),vec->traits->nrows*vec->traits->elSize);
#endif
            }
            else
            {
#if GHOST_HAVE_CUDA
                ghost_cu_upload(CUVECVAL(vec,vec->cu_val,v,0),VECVAL(vec2,vec2->val,coffs+v,0),vec->traits->nrows*vec->traits->elSize);
#endif
            }
        }
        else
        {
            if (vec2->traits->flags & GHOST_VEC_DEVICE)
            {
#if GHOST_HAVE_CUDA
                ghost_cu_download(VECVAL(vec,vec->val,v,0),CUVECVAL(vec2,vec2->cu_val,coffs+v,0),vec->traits->nrows*vec->traits->elSize);
#endif
            }
            else
            {
                memcpy(VECVAL(vec,vec->val,v,0),VECVAL(vec2,vec2->val,coffs+v,0),vec->traits->nrows*vec->traits->elSize);
            }
        }

    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
    GHOST_INSTR_START(axpy);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_vidx_t nc = MIN(vec->traits->nvecs,vec2->traits->nvecs);
    char *s = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->traits->elSize),err,ret);

    ghost_vidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->traits->elSize],scale,vec->traits->elSize);
    }

    GHOST_CALL_GOTO(ghost_vec_vaxpy_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,s),err,ret);

    goto out;
err:

out:
    free(s);
    GHOST_INSTR_STOP(axpy);
    return ret;
}

static ghost_error_t vec_axpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *_b)
{
    GHOST_INSTR_START(axpby);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_vidx_t nc = MIN(vec->traits->nvecs,vec2->traits->nvecs);
    char *s = NULL;
    char *b = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->traits->elSize),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&b,nc*vec->traits->elSize),err,ret);

    ghost_vidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->traits->elSize],scale,vec->traits->elSize);
        memcpy(&b[i*vec->traits->elSize],_b,vec->traits->elSize);
    }
    GHOST_CALL_GOTO(ghost_vec_vaxpby_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,s,b),err,ret);

    goto out;
err:

out:
    free(s);
    free(b);
    GHOST_INSTR_STOP(axpby);
    return ret;
}

static ghost_error_t vec_vaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
    GHOST_INSTR_START(vaxpy);
    ghost_error_t ret = ghost_vec_vaxpy_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,scale);
    GHOST_INSTR_STOP(vaxpy);
    return ret;
}

static ghost_error_t vec_vaxpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b)
{
    GHOST_INSTR_START(vaxpby);
    ghost_error_t ret = ghost_vec_vaxpby_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,scale,b);
    GHOST_INSTR_STOP(vaxpby);
    return ret;
}

static ghost_error_t vec_scale(ghost_vec_t *vec, void *scale)
{
    GHOST_INSTR_START(scale);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_vidx_t nc = vec->traits->nvecs;
    char *s;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->traits->elSize),err,ret);

    ghost_vidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->traits->elSize],scale,vec->traits->elSize);
    }
    GHOST_CALL_GOTO(ghost_vec_vscale_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,s),err,ret);

    goto out;
err:

out:
    free(s);
    GHOST_INSTR_STOP(scale);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_vscale(ghost_vec_t *vec, void *scale)
{
    GHOST_INSTR_START(vscale);
    ghost_error_t ret = ghost_vec_vscale_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,scale);
    GHOST_INSTR_STOP(vscale);

    return ret;
}

static ghost_error_t vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{
    GHOST_INSTR_START(dot);
    ghost_error_t ret = ghost_vec_dotprod_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,res);
    GHOST_INSTR_STOP(dot);

    return ret;
}

static ghost_error_t vec_entry(ghost_vec_t * vec, ghost_vidx_t r, ghost_vidx_t c, void *val) 
{
    if (vec->traits->flags & GHOST_VEC_DEVICE)
    {
#if GHOST_HAVE_CUDA
        ghost_cu_download(val,&vec->cu_val[(c*vec->traits->nrowspadded+r)*vec->traits->elSize],vec->traits->elSize);
#endif
    }
    else if (vec->traits->flags & GHOST_VEC_HOST)
    {
        memcpy(val,VECVAL(vec,vec->val,c,r),vec->traits->elSize);
    }

    return GHOST_SUCCESS;
}

static ghost_error_t vec_fromRand(ghost_vec_t *vec)
{
    return ghost_vec_fromRand_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec);
}

static ghost_error_t vec_fromScalar(ghost_vec_t *vec, void *val)
{
    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from scalar value with %"PRvecIDX" rows",vec->traits->nrows);

    int i,v;
#pragma omp parallel for schedule(runtime) private(v)
    for (i=0; i<vec->traits->nrows; i++) {
        for (v=0; v<vec->traits->nvecs; v++) {
            memcpy(VECVAL(vec,vec->val,v,i),val,vec->traits->elSize);
        }
    }
    vec->upload(vec);

    return GHOST_SUCCESS;
}

static ghost_error_t vec_toFile(ghost_vec_t *vec, char *path)
{ // TODO two separate functions

#ifdef GHOST_HAVE_MPI
    int rank;
    GHOST_CALL_RETURN(ghost_getRank(vec->context->mpicomm,&rank));

    int32_t endianess = ghost_machineIsBigEndian();
    int32_t version = 1;
    int32_t order = GHOST_BINVEC_ORDER_COL_FIRST;
    int32_t datatype = vec->traits->datatype;
    int64_t nrows = (int64_t)vec->context->gnrows;
    int64_t ncols = (int64_t)vec->traits->nvecs;
    MPI_File fileh;
    MPI_Status status;
    MPI_CALL_RETURN(MPI_File_open(vec->context->mpicomm,path,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&fileh));

    if (rank == 0) 
    { // write header AND portion
        MPI_CALL_RETURN(MPI_File_write(fileh,&endianess,1,MPI_INT,&status));
        MPI_CALL_RETURN(MPI_File_write(fileh,&version,1,MPI_INT,&status));
        MPI_CALL_RETURN(MPI_File_write(fileh,&order,1,MPI_INT,&status));
        MPI_CALL_RETURN(MPI_File_write(fileh,&datatype,1,MPI_INT,&status));
        MPI_CALL_RETURN(MPI_File_write(fileh,&nrows,1,MPI_LONG_LONG,&status));
        MPI_CALL_RETURN(MPI_File_write(fileh,&ncols,1,MPI_LONG_LONG,&status));

    }    
    ghost_vidx_t v;
    ghost_mpi_datatype_t mpidt;
    GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits->datatype));
    MPI_CALL_RETURN(MPI_File_set_view(fileh,4*sizeof(int32_t)+2*sizeof(int64_t),mpidt,mpidt,"native",MPI_INFO_NULL));
    MPI_Offset fileoffset = vec->context->lfRow[rank];
    ghost_vidx_t vecoffset = 0;
    for (v=0; v<vec->traits->nvecs; v++) {
        char *val = NULL;
        int copied = 0;
        if (vec->traits->flags & GHOST_VEC_HOST)
        {
            vec->download(vec);
            val = VECVAL(vec,vec->val,v,0);
        }
        else if (vec->traits->flags & GHOST_VEC_DEVICE)
        {
#if GHOST_HAVE_CUDA
            GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits->nrows*vec->traits->elSize));
            copied = 1;
            ghost_cu_download(val,&vec->cu_val[v*vec->traits->nrowspadded*vec->traits->elSize],vec->traits->nrows*vec->traits->elSize);
#endif
        }
        MPI_CALL_RETURN(MPI_File_write_at(fileh,fileoffset,val,vec->traits->nrows,mpidt,&status));
        fileoffset += nrows;
        vecoffset += vec->traits->nrowspadded*vec->traits->elSize;
        if (copied)
            free(val);
    }
    MPI_CALL_RETURN(MPI_File_close(&fileh));


#else
    DEBUG_LOG(1,"Writing (local) vector to file %s",path);
    size_t ret;

    int32_t endianess = ghost_machineIsBigEndian();
    int32_t version = 1;
    int32_t order = GHOST_BINVEC_ORDER_COL_FIRST;
    int32_t datatype = vec->traits->datatype;
    int64_t nrows = (int64_t)vec->traits->nrows;
    int64_t ncols = (int64_t)vec->traits->nvecs;

    FILE *filed;

    if ((filed = fopen64(path, "w")) == NULL){
        ERROR_LOG("Could not open vector file %s: %s",path,strerror(errno));
        return GHOST_ERR_IO;
    }

    if ((ret = fwrite(&endianess,sizeof(endianess),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&version,sizeof(version),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&order,sizeof(order),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&datatype,sizeof(datatype),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&nrows,sizeof(nrows),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&ncols,sizeof(ncols),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }

    ghost_vidx_t v;
    for (v=0; v<vec->traits->nvecs; v++) {
        char *val = NULL;
        int copied = 0;
        if (vec->traits->flags & GHOST_VEC_HOST)
        {
            vec->download(vec);
            val = VECVAL(vec,vec->val,v,0);
        }
        else if (vec->traits->flags & GHOST_VEC_DEVICE)
        {
#if GHOST_HAVE_CUDA
            GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits->nrows*vec->traits->elSize));
            copied = 1;
            ghost_cu_download(val,&vec->cu_val[v*vec->traits->nrowspadded*vec->traits->elSize],vec->traits->nrows*vec->traits->elSize);
#endif
        }

        if ((ret = fwrite(val, vec->traits->elSize, vec->traits->nrows,filed)) != vec->traits->nrows) {
            ERROR_LOG("fwrite failed: %zu",ret);
            fclose(filed);
            if (copied) {
                free(val); val = NULL;
            }
            return GHOST_ERR_IO;
        }

        if (copied) {
            free(val);
        }
    }
    fclose(filed);
#endif

    return GHOST_SUCCESS;

}

static ghost_error_t vec_fromFile(ghost_vec_t *vec, char *path)
{
    int rank;
    GHOST_CALL_RETURN(ghost_getRank(vec->context->mpicomm,&rank));

    off_t offset;
    if ((vec->context == NULL) || !(vec->context->flags & GHOST_CONTEXT_DISTRIBUTED)) {
        offset = 0;
    } else {
        offset = vec->context->lfRow[rank];
    }

    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Reading vector from file %s",path);

    FILE *filed;
    size_t ret;

    if ((filed = fopen64(path, "r")) == NULL){
        ERROR_LOG("Could not open vector file %s: %s",path,strerror(errno));
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    int32_t endianess;
    int32_t version;
    int32_t order;
    int32_t datatype;

    int64_t nrows;
    int64_t ncols;


    if ((ret = fread(&endianess, sizeof(endianess), 1,filed)) != 1) {
        ERROR_LOG("fread failed: %zu",ret);
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    if (endianess != GHOST_BINCRS_LITTLE_ENDIAN) {
        ERROR_LOG("Cannot read big endian vectors");
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    if ((ret = fread(&version, sizeof(version), 1,filed)) != 1) {
        ERROR_LOG("fread failed: %zu",ret);
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    if (version != 1) {
        ERROR_LOG("Cannot read vector files with format != 1 (is %d)",version);
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    if ((ret = fread(&order, sizeof(order), 1,filed)) != 1) {
        ERROR_LOG("fread failed: %zu",ret);
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }
    // Order does not matter for vectors

    if ((ret = fread(&datatype, sizeof(datatype), 1,filed)) != 1) {
        ERROR_LOG("fread failed: %zu",ret);
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    if (datatype != vec->traits->datatype) {
        ERROR_LOG("The data types don't match! Cast-while-read is not yet implemented for vectors.");
        return GHOST_ERR_IO;
    }

    if ((ret = fread(&nrows, sizeof(nrows), 1,filed)) != 1) {
        ERROR_LOG("fread failed: %zu",ret);
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }
    // I will read as many rows as the vector has

    if ((ret = fread(&ncols, sizeof(ncols), 1,filed)) != 1) {
        ERROR_LOG("fread failed: %zu",ret);
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    int v;
    for (v=0; v<vec->traits->nvecs; v++) {
        if (fseeko(filed,offset*vec->traits->elSize,SEEK_CUR)) {
            ERROR_LOG("seek failed");
            vec->destroy(vec);
            return GHOST_ERR_IO;
        }
        if (vec->traits->flags & GHOST_VEC_HOST)
        {
            if ((ghost_midx_t)(ret = fread(VECVAL(vec,vec->val,v,0), vec->traits->elSize, vec->traits->nrows,filed)) != vec->traits->nrows) {
                ERROR_LOG("fread failed: %zu",ret);
                vec->destroy(vec);
                return GHOST_ERR_IO;
            }
            vec->upload(vec);
        }
        else if (vec->traits->flags & GHOST_VEC_DEVICE)
        {
#if GHOST_HAVE_CUDA
            GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits->nrows*vec->traits->elSize));
            if ((ret = fread(val, vec->traits->elSize, vec->traits->nrows,filed)) != vec->traits->nrows) {
                ERROR_LOG("fread failed: %zu",ret);
                vec->destroy(vec);
                return GHOST_ERR_IO;
            }
            ghost_cu_upload(&vec->cu_val[v*vec->traits->nrowspadded*vec->traits->elSize],val,vec->traits->nrows*vec->traits->elSize);
            free(val);
#endif
        }
        else
        {
            WARNING_LOG("Invalid vector placement, not writing vector");
            fclose(filed);
        }

    }

    fclose(filed);

    return GHOST_SUCCESS;

}

static ghost_error_t vec_fromFunc(ghost_vec_t *vec, void (*fp)(int,int,void *))
{
    int rank;
    GHOST_CALL_RETURN(ghost_getRank(vec->context->mpicomm,&rank));
    GHOST_CALL_RETURN(ghost_vec_malloc(vec));
    DEBUG_LOG(1,"Filling vector via function");

    int i,v;

    for (v=0; v<vec->traits->nvecs; v++) {
#pragma omp parallel for schedule(runtime) private(i)
        for (i=0; i<vec->traits->nrows; i++) {
            fp(vec->context->lfRow[rank]+i,v,VECVAL(vec,vec->val,v,i));
        }
    }

    vec->upload(vec);

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_zeroVector(ghost_vec_t *vec) 
{
    DEBUG_LOG(1,"Zeroing vector");
    ghost_vidx_t v;

    if (vec->traits->flags & GHOST_VEC_DEVICE)
    {
#if GHOST_HAVE_CUDA
        ghost_cu_memset(vec->cu_val,0,vec->traits->nrowspadded*vec->traits->nvecs*vec->traits->elSize);
#endif
    }
    if (vec->traits->flags & GHOST_VEC_HOST)
    {
        for (v=0; v<vec->traits->nvecs; v++) 
        {
            memset(VECVAL(vec,vec->val,v,0),0,vec->traits->nrowspadded*vec->traits->elSize);
        }
    } 

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_distributeVector(ghost_vec_t *vec, ghost_vec_t *nodeVec)
{
    DEBUG_LOG(1,"Distributing vector");
    int me;
    int nprocs;
    GHOST_CALL_RETURN(ghost_getRank(nodeVec->context->mpicomm,&me));
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(nodeVec->context->mpicomm,&nprocs));

    ghost_vidx_t c;
#ifdef GHOST_HAVE_MPI
    DEBUG_LOG(2,"Scattering global vector to local vectors");

    ghost_mpi_datatype_t mpidt;
    GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits->datatype));

    int i;

    MPI_Request req[vec->traits->nvecs*2*(nprocs-1)];
    MPI_Status stat[vec->traits->nvecs*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits->nvecs*2*(nprocs-1);i++) 
        req[i] = MPI_REQUEST_NULL;

    if (me != 0) {
        for (c=0; c<vec->traits->nvecs; c++) {
            MPI_CALL_RETURN(MPI_Irecv(nodeVec->val[c],nodeVec->context->lnrows[me],mpidt,0,me,nodeVec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits->nvecs; c++) {
            memcpy(nodeVec->val[c],vec->val[c],vec->traits->elSize*nodeVec->context->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Isend(VECVAL(vec,vec->val,c,nodeVec->context->lfRow[i]),nodeVec->context->lnrows[i],mpidt,i,i,nodeVec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else

    for (c=0; c<vec->traits->nvecs; c++) {
        memcpy(nodeVec->val[c],vec->val[c],vec->traits->nrows*vec->traits->elSize);
    }
    //    *nodeVec = vec->clone(vec);
#endif

    nodeVec->upload(nodeVec);

    DEBUG_LOG(1,"Vector distributed successfully");

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec) 
{
    ghost_vidx_t c;
#ifdef GHOST_HAVE_MPI
    int me;
    int nprocs;
    ghost_mpi_datatype_t mpidt;
    GHOST_CALL_RETURN(ghost_getRank(vec->context->mpicomm,&me));
    GHOST_CALL_RETURN(ghost_getNumberOfRanks(vec->context->mpicomm,&nprocs));
    GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits->datatype));

    if (vec->context != NULL)
        vec->permute(vec,vec->context->invRowPerm); 

    int i;

    MPI_Request req[vec->traits->nvecs*2*(nprocs-1)];
    MPI_Status stat[vec->traits->nvecs*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits->nvecs*2*(nprocs-1);i++) 
        req[i] = MPI_REQUEST_NULL;

    if (me != 0) {
        for (c=0; c<vec->traits->nvecs; c++) {
            MPI_CALL_RETURN(MPI_Isend(vec->val[c],vec->context->lnrows[me],mpidt,0,me,vec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits->nvecs; c++) {
            memcpy(totalVec->val[c],vec->val[c],vec->traits->elSize*vec->context->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Irecv(VECVAL(totalVec,totalVec->val,c,vec->context->lfRow[i]),vec->context->lnrows[i],mpidt,i,i,vec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else
    if (vec->context != NULL) {
        vec->permute(vec,vec->context->invRowPerm);
        for (c=0; c<vec->traits->nvecs; c++) {
            memcpy(totalVec->val[c],vec->val[c],totalVec->traits->nrows*vec->traits->elSize);
        }
    }
#endif

    return GHOST_SUCCESS;

}

static ghost_error_t ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2) 
{
    if ((v1->traits->nvecs > 1) || (v2->traits->nvecs > 1)) {
        ERROR_LOG("Multi-column vector swapping not yet implemented");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    if (!v1 || !v2) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    char *dtmp = NULL;

    dtmp = v1->val[0];
    v1->val[0] = v2->val[0];
    v2->val[0] = dtmp;
#ifdef GHOST_HAVE_CUDA
    dtmp = v1->cu_val;
    v1->cu_val = v2->cu_val;
    v2->cu_val = dtmp;
#endif

    return GHOST_SUCCESS;
}


static void ghost_freeVector( ghost_vec_t* vec ) 
{
    if( vec ) {
        if (!(vec->traits->flags & GHOST_VEC_VIEW)) {
            ghost_vidx_t v;
#ifdef GHOST_HAVE_CUDA_PINNEDMEM
            WARNING_LOG("CUDA pinned memory is disabled");
            /*if (vec->traits->flags & GHOST_VEC_DEVICE) {
              for (v=0; v<vec->traits->nvecs; v++) { 
              ghost_cu_safecall(cudaFreeHost(vec->val[v]));
              }
              }*/
            if (vec->traits->flags & GHOST_VEC_SCATTERED) {
                for (v=0; v<vec->traits->nvecs; v++) {
                    free(vec->val[v]);
                }
            }
            else {
                free(vec->val[0]);
            }
#else
            //note: a 'scattered' vector (one with non-constant stride) is
            //      always a view of some other (there is no method to                        
            //      construct it otherwise), but we check anyway in case
            //      the user has built his own funny vector in memory.
            if (vec->traits->flags & GHOST_VEC_SCATTERED) {
                for (v=0; v<vec->traits->nvecs; v++) {
                    free(vec->val[v]);
                }
            }
            else {
                free(vec->val[0]);
            }
#endif
#ifdef GHOST_HAVE_CUDA
            if (vec->traits->flags & GHOST_VEC_DEVICE) {
                ghost_cu_free(vec->cu_val);
            }
#endif
        }
        free(vec->val);
        free(vec);
        // TODO free traits ???
    }
}
static ghost_error_t ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm) 
{
    // TODO enhance performance
    /* permutes values in vector so that i-th entry is mapped to position perm[i] */
    ghost_midx_t i;
    ghost_vidx_t len = vec->traits->nrows, c;
    char* tmp = NULL;

    if (perm == NULL) {
        DEBUG_LOG(1,"Permutation vector is NULL, returning.");
        return GHOST_SUCCESS;
    } else {
        DEBUG_LOG(1,"Permuting vector");
    }


    for (c=0; c<vec->traits->nvecs; c++) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,vec->traits->elSize*len));
        for(i = 0; i < len; ++i) {
            if( perm[i] >= len ) {
                ERROR_LOG("Permutation index out of bounds: %"PRmatIDX" > %"PRmatIDX,perm[i],len);
                free(tmp);
                return GHOST_ERR_UNKNOWN;
            }

            memcpy(&tmp[vec->traits->elSize*perm[i]],VECVAL(vec,vec->val,c,i),vec->traits->elSize);
        }
        for(i=0; i < len; ++i) {
            memcpy(VECVAL(vec,vec->val,c,i),&tmp[vec->traits->elSize*i],vec->traits->elSize);
        }
        free(tmp);
    }

    return GHOST_SUCCESS;
}

static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t coffs)
{
    ghost_vec_t *new;
    ghost_vtraits_t *newTraits;
    ghost_cloneVtraits(src->traits,&newTraits);
    ghost_createVector(&new,src->context,newTraits);
    new->traits->nvecs = nc;

    // copy the data even if the input vector is itself a view
    // (bitwise NAND operation to unset the view flag if set)
    new->traits->flags &= ~GHOST_VEC_VIEW;

    new->fromVec(new,src,coffs);
    return new;
}

static ghost_error_t vec_compress(ghost_vec_t *vec)
{
    if (!(vec->traits->flags & GHOST_VEC_SCATTERED)) {
        return GHOST_SUCCESS;
    }

    ghost_vidx_t v,i;

    char *val = NULL;
    GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits->nrowspadded*vec->traits->nvecs*vec->traits->elSize));

#pragma omp parallel for schedule(runtime) private(v)
    for (i=0; i<vec->traits->nrowspadded; i++)
    {
        for (v=0; v<vec->traits->nvecs; v++)
        {
            val[(v*vec->traits->nrowspadded+i)*vec->traits->elSize] = 0;
        }
    }

    for (v=0; v<vec->traits->nvecs; v++)
    {
        memcpy(&val[(v*vec->traits->nrowspadded)*vec->traits->elSize],
                VECVAL(vec,vec->val,v,0),vec->traits->nrowspadded*vec->traits->elSize);

        if (!(vec->traits->flags & GHOST_VEC_VIEW))
        {
            free(vec->val[v]);
        }
        vec->val[v] = &val[(v*vec->traits->nrowspadded)*vec->traits->elSize];
    }

    vec->traits->flags &= ~GHOST_VEC_VIEW;
    vec->traits->flags &= ~GHOST_VEC_SCATTERED;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_cloneVtraits(ghost_vtraits_t *t1, ghost_vtraits_t **t2)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)t2,sizeof(ghost_vtraits_t)));
    memcpy(t2,t1,sizeof(ghost_vtraits_t));

    return GHOST_SUCCESS;
}
