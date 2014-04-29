#define _XOPEN_SOURCE 500 
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/densemat_cm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/context.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/bindensemat.h"
#include "ghost/densemat_rm.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

static ghost_error_t (*ghost_densemat_cm_normalize_funcs[4]) (ghost_densemat_t *) = 
{&s_ghost_densemat_cm_normalize, &d_ghost_densemat_cm_normalize, &c_ghost_densemat_cm_normalize, &z_ghost_densemat_cm_normalize};

static ghost_error_t (*ghost_densemat_cm_dotprod_funcs[4]) (ghost_densemat_t *, void *, ghost_densemat_t *) = 
{&s_ghost_densemat_cm_dotprod, &d_ghost_densemat_cm_dotprod, &c_ghost_densemat_cm_dotprod, &z_ghost_densemat_cm_dotprod};

static ghost_error_t (*ghost_densemat_cm_vscale_funcs[4]) (ghost_densemat_t *, void*) = 
{&s_ghost_densemat_cm_vscale, &d_ghost_densemat_cm_vscale, &c_ghost_densemat_cm_vscale, &z_ghost_densemat_cm_vscale};

static ghost_error_t (*ghost_densemat_cm_vaxpy_funcs[4]) (ghost_densemat_t *, ghost_densemat_t *, void*) = 
{&s_ghost_densemat_cm_vaxpy, &d_ghost_densemat_cm_vaxpy, &c_ghost_densemat_cm_vaxpy, &z_ghost_densemat_cm_vaxpy};

static ghost_error_t (*ghost_densemat_cm_vaxpby_funcs[4]) (ghost_densemat_t *, ghost_densemat_t *, void*, void*) = 
{&s_ghost_densemat_cm_vaxpby, &d_ghost_densemat_cm_vaxpby, &c_ghost_densemat_cm_vaxpby, &z_ghost_densemat_cm_vaxpby};

static ghost_error_t (*ghost_densemat_cm_fromRand_funcs[4]) (ghost_densemat_t *) = 
{&s_ghost_densemat_cm_fromRand, &d_ghost_densemat_cm_fromRand, &c_ghost_densemat_cm_fromRand, &z_ghost_densemat_cm_fromRand};

static ghost_error_t (*ghost_densemat_cm_string_funcs[4]) (char **str, ghost_densemat_t *) = 
{&s_ghost_densemat_cm_string, &d_ghost_densemat_cm_string, &c_ghost_densemat_cm_string, &z_ghost_densemat_cm_string};

static ghost_error_t vec_cm_print(ghost_densemat_t *vec, char **str);
static ghost_error_t vec_cm_scale(ghost_densemat_t *vec, void *scale);
static ghost_error_t vec_cm_vscale(ghost_densemat_t *vec, void *scale);
static ghost_error_t vec_cm_vaxpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale);
static ghost_error_t vec_cm_vaxpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b);
static ghost_error_t vec_cm_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale);
static ghost_error_t vec_cm_axpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b);
static ghost_error_t vec_cm_dotprod(ghost_densemat_t *vec, void * res, ghost_densemat_t *vec2);
static ghost_error_t vec_cm_fromFunc(ghost_densemat_t *vec, void (*fp)(ghost_idx_t, ghost_idx_t, void *));
static ghost_error_t vec_cm_fromVec(ghost_densemat_t *vec, ghost_densemat_t *vec2, ghost_idx_t roffs, ghost_idx_t coffs);
static ghost_error_t vec_cm_fromRand(ghost_densemat_t *vec);
static ghost_error_t vec_cm_fromScalar(ghost_densemat_t *vec, void *val);
static ghost_error_t vec_cm_fromFile(ghost_densemat_t *vec, char *path);
static ghost_error_t vec_cm_toFile(ghost_densemat_t *vec, char *path);
static ghost_error_t ghost_densemat_cm_normalize( ghost_densemat_t *vec);
static ghost_error_t ghost_distributeVector(ghost_densemat_t *vec, ghost_densemat_t *nodeVec);
static ghost_error_t ghost_collectVectors(ghost_densemat_t *vec, ghost_densemat_t *totalVec); 
static void ghost_freeVector( ghost_densemat_t* const vec );
static ghost_error_t ghost_permuteVector( ghost_densemat_t* vec, ghost_permutation_t *permutation, ghost_permutation_direction_t dir); 
static ghost_error_t ghost_cloneVector(ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nr, ghost_idx_t roffs, ghost_idx_t nc, ghost_idx_t coffs);
static ghost_error_t vec_cm_entry(ghost_densemat_t *, void *, ghost_idx_t, ghost_idx_t);
static ghost_error_t vec_cm_view (ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nr, ghost_idx_t roffs, ghost_idx_t nc, ghost_idx_t coffs);
static ghost_error_t vec_cm_viewScatteredVec (ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nr, ghost_idx_t *roffs, ghost_idx_t nc, ghost_idx_t *coffs);
static ghost_error_t vec_cm_viewScatteredCols (ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nc, ghost_idx_t *coffs);
static ghost_error_t vec_cm_viewCols (ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nc, ghost_idx_t coffs);
static ghost_error_t vec_cm_viewPlain (ghost_densemat_t *vec, void *data, ghost_idx_t nr, ghost_idx_t nc, ghost_idx_t roffs, ghost_idx_t coffs, ghost_idx_t lda);
static ghost_error_t vec_cm_compress(ghost_densemat_t *vec);
static ghost_error_t vec_cm_upload(ghost_densemat_t *vec);
static ghost_error_t vec_cm_download(ghost_densemat_t *vec);
static ghost_error_t vec_cm_uploadHalo(ghost_densemat_t *vec);
static ghost_error_t vec_cm_downloadHalo(ghost_densemat_t *vec);
static ghost_error_t vec_cm_uploadNonHalo(ghost_densemat_t *vec);
static ghost_error_t vec_cm_downloadNonHalo(ghost_densemat_t *vec);
static ghost_error_t vec_cm_memtranspose(ghost_densemat_t *vec);
ghost_error_t ghost_densemat_cm_malloc(ghost_densemat_t *vec);

ghost_error_t ghost_densemat_cm_setfuncs(ghost_densemat_t *vec)
{
    ghost_error_t ret = GHOST_SUCCESS;

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
    {
#ifdef GHOST_HAVE_CUDA
        vec->dot = &ghost_densemat_cm_cu_dotprod;
        vec->vaxpy = &ghost_densemat_cm_cu_vaxpy;
        vec->vaxpby = &ghost_densemat_cm_cu_vaxpby;
        vec->axpy = &ghost_densemat_cm_cu_axpy;
        vec->axpby = &ghost_densemat_cm_cu_axpby;
        vec->scale = &ghost_densemat_cm_cu_scale;
        vec->vscale = &ghost_densemat_cm_cu_vscale;
        vec->fromScalar = &ghost_densemat_cm_cu_fromScalar;
        vec->fromRand = &ghost_densemat_cm_cu_fromRand;
#endif
    }
    else if (vec->traits.flags & GHOST_DENSEMAT_HOST)
    {
        vec->dot = &vec_cm_dotprod;
        vec->vaxpy = &vec_cm_vaxpy;
        vec->vaxpby = &vec_cm_vaxpby;
        vec->axpy = &vec_cm_axpy;
        vec->axpby = &vec_cm_axpby;
        vec->scale = &vec_cm_scale;
        vec->vscale = &vec_cm_vscale;
        vec->fromScalar = &vec_cm_fromScalar;
        vec->fromRand = &vec_cm_fromRand;
    }

    vec->memtranspose = &vec_cm_memtranspose;
    vec->compress = &vec_cm_compress;
    vec->string = &vec_cm_print;
    vec->fromFunc = &vec_cm_fromFunc;
    vec->fromVec = &vec_cm_fromVec;
    vec->fromFile = &vec_cm_fromFile;
    vec->toFile = &vec_cm_toFile;
    vec->distribute = &ghost_distributeVector;
    vec->collect = &ghost_collectVectors;
    vec->normalize = &ghost_densemat_cm_normalize;
    vec->destroy = &ghost_freeVector;
    vec->permute = &ghost_permuteVector;
    vec->clone = &ghost_cloneVector;
    vec->entry = &vec_cm_entry;
    vec->viewVec = &vec_cm_view;
    vec->viewPlain = &vec_cm_viewPlain;
    vec->viewScatteredVec = &vec_cm_viewScatteredVec;
    vec->viewScatteredCols = &vec_cm_viewScatteredCols;
    vec->viewCols = &vec_cm_viewCols;

    vec->upload = &vec_cm_upload;
    vec->download = &vec_cm_download;
    vec->uploadHalo = &vec_cm_uploadHalo;
    vec->downloadHalo = &vec_cm_downloadHalo;
    vec->uploadNonHalo = &vec_cm_uploadNonHalo;
    vec->downloadNonHalo = &vec_cm_downloadNonHalo;
#ifdef GHOST_HAVE_CUDA
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        vec->cu_val = NULL;
    }
#endif

    return ret;
}

static ghost_error_t vec_cm_memtranspose(ghost_densemat_t *vec)
{
    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        ERROR_LOG("Cannot memtranspose scattered densemat views!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
        ERROR_LOG("Memtranspose of densemat views currently broken");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    ghost_idx_t col,row;
    
    vec->traits.storage = GHOST_DENSEMAT_ROWMAJOR;
    ghost_densemat_rm_setfuncs(vec);
    char *oldval = vec->val[0];
    free(vec->val); 
    vec->val = NULL;
    GHOST_CALL_RETURN(ghost_malloc((void **)&vec->val,vec->traits.nrowspadded*sizeof(char *)));
    vec->val[0] = oldval;
    for (row=1; row<vec->traits.nrowspadded; row++) {
        vec->val[row] = vec->val[0]+vec->traits.ncolspadded*row*vec->elSize;
    }

    char *tmp;
    GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,vec->elSize*vec->traits.nrowspadded*vec->traits.ncolspadded));
    memcpy(tmp,vec->val[0],vec->elSize*vec->traits.nrowspadded*vec->traits.ncolspadded);

    for (col=0; col<vec->traits.ncols; col++) {
        for (row=0; row<vec->traits.nrows; row++) {
            memcpy(vec->val[0]+row*vec->traits.ncolspadded*vec->elSize+col*vec->elSize,
                    tmp+vec->traits.nrowspadded*col*vec->elSize+row*vec->elSize,
                    vec->elSize);
        }
    }

    free(tmp);

    return GHOST_SUCCESS; 
}

static ghost_error_t vec_cm_uploadHalo(ghost_densemat_t *vec)
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Uploading halo elements of vector");
#ifdef GHOST_HAVE_CUDA
        ghost_idx_t v;
        for (v=0; v<vec->traits.ncols; v++) {
            ghost_cu_upload(CUVECVAL_CM(vec,vec->cu_val,v,vec->traits.nrows),
                    VECVAL_CM(vec,vec->val,v,vec->traits.nrows), 
                    vec->context->halo_elements*vec->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_downloadHalo(ghost_densemat_t *vec)
{

    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Downloading halo elements of vector");
        WARNING_LOG("Not yet implemented!");
    }
    return GHOST_SUCCESS;
}
static ghost_error_t vec_cm_uploadNonHalo(ghost_densemat_t *vec)
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Uploading %"PRIDX" rows of vector",vec->traits.nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_idx_t v;
        for (v=0; v<vec->traits.ncols; v++) {
            ghost_cu_upload(&vec->cu_val[vec->traits.nrowspadded*v*vec->elSize],VECVAL_CM(vec,vec->val,v,0), vec->traits.nrows*vec->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_downloadNonHalo(ghost_densemat_t *vec)
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Downloading vector");
#ifdef GHOST_HAVE_CUDA
        ghost_idx_t v;
        for (v=0; v<vec->traits.ncols; v++) {
            ghost_cu_download(VECVAL_CM(vec,vec->val,v,0),&vec->cu_val[vec->traits.nrowspadded*v*vec->elSize],vec->traits.nrows*vec->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_upload(ghost_densemat_t *vec) 
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Uploading %"PRIDX" rows of vector",vec->traits.nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_idx_t v,c,r;
        for (v=0, c=0; v<vec->traits.ncolsorig; v++) {
            if (hwloc_bitmap_isset(vec->trmask,v)) {
                for (r=0; r<vec->traits.nrowsorig; r++) {
                    if (hwloc_bitmap_isset(vec->ldmask,r)) {
                        ghost_cu_upload(&vec->cu_val[(vec->traits.nrowspadded*v+r)*vec->elSize],VECVAL_CM(vec,vec->val,c,r), vec->elSize);
                    }
                }
                c++;
            }
        }
#endif
    }
    vec->uploadHalo(vec);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_download(ghost_densemat_t *vec)
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Downloading %"PRIDX" rows of vector",vec->traits.nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_idx_t v,c,r;
        for (v=0, c=0; v<vec->traits.ncolsorig; v++) {
            if (hwloc_bitmap_isset(vec->trmask,v)) {
                for (r=0; r<vec->traits.nrowsorig; r++) {
                    if (hwloc_bitmap_isset(vec->ldmask,r)) {
                        ghost_cu_download(VECVAL_CM(vec,vec->val,c,r),&vec->cu_val[(vec->traits.nrowspadded*v+r)*vec->elSize], vec->elSize);
                    }
                }
                c++;
            }
        }
#endif
    }
    vec->downloadHalo(vec);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_view (ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nr, ghost_idx_t roffs, ghost_idx_t nc, ghost_idx_t coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRIDX"x%"PRIDX" densemat from a %"PRIDX"x%"PRIDX" densemat with offset %"PRIDX"x%"PRIDX,nr,nc,src->traits.nrows,src->traits.ncols,roffs,coffs);
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc;
    newTraits.nrows = nr;
    newTraits.flags |= GHOST_DENSEMAT_VIEW;

    ghost_densemat_create(new,src->context,newTraits);
    hwloc_bitmap_copy((*new)->ldmask,src->ldmask);
    ghost_densemat_cm_malloc(*new);
    ghost_idx_t v,r,viewedrow;

    //char *bm;
    //hwloc_bitmap_list_asprintf(&bm,src->ldmask);
    //WARNING_LOG("%s %p",bm,src->ldmask);

    //roffs += hwloc_bitmap_first((*new)->ldmask);
    
    for (viewedrow=0, r=0; r<src->traits.nrowsorig; r++) {
        //INFO_LOG("to view: %d..%d, current viewed row: %d, bitmap[%d] %d",roffs,roffs+nr-1,viewedrow,r,hwloc_bitmap_isset(src->ldmask,r));
        if (viewedrow<roffs || (viewedrow >= roffs+nr)) {
            //INFO_LOG("clr");
            hwloc_bitmap_clr((*new)->ldmask,r);
        }
        if (hwloc_bitmap_isset(src->ldmask,r)) {
            viewedrow++;
        }
    }
    //hwloc_bitmap_list_asprintf(&bm,(*new)->ldmask);
    //WARNING_LOG("%s %p",bm,(*new)->ldmask);

    if ((*new)->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        (*new)->cu_val = src->cu_val;
        for (v=0; v<src->traits.ncolsorig; v++) {
            if (v<coffs || (v >= coffs+nc)) {
                hwloc_bitmap_clr((*new)->trmask,v);
            }
        }
#endif
    } 
    if ((*new)->traits.flags & GHOST_DENSEMAT_HOST) {
        for (v=0; v<(*new)->traits.ncols; v++) {
            (*new)->val[v] = VECVAL_CM(src,src->val,coffs+v,0);
        }
    }
    //char *newstr;
    //(*new)->string(*new,&newstr);
    //printf("\n\n\n%s\n\n\n",newstr);

    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_viewPlain (ghost_densemat_t *vec, void *data, ghost_idx_t nr, ghost_idx_t nc, ghost_idx_t roffs, ghost_idx_t coffs, ghost_idx_t lda)
{
    DEBUG_LOG(1,"Viewing a %"PRIDX"x%"PRIDX" dense matrix from plain data with offset %"PRIDX"x%"PRIDX,nr,nc,roffs,coffs);
    ghost_densemat_cm_malloc(vec);

    ghost_idx_t v;

    for (v=0; v<vec->traits.ncols; v++) {
        vec->val[v] = &((char *)data)[(lda*(coffs+v)+roffs)*vec->elSize];
    }
    vec->traits.flags |= GHOST_DENSEMAT_VIEW;

    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_viewCols (ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nc, ghost_idx_t coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRIDX"x%"PRIDX" scattered dense matrix",src->traits.nrows,nc);
    ghost_idx_t v;
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc;
    newTraits.flags |= GHOST_DENSEMAT_VIEW;

    ghost_densemat_create(new,src->context,newTraits);
    ghost_densemat_cm_malloc(*new);

    if ((*new)->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        (*new)->cu_val = src->cu_val;
        for (v=0; v<src->traits.ncolsorig; v++) {
            if (v<coffs || (v >= coffs+nc)) {
                hwloc_bitmap_clr((*new)->trmask,v);
                //WARNING_LOG("clr %d",v);
            }
        }
#endif
    } 
    if ((*new)->traits.flags & GHOST_DENSEMAT_HOST) {
        for (v=0; v<(*new)->traits.ncols; v++) {
            (*new)->val[v] = VECVAL_CM(src,src->val,coffs+v,0);
        }
    }

    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_viewScatteredCols (ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nc, ghost_idx_t *coffs)
{
#ifdef GHOST_HAVE_CUDA
    if (src->traits.flags & GHOST_DENSEMAT_DEVICE) {
        if (!array_strictly_ascending(coffs,nc)) {
            ERROR_LOG("Can only view sctrictly ascending scattered columns for row-major densemats!");
            return GHOST_ERR_INVALID_ARG;
        }
    }
#endif
    DEBUG_LOG(1,"Viewing a %"PRIDX"x%"PRIDX" scattered dense matrix",src->traits.nrows,nc);
    ghost_idx_t v;
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc;
    newTraits.flags |= GHOST_DENSEMAT_VIEW;
    newTraits.flags |= GHOST_DENSEMAT_SCATTERED;

    ghost_densemat_create(new,src->context,newTraits);
    ghost_densemat_cm_malloc(*new);

    if ((*new)->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        ghost_idx_t c;
        (*new)->cu_val = src->cu_val;
        for (c=0,v=0; c<(*new)->traits.ncolsorig; c++) {
            if (coffs[v] != c) {
                hwloc_bitmap_clr((*new)->trmask,c);
            } else {
                v++;
            }
        }
#endif
    } 
    if ((*new)->traits.flags & GHOST_DENSEMAT_HOST) {
        for (v=0; v<nc; v++) {
            (*new)->val[v] = VECVAL_CM(src,src->val,coffs[v],0);
        }    
    }

    return GHOST_SUCCESS;
}


static ghost_error_t vec_cm_viewScatteredVec (ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nr, ghost_idx_t *roffs, ghost_idx_t nc, ghost_idx_t *coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRIDX"x%"PRIDX" scattered dense matrix",src->traits.nrows,nc);
    ghost_idx_t v,r,i,viewedrow;
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc;
    newTraits.nrows = nr;
    newTraits.flags |= GHOST_DENSEMAT_VIEW;
    newTraits.flags |= GHOST_DENSEMAT_SCATTERED;

    ghost_densemat_create(new,src->context,newTraits);
    hwloc_bitmap_copy((*new)->ldmask,src->ldmask);
    ghost_densemat_cm_malloc(*new);
    for (viewedrow=-1,r=0,i=0; r<(*new)->traits.nrowsorig; r++) {
        if (hwloc_bitmap_isset(src->ldmask,r)) {
            viewedrow++;
        }
        if (roffs[i] != viewedrow) {
            hwloc_bitmap_clr((*new)->ldmask,r);
        } else {
            i++;
        }
    }


    if ((*new)->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        ghost_idx_t c;
        (*new)->cu_val = src->cu_val;
        for (c=0,v=0; c<(*new)->traits.ncolsorig; c++) {
            if (coffs[v] != c) {
                hwloc_bitmap_clr((*new)->trmask,c);
            } else {
                v++;
            }
        }
#endif
    } 
    if ((*new)->traits.flags & GHOST_DENSEMAT_HOST) {
        for (v=0; v<nc; v++) {
            (*new)->val[v] = VECVAL_CM(src,src->val,coffs[v],0);
            INFO_LOG("val[%d] = %d = %p",v,coffs[v],(*new)->val[v]);
        }    
    }
    return GHOST_SUCCESS;
}

static ghost_error_t ghost_densemat_cm_normalize( ghost_densemat_t *vec)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,vec->traits.datatype));
    ghost_densemat_cm_normalize_funcs[dtIdx](vec);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_print(ghost_densemat_t *vec, char **str)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,vec->traits.datatype));
    return ghost_densemat_cm_string_funcs[dtIdx](str,vec);
}

ghost_error_t ghost_densemat_cm_malloc(ghost_densemat_t *vec)
{
    ghost_idx_t v;
    if (vec->val == NULL) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&vec->val,vec->traits.ncols*sizeof(char *)));

        for (v=0; v<vec->traits.ncols; v++) {
            vec->val[v] = NULL;
        }
    }

    if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
        return GHOST_SUCCESS;
    }

    if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
        if (vec->val[0] == NULL) {
            DEBUG_LOG(2,"Allocating host side of vector");
            GHOST_CALL_RETURN(ghost_malloc_align((void **)&vec->val[0],vec->traits.ncols*vec->traits.nrowspadded*vec->elSize,GHOST_DATA_ALIGNMENT));
            for (v=1; v<vec->traits.ncols; v++) {
                vec->val[v] = vec->val[0]+v*vec->traits.nrowspadded*vec->elSize;
            }
        }
    }

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        DEBUG_LOG(2,"Allocating device side of vector");
#ifdef GHOST_HAVE_CUDA
        if (vec->cu_val == NULL) {
#ifdef GHOST_HAVE_CUDA_PINNEDMEM
            WARNING_LOG("CUDA pinned memory is disabled");
            //ghost_cu_safecall(cudaHostGetDevicePointer((void **)&vec->cu_val,vec->val,0));
            GHOST_CALL_RETURN(ghost_cu_malloc(&vec->cu_val,vec->traits.nrowspadded*vec->traits.ncols*vec->elSize));
#else
            //ghost_cu_safecall(cudaMallocPitch(&(void *)vec->cu_val,&vec->traits.nrowspadded,vec->traits.nrowshalo*sizeofdt,vec->traits.ncols));
            GHOST_CALL_RETURN(ghost_cu_malloc(&vec->cu_val,vec->traits.nrowspadded*vec->traits.ncols*vec->elSize));
#endif
        }
#endif
    }   

    return GHOST_SUCCESS; 
}


static ghost_error_t vec_cm_fromVec(ghost_densemat_t *vec, ghost_densemat_t *vec2, ghost_idx_t roffs, ghost_idx_t coffs)
{
    ghost_densemat_cm_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from vector w/ col offset %"PRIDX,coffs);
    ghost_idx_t v;
    roffs += hwloc_bitmap_first(vec2->ldmask);
            
    if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE) {
        coffs += hwloc_bitmap_first(vec2->trmask);
    }

    for (v=0; v<vec->traits.ncols; v++) {
        if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
        {
            if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE)
            {
#ifdef GHOST_HAVE_CUDA
                ghost_cu_memcpy(CUVECVAL_CM(vec,vec->cu_val,v,hwloc_bitmap_first(vec->ldmask))),CUVECVAL_CM(vec2,vec2->cu_val,coffs+v,roffs),vec->traits.nrows*vec->elSize);
#endif
            }
            else
            {
#ifdef GHOST_HAVE_CUDA
                ghost_cu_upload(CUVECVAL_CM(vec,vec->cu_val,v,hwloc_bitmap_first(vec->ldmask))),VECVAL_CM(vec2,vec2->val,coffs+v,roffs),vec->traits.nrows*vec->elSize);
#endif
            }
        }
        else
        {
            if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE)
            {
#ifdef GHOST_HAVE_CUDA
                ghost_cu_download(VECVAL_CM(vec,vec->val,v,hwloc_bitmap_first(vec->ldmask))),CUVECVAL_CM(vec2,vec2->cu_val,coffs+v,roffs),vec->traits.nrows*vec->elSize);
#endif
            }
            else
            {
                memcpy(VECVAL_CM(vec,vec->val,v,hwloc_bitmap_first(vec->ldmask)),VECVAL_CM(vec2,vec2->val,coffs+v,roffs),vec->traits.nrows*vec->elSize);
            }
        }

    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale)
{
    GHOST_INSTR_START(axpy);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_idx_t nc = MIN(vec->traits.ncols,vec2->traits.ncols);
    char *s = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->elSize),err,ret);

    ghost_idx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->elSize],scale,vec->elSize);
    }

    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cm_vaxpy_funcs[dtIdx](vec,vec2,s),err,ret);

    goto out;
err:

out:
    free(s);
    GHOST_INSTR_STOP(axpy);
    return ret;
}

static ghost_error_t vec_cm_axpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *_b)
{
    GHOST_INSTR_START(axpby);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_idx_t nc = MIN(vec->traits.ncols,vec2->traits.ncols);
    char *s = NULL;
    char *b = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->elSize),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&b,nc*vec->elSize),err,ret);

    ghost_idx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->elSize],scale,vec->elSize);
        memcpy(&b[i*vec->elSize],_b,vec->elSize);
    }
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cm_vaxpby_funcs[dtIdx](vec,vec2,s,b),err,ret);

    goto out;
err:

out:
    free(s);
    free(b);
    GHOST_INSTR_STOP(axpby);
    return ret;
}

static ghost_error_t vec_cm_vaxpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_INSTR_START(vaxpy);
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    ret = ghost_densemat_cm_vaxpy_funcs[dtIdx](vec,vec2,scale);
    GHOST_INSTR_STOP(vaxpy);

    goto out;
err:
out:
    return ret;
}

static ghost_error_t vec_cm_vaxpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_INSTR_START(vaxpby);
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    ret = ghost_densemat_cm_vaxpby_funcs[dtIdx](vec,vec2,scale,b);
    GHOST_INSTR_STOP(vaxpby);
    goto out;
err:
out:
    return ret;
}

static ghost_error_t vec_cm_scale(ghost_densemat_t *vec, void *scale)
{
    GHOST_INSTR_START(scale);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_idx_t nc = vec->traits.ncols;
    char *s;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->elSize),err,ret);

    ghost_idx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->elSize],scale,vec->elSize);
    }
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_cm_vscale_funcs[dtIdx](vec,s),err,ret);

    goto out;
err:

out:
    free(s);
    GHOST_INSTR_STOP(scale);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_vscale(ghost_densemat_t *vec, void *scale)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_INSTR_START(vscale);
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    ret = ghost_densemat_cm_vscale_funcs[dtIdx](vec,scale);
    GHOST_INSTR_STOP(vscale);

    goto out;
err:
out:
    return ret;
}

static ghost_error_t vec_cm_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_INSTR_START(dot);
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    ret = ghost_densemat_cm_dotprod_funcs[dtIdx](vec,res,vec2);
    GHOST_INSTR_STOP(dot);
 
    goto out;
err:
out:
    return ret;
}

static ghost_error_t vec_cm_entry(ghost_densemat_t * vec, void *val, ghost_idx_t r, ghost_idx_t c) 
{
    int i = 0;
    int idx = hwloc_bitmap_first(vec->ldmask);
    for (i=0; i<r; i++) {
        idx = hwloc_bitmap_next(vec->ldmask,idx);
    }
    
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
    {
#ifdef GHOST_HAVE_CUDA
        int cidx = hwloc_bitmap_first(vec->trmask);
        for (i=0; i<c; i++) {
            cidx = hwloc_bitmap_next(vec->trmask,cidx);
        }
        ghost_cu_download(val,&vec->cu_val[(cidx*vec->traits.nrowspadded+idx)*vec->elSize],vec->elSize);
#endif
    }
    else if (vec->traits.flags & GHOST_DENSEMAT_HOST)
    {
        memcpy(val,VECVAL_CM(vec,vec->val,c,idx),vec->elSize);
    }

    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_fromRand(ghost_densemat_t *vec)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,vec->traits.datatype));
    return ghost_densemat_cm_fromRand_funcs[dtIdx](vec);
}

static ghost_error_t vec_cm_fromScalar(ghost_densemat_t *vec, void *val)
{
    ghost_densemat_cm_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from scalar value with %"PRIDX" rows",vec->traits.nrows);

    int row,col,rowidx;
    ITER_BEGIN_CM(vec,col,row,rowidx)
    memcpy(VECVAL_CM(vec,vec->val,col,row),val,vec->elSize);
    ITER_END_CM(rowidx)
    vec->upload(vec);

    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_toFile(ghost_densemat_t *vec, char *path)
{ // TODO two separate functions

#ifdef GHOST_HAVE_MPI
    int rank;
    GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));

    int32_t endianess = ghost_machine_bigendian();
    int32_t version = 1;
    int32_t order = GHOST_BINDENSEMAT_ORDER_COL_FIRST;
    int32_t datatype = vec->traits.datatype;
    int64_t nrows = (int64_t)vec->context->gnrows;
    int64_t ncols = (int64_t)vec->traits.ncols;
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
    ghost_idx_t v;
    ghost_mpi_datatype_t mpidt;
    GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits.datatype));
    MPI_CALL_RETURN(MPI_File_set_view(fileh,4*sizeof(int32_t)+2*sizeof(int64_t),mpidt,mpidt,"native",MPI_INFO_NULL));
    MPI_Offset fileoffset = vec->context->lfRow[rank];
    ghost_idx_t vecoffset = 0;
    for (v=0; v<vec->traits.ncols; v++) {
        char *val = NULL;
        int copied = 0;
        if (vec->traits.flags & GHOST_DENSEMAT_HOST)
        {
            vec->download(vec);
            val = VECVAL_CM(vec,vec->val,v,0);
        }
        else if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
        {
#ifdef GHOST_HAVE_CUDA
            GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits.nrows*vec->elSize));
            copied = 1;
            ghost_cu_download(val,&vec->cu_val[v*vec->traits.nrowspadded*vec->elSize],vec->traits.nrows*vec->elSize);
#endif
        }
        MPI_CALL_RETURN(MPI_File_write_at(fileh,fileoffset,val,vec->traits.nrows,mpidt,&status));
        fileoffset += nrows;
        vecoffset += vec->traits.nrowspadded*vec->elSize;
        if (copied)
            free(val);
    }
    MPI_CALL_RETURN(MPI_File_close(&fileh));


#else
    DEBUG_LOG(1,"Writing (local) vector to file %s",path);
    size_t ret;

    int32_t endianess = ghost_machine_bigendian();
    int32_t version = 1;
    int32_t order = GHOST_BINDENSEMAT_ORDER_COL_FIRST;
    int32_t datatype = vec->traits.datatype;
    int64_t nrows = (int64_t)vec->traits.nrows;
    int64_t ncols = (int64_t)vec->traits.ncols;

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

    ghost_idx_t v;
    for (v=0; v<vec->traits.ncols; v++) {
        char *val = NULL;
        int copied = 0;
        if (vec->traits.flags & GHOST_DENSEMAT_HOST)
        {
            vec->download(vec);
            val = VECVAL_CM(vec,vec->val,v,0);
        }
        else if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
        {
#ifdef GHOST_HAVE_CUDA
            GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits.nrows*vec->elSize));
            copied = 1;
            ghost_cu_download(val,&vec->cu_val[v*vec->traits.nrowspadded*vec->elSize],vec->traits.nrows*vec->elSize);
#endif
        }

        if ((ret = fwrite(val, vec->elSize, vec->traits.nrows,filed)) != vec->traits.nrows) {
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

static ghost_error_t vec_cm_fromFile(ghost_densemat_t *vec, char *path)
{
    int rank;
    GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));

    off_t offset;
    if ((vec->context == NULL) || !(vec->context->flags & GHOST_CONTEXT_DISTRIBUTED)) {
        offset = 0;
    } else {
        offset = vec->context->lfRow[rank];
    }

    ghost_densemat_cm_malloc(vec);
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

    if (endianess != GHOST_BINDENSEMAT_LITTLE_ENDIAN) {
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

    if (datatype != (int)vec->traits.datatype) {
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
    for (v=0; v<vec->traits.ncols; v++) {
        if (fseeko(filed,offset*vec->elSize,SEEK_CUR)) {
            ERROR_LOG("seek failed");
            vec->destroy(vec);
            return GHOST_ERR_IO;
        }
        if (vec->traits.flags & GHOST_DENSEMAT_HOST)
        {
            if ((ghost_idx_t)(ret = fread(VECVAL_CM(vec,vec->val,v,0), vec->elSize, vec->traits.nrows,filed)) != vec->traits.nrows) {
                ERROR_LOG("fread failed: %zu",ret);
                vec->destroy(vec);
                return GHOST_ERR_IO;
            }
            vec->upload(vec);
        }
        else if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
        {
#ifdef GHOST_HAVE_CUDA
            char *val;
            GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits.nrows*vec->elSize));
            if ((ret = fread(val, vec->elSize, vec->traits.nrows,filed)) != vec->traits.nrows) {
                ERROR_LOG("fread failed: %zu",ret);
                vec->destroy(vec);
                return GHOST_ERR_IO;
            }
            ghost_cu_upload(&vec->cu_val[v*vec->traits.nrowspadded*vec->elSize],val,vec->traits.nrows*vec->elSize);
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

static ghost_error_t vec_cm_fromFunc(ghost_densemat_t *vec, void (*fp)(ghost_idx_t, ghost_idx_t, void *))
{
    int rank;
    ghost_idx_t offset;
    if (vec->context) {
        GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));
        offset = vec->context->lfRow[rank];
    } else {
        rank = 0;
        offset = 0;
    }
    GHOST_CALL_RETURN(ghost_densemat_cm_malloc(vec));
    DEBUG_LOG(1,"Filling vector via function");

    ghost_idx_t row,col,rowidx;

    if (vec->traits.flags & GHOST_DENSEMAT_HOST) { // vector is stored _at least_ at host
        ITER_BEGIN_CM(vec,col,row,rowidx)
        fp(offset+rowidx,col,VECVAL_CM(vec,vec->val,col,row));
        ITER_END_CM(rowidx)
        vec->upload(vec);
    } else {
        ghost_densemat_t *hostVec;
        ghost_densemat_traits_t htraits = vec->traits;
        htraits.flags &= ~GHOST_DENSEMAT_DEVICE;
        htraits.flags |= GHOST_DENSEMAT_HOST;
        GHOST_CALL_RETURN(ghost_densemat_create(&hostVec,vec->context,htraits));
        GHOST_CALL_RETURN(hostVec->fromFunc(hostVec,fp));
        GHOST_CALL_RETURN(vec->fromVec(vec,hostVec,0,0));
        hostVec->destroy(hostVec);
    }

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_distributeVector(ghost_densemat_t *vec, ghost_densemat_t *nodeVec)
{
    DEBUG_LOG(1,"Distributing vector");
    int me;
    int nprocs;
    GHOST_CALL_RETURN(ghost_rank(&me, nodeVec->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, nodeVec->context->mpicomm));

    ghost_idx_t c;
#ifdef GHOST_HAVE_MPI
    DEBUG_LOG(2,"Scattering global vector to local vectors");

    ghost_mpi_datatype_t mpidt;
    GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits.datatype));

    int i;

    MPI_Request req[vec->traits.ncols*2*(nprocs-1)];
    MPI_Status stat[vec->traits.ncols*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits.ncols*2*(nprocs-1);i++) 
        req[i] = MPI_REQUEST_NULL;

    if (me != 0) {
        for (c=0; c<vec->traits.ncols; c++) {
            MPI_CALL_RETURN(MPI_Irecv(nodeVec->val[c],nodeVec->context->lnrows[me],mpidt,0,me,nodeVec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(nodeVec->val[c],vec->val[c],vec->elSize*nodeVec->context->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Isend(VECVAL_CM(vec,vec->val,c,nodeVec->context->lfRow[i]),nodeVec->context->lnrows[i],mpidt,i,i,nodeVec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else

    for (c=0; c<vec->traits.ncols; c++) {
        memcpy(nodeVec->val[c],vec->val[c],vec->traits.nrows*vec->elSize);
    }
    //    *nodeVec = vec->clone(vec);
#endif

    nodeVec->upload(nodeVec);

    DEBUG_LOG(1,"Vector distributed successfully");

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_collectVectors(ghost_densemat_t *vec, ghost_densemat_t *totalVec) 
{
    ghost_idx_t c;
#ifdef GHOST_HAVE_MPI
    int me;
    int nprocs;
    ghost_mpi_datatype_t mpidt;
    GHOST_CALL_RETURN(ghost_rank(&me, vec->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, vec->context->mpicomm));
    GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits.datatype));

//    if (vec->context != NULL)
//        vec->permute(vec,vec->context->invRowPerm); 

    int i;

    MPI_Request req[vec->traits.ncols*2*(nprocs-1)];
    MPI_Status stat[vec->traits.ncols*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits.ncols*2*(nprocs-1);i++) 
        req[i] = MPI_REQUEST_NULL;

    if (me != 0) {
        for (c=0; c<vec->traits.ncols; c++) {
            MPI_CALL_RETURN(MPI_Isend(vec->val[c],vec->context->lnrows[me],mpidt,0,me,vec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(totalVec->val[c],vec->val[c],vec->elSize*vec->context->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Irecv(VECVAL_CM(totalVec,totalVec->val,c,vec->context->lfRow[i]),vec->context->lnrows[i],mpidt,i,i,vec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else
    if (vec->context != NULL) {
//        vec->permute(vec,vec->context->invRowPerm);
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(totalVec->val[c],vec->val[c],totalVec->traits.nrows*vec->elSize);
        }
    }
#endif

    return GHOST_SUCCESS;

}

static void ghost_freeVector( ghost_densemat_t* vec ) 
{
    if( vec ) {
        if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
            ghost_idx_t v;
#ifdef GHOST_HAVE_CUDA_PINNEDMEM
            WARNING_LOG("CUDA pinned memory is disabled");
            /*if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
              for (v=0; v<vec->traits.ncols; v++) { 
              ghost_cu_safecall(cudaFreeHost(vec->val[v]));
              }
              }*/
            if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
                for (v=0; v<vec->traits.ncols; v++) {
                    free(vec->val[v]); vec->val[v] = NULL
                }
            }
            else {
                free(vec->val[0]); vec->val[0] = NULL;
            }
#else
            //note: a 'scattered' vector (one with non-constant stride) is
            //      always a view of some other (there is no method to                        
            //      construct it otherwise), but we check anyway in case
            //      the user has built his own funny vector in memory.
            if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
                for (v=0; v<vec->traits.ncols; v++) {
                    free(vec->val[v]); vec->val[v] = NULL;
                }
            }
            else {
                free(vec->val[0]); vec->val[0] = NULL;
            }
#endif
#ifdef GHOST_HAVE_CUDA
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
                ghost_cu_free(vec->cu_val);
            }
#endif
        }
        free(vec->val); vec->val = NULL;
        free(vec);
        hwloc_bitmap_free(vec->ldmask);
        hwloc_bitmap_free(vec->trmask);
    }
}
static ghost_error_t ghost_permuteVector( ghost_densemat_t* vec, ghost_permutation_t *permutation, ghost_permutation_direction_t dir) 
{
    // TODO enhance performance
    
    if (!permutation) {
        return GHOST_SUCCESS;
    }

    ghost_idx_t i;
    ghost_idx_t len, c;
    char* tmp = NULL;
    ghost_densemat_t *permvec = NULL;
    ghost_densemat_t *combined = NULL; 
    ghost_densemat_traits_t traits;

    if (permutation->len > vec->traits.nrows && !vec->context) {
        ERROR_LOG("The permutation scope is larger than the vector but the vector does not have a context (i.e.,\
            process-local vectors cannot be combined to a big vector for permuting.");
        return GHOST_ERR_INVALID_ARG;
    }
    if (permutation->len > vec->traits.nrows && vec->context->gnrows != permutation->len) {
        ERROR_LOG("The permutation scope and the context size do not match!");
        return GHOST_ERR_INVALID_ARG;
    }

    if (permutation->scope == GHOST_PERMUTATION_GLOBAL && vec->traits.nrows != permutation->len) {
        traits = vec->traits;
        traits.nrows = vec->context->gnrows;
        traits.flags = GHOST_DENSEMAT_HOST;
        char zero[vec->elSize];
        memset(zero,0,vec->elSize);

        ghost_densemat_create(&combined,vec->context,traits);
        combined->fromScalar(combined,&zero);
        vec->collect(vec,combined);
        permvec = combined;

        WARNING_LOG("Global permutation not tested");
    } else {
        permvec = vec;
    }
    if (permvec->traits.nrows != permutation->len) {
        WARNING_LOG("Lenghts do not match!");
        return GHOST_ERR_INVALID_ARG;
    }
    len = permvec->traits.nrows;

    ghost_idx_t *perm = NULL;
    if (dir == GHOST_PERMUTATION_ORIG2PERM) {
        perm = permutation->perm;
    } else {
        perm = permutation->invPerm;
    }

    if (perm == NULL) {
        DEBUG_LOG(1,"Permutation vector is NULL, returning.");
        return GHOST_SUCCESS;
    } else {
        DEBUG_LOG(1,"Permuting vector");
    }


    for (c=0; c<permvec->traits.ncols; c++) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,permvec->elSize*len));
        for(i = 0; i < len; ++i) {
            if( perm[i] >= len ) {
                ERROR_LOG("Permutation index out of bounds: %"PRIDX" > %"PRIDX,perm[i],len);
                free(tmp);
                return GHOST_ERR_UNKNOWN;
            }

            memcpy(&tmp[vec->elSize*perm[i]],VECVAL_CM(permvec,permvec->val,c,i),permvec->elSize);
        }
        for(i=0; i < len; ++i) {
            memcpy(VECVAL_CM(permvec,permvec->val,c,i),&tmp[permvec->elSize*i],permvec->elSize);
        }
        free(tmp);
    }
    
    if (permutation->scope == GHOST_PERMUTATION_GLOBAL && vec->traits.nrows != permutation->len) {
        permvec->distribute(permvec,vec);
        permvec->destroy(permvec);
    }

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_cloneVector(ghost_densemat_t *src, ghost_densemat_t **new, ghost_idx_t nr, ghost_idx_t roffs, ghost_idx_t nc, ghost_idx_t coffs)
{
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc;
    newTraits.ncolsorig = nc;
    newTraits.nrows = nr;
    newTraits.nrowsorig = nr;
    ghost_densemat_create(new,src->context,newTraits);

    // copy the data even if the input vector is itself a view
    // (bitwise NAND operation to unset the view flag if set)
    (*new)->traits.flags &= ~GHOST_DENSEMAT_VIEW;

    (*new)->fromVec(*new,src,roffs,coffs);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_cm_compress(ghost_densemat_t *vec)
{
    if (!(vec->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        return GHOST_SUCCESS;
    }

    if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
        ghost_idx_t v,i;

        char *val = NULL;
        GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits.nrowspadded*vec->traits.ncols*vec->elSize));

#pragma omp parallel for schedule(runtime) private(v)
        for (i=0; i<vec->traits.nrowspadded; i++)
        {
            for (v=0; v<vec->traits.ncols; v++)
            {
                val[(v*vec->traits.nrowspadded+i)*vec->elSize] = 0;
            }
        }

        for (v=0; v<vec->traits.ncols; v++)
        {
            memcpy(&val[(v*vec->traits.nrowspadded)*vec->elSize],
                    VECVAL_CM(vec,vec->val,v,0),vec->traits.nrowspadded*vec->elSize);

            if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
                free(vec->val[v]);
            }
            vec->val[v] = &val[(v*vec->traits.nrowspadded)*vec->elSize];
        }
    }
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        ghost_idx_t v,i,r,j;

        char *cu_val;
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_val,vec->traits.nrowspadded*vec->traits.ncols*vec->elSize));

        for (v=0, i=0; v<vec->traits.ncolsorig; v++) {
            if (hwloc_bitmap_isset(vec->trmask,v)) {
                for (r=0, j=0; r<vec->traits.nrowsorig; r++) {
                    if (hwloc_bitmap_isset(vec->ldmask,r)) {
                        ghost_cu_memcpy(&cu_val[(i*vec->traits.nrowspadded+j)*vec->elSize],
                                &vec->cu_val[(v*vec->traits.nrowspadded+r)*vec->elSize],
                                vec->elSize);
                        j++;
                    }
                }
                i++;
            }
        }
        if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
            ghost_cu_free(vec->cu_val);
        }
        vec->cu_val = cu_val;
#endif 
    }

    hwloc_bitmap_fill(vec->ldmask);
    hwloc_bitmap_fill(vec->trmask);
    vec->traits.flags &= ~GHOST_DENSEMAT_VIEW;
    vec->traits.flags &= ~GHOST_DENSEMAT_SCATTERED;
    vec->traits.ncolsorig = vec->traits.ncols;
    vec->traits.nrowsorig = vec->traits.nrows;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_densemat_traits_clone(ghost_densemat_traits_t *t1, ghost_densemat_traits_t **t2)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)t2,sizeof(ghost_densemat_traits_t)));
    memcpy(*t2,t1,sizeof(ghost_densemat_traits_t));

    return GHOST_SUCCESS;
}
