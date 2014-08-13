#define _XOPEN_SOURCE 500 
#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/core.h"
#include "ghost/densemat_rm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/context.h"
#include "ghost/instr.h"
#include "ghost/machine.h"
#include "ghost/log.h"
#include "ghost/bindensemat.h"
#include "ghost/densemat_cm.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

static ghost_error_t (*ghost_densemat_rm_normalize_funcs[4]) (ghost_densemat_t *) = 
{&s_ghost_densemat_rm_normalize, &d_ghost_densemat_rm_normalize, &c_ghost_densemat_rm_normalize, &z_ghost_densemat_rm_normalize};

static ghost_error_t (*ghost_densemat_rm_dotprod_funcs[4]) (ghost_densemat_t *, void *, ghost_densemat_t *) = 
{&s_ghost_densemat_rm_dotprod, &d_ghost_densemat_rm_dotprod, &c_ghost_densemat_rm_dotprod, &z_ghost_densemat_rm_dotprod};

static ghost_error_t (*ghost_densemat_rm_vscale_funcs[4]) (ghost_densemat_t *, void*) = 
{&s_ghost_densemat_rm_vscale, &d_ghost_densemat_rm_vscale, &c_ghost_densemat_rm_vscale, &z_ghost_densemat_rm_vscale};

static ghost_error_t (*ghost_densemat_rm_vaxpy_funcs[4]) (ghost_densemat_t *, ghost_densemat_t *, void*) = 
{&s_ghost_densemat_rm_vaxpy, &d_ghost_densemat_rm_vaxpy, &c_ghost_densemat_rm_vaxpy, &z_ghost_densemat_rm_vaxpy};

static ghost_error_t (*ghost_densemat_rm_vaxpby_funcs[4]) (ghost_densemat_t *, ghost_densemat_t *, void*, void*) = 
{&s_ghost_densemat_rm_vaxpby, &d_ghost_densemat_rm_vaxpby, &c_ghost_densemat_rm_vaxpby, &z_ghost_densemat_rm_vaxpby};

static ghost_error_t (*ghost_densemat_rm_fromRand_funcs[4]) (ghost_densemat_t *) = 
{&s_ghost_densemat_rm_fromRand, &d_ghost_densemat_rm_fromRand, &c_ghost_densemat_rm_fromRand, &z_ghost_densemat_rm_fromRand};

static ghost_error_t (*ghost_densemat_rm_string_funcs[4]) (char **str, ghost_densemat_t *) = 
{&s_ghost_densemat_rm_string, &d_ghost_densemat_rm_string, &c_ghost_densemat_rm_string, &z_ghost_densemat_rm_string};

static ghost_error_t vec_rm_print(ghost_densemat_t *vec, char **str);
static ghost_error_t vec_rm_scale(ghost_densemat_t *vec, void *scale);
static ghost_error_t vec_rm_vscale(ghost_densemat_t *vec, void *scale);
static ghost_error_t vec_rm_vaxpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale);
static ghost_error_t vec_rm_vaxpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b);
static ghost_error_t vec_rm_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale);
static ghost_error_t vec_rm_axpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b);
static ghost_error_t vec_rm_dotprod(ghost_densemat_t *vec, void * res, ghost_densemat_t *vec2);
static ghost_error_t vec_rm_fromFunc(ghost_densemat_t *vec, void (*fp)(ghost_gidx_t, ghost_lidx_t, void *));
static ghost_error_t vec_rm_fromVec(ghost_densemat_t *vec, ghost_densemat_t *vec2, ghost_lidx_t roffs, ghost_lidx_t coffs);
static ghost_error_t vec_rm_fromRand(ghost_densemat_t *vec);
static ghost_error_t vec_rm_fromScalar(ghost_densemat_t *vec, void *val);
static ghost_error_t vec_rm_fromFile(ghost_densemat_t *vec, char *path);
static ghost_error_t vec_rm_toFile(ghost_densemat_t *vec, char *path);
static ghost_error_t ghost_densemat_rm_normalize( ghost_densemat_t *vec);
static ghost_error_t ghost_distributeVector(ghost_densemat_t *vec, ghost_densemat_t *nodeVec);
static ghost_error_t ghost_collectVectors(ghost_densemat_t *vec, ghost_densemat_t *totalVec); 
static void ghost_freeVector( ghost_densemat_t* const vec );
static ghost_error_t ghost_permuteVector( ghost_densemat_t* vec, ghost_permutation_t *permutation, ghost_permutation_direction_t dir); 
static ghost_error_t ghost_cloneVector(ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nr, ghost_lidx_t roffs, ghost_lidx_t nc, ghost_lidx_t coffs);
static ghost_error_t vec_rm_entry(ghost_densemat_t *, void *, ghost_lidx_t, ghost_lidx_t);
static ghost_error_t vec_rm_view (ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nr, ghost_lidx_t roffs, ghost_lidx_t nc, ghost_lidx_t coffs);
static ghost_error_t vec_rm_viewScatteredVec (ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nr, ghost_lidx_t *roffs, ghost_lidx_t nc, ghost_lidx_t *coffs);
static ghost_error_t vec_rm_viewScatteredCols (ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nc, ghost_lidx_t *coffs);
static ghost_error_t vec_rm_viewCols (ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nc, ghost_lidx_t coffs);
static ghost_error_t vec_rm_viewPlain (ghost_densemat_t *vec, void *data, ghost_lidx_t roffs, ghost_lidx_t coffs, ghost_lidx_t lda);
static ghost_error_t vec_rm_compress(ghost_densemat_t *vec);
static ghost_error_t vec_rm_upload(ghost_densemat_t *vec);
static ghost_error_t vec_rm_download(ghost_densemat_t *vec);
static ghost_error_t vec_rm_uploadHalo(ghost_densemat_t *vec);
static ghost_error_t vec_rm_downloadHalo(ghost_densemat_t *vec);
static ghost_error_t vec_rm_uploadNonHalo(ghost_densemat_t *vec);
static ghost_error_t vec_rm_downloadNonHalo(ghost_densemat_t *vec);
static ghost_error_t vec_rm_memtranspose(ghost_densemat_t *vec);
ghost_error_t ghost_densemat_rm_malloc(ghost_densemat_t *vec);

ghost_error_t ghost_densemat_rm_setfuncs(ghost_densemat_t *vec)
{
    ghost_error_t ret = GHOST_SUCCESS;

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
    {
#ifdef GHOST_HAVE_CUDA
        vec->dot = &ghost_densemat_rm_cu_dotprod;
        vec->vaxpy = &ghost_densemat_rm_cu_vaxpy;
        vec->vaxpby = &ghost_densemat_rm_cu_vaxpby;
        vec->axpy = &ghost_densemat_rm_cu_axpy;
        vec->axpby = &ghost_densemat_rm_cu_axpby;
        vec->scale = &ghost_densemat_rm_cu_scale;
        vec->vscale = &ghost_densemat_rm_cu_vscale;
        vec->fromScalar = &ghost_densemat_rm_cu_fromScalar;
        vec->fromRand = &ghost_densemat_rm_cu_fromRand;
#endif
    }
    else if (vec->traits.flags & GHOST_DENSEMAT_HOST)
    {
        vec->dot = &vec_rm_dotprod;
        vec->vaxpy = &vec_rm_vaxpy;
        vec->vaxpby = &vec_rm_vaxpby;
        vec->axpy = &vec_rm_axpy;
        vec->axpby = &vec_rm_axpby;
        vec->scale = &vec_rm_scale;
        vec->vscale = &vec_rm_vscale;
        vec->fromScalar = &vec_rm_fromScalar;
        vec->fromRand = &vec_rm_fromRand;
    }

    vec->memtranspose = &vec_rm_memtranspose;
    vec->compress = &vec_rm_compress;
    vec->string = &vec_rm_print;
    vec->fromFunc = &vec_rm_fromFunc;
    vec->fromVec = &vec_rm_fromVec;
    vec->fromFile = &vec_rm_fromFile;
    vec->toFile = &vec_rm_toFile;
    vec->distribute = &ghost_distributeVector;
    vec->collect = &ghost_collectVectors;
    vec->normalize = &ghost_densemat_rm_normalize;
    vec->destroy = &ghost_freeVector;
    vec->permute = &ghost_permuteVector;
    vec->clone = &ghost_cloneVector;
    vec->entry = &vec_rm_entry;
    vec->viewVec = &vec_rm_view;
    vec->viewPlain = &vec_rm_viewPlain;
    vec->viewScatteredVec = &vec_rm_viewScatteredVec;
    vec->viewScatteredCols = &vec_rm_viewScatteredCols;
    vec->viewCols = &vec_rm_viewCols;

    vec->upload = &vec_rm_upload;
    vec->download = &vec_rm_download;
    vec->uploadHalo = &vec_rm_uploadHalo;
    vec->downloadHalo = &vec_rm_downloadHalo;
    vec->uploadNonHalo = &vec_rm_uploadNonHalo;
    vec->downloadNonHalo = &vec_rm_downloadNonHalo;
#ifdef GHOST_HAVE_CUDA
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        vec->cu_val = NULL;
    }
#endif

    return ret;
}

static ghost_error_t vec_rm_memtranspose(ghost_densemat_t *vec)
{
    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        ERROR_LOG("Cannot memtranspose scattered densemat views!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    ghost_lidx_t col,row;
    if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
        INFO_LOG("Memtranspose of a view. The densemat will loose its view property.");
        char *oldval;
        ghost_densemat_valptr(vec,(void **)&oldval);
        
        vec->traits.storage = GHOST_DENSEMAT_COLMAJOR;
        vec->traits.flags &= ~GHOST_DENSEMAT_VIEW;
        ghost_bitmap_set_range(vec->ldmask,0,vec->traits.ncols-1);
        vec->traits.ncolsorig = vec->traits.ncols;
        vec->traits.nrowsorig = vec->traits.nrows;
        ghost_densemat_cm_setfuncs(vec);
        
        free(vec->val); vec->val = NULL; 
        ghost_densemat_cm_malloc(vec);
        for (row=0; row<vec->traits.nrows; row++) {
            for (col=0; col<vec->traits.ncols; col++) {
                memcpy(vec->val[0]+col*vec->traits.nrowspadded*vec->elSize+row*vec->elSize,
                        oldval+vec->traits.ncolspadded*row*vec->elSize+col*vec->elSize,
                        vec->elSize);
            }
        }

    } else {
        vec->traits.storage = GHOST_DENSEMAT_COLMAJOR;
        ghost_densemat_cm_setfuncs(vec);
        if (vec->viewing) {
            INFO_LOG("In-place back-transpose. The densemat will regain its view property.");
            vec->traits.flags |= GHOST_DENSEMAT_VIEW;
            vec->traits.ncolsorig = vec->viewing->traits.ncols;
            vec->traits.nrowsorig = vec->viewing->traits.nrows;
            
            if (vec->viewing_row) {
                ghost_bitmap_clr_range(vec->ldmask,0,vec->viewing_row-1);
            }
            ghost_bitmap_clr_range(vec->ldmask,vec->traits.nrows+vec->viewing_row,-1);

            for (row=0; row<vec->traits.nrows; row++) {
                for (col=0; col<vec->traits.ncols; col++) {
                    memcpy(&vec->viewing->val[vec->viewing_col+col][(vec->viewing_row+row)*vec->elSize],
                            vec->val[0]+vec->traits.ncolspadded*row*vec->elSize+col*vec->elSize,
                            vec->elSize);
                }
            }
            vec->val = NULL;
            GHOST_CALL_RETURN(ghost_malloc((void **)&vec->val,vec->traits.ncolspadded*sizeof(char *)));
            vec->val[0] = vec->viewing->val[vec->viewing_col];
            for (col=1; col<vec->traits.ncolspadded; col++) {
                vec->val[col] = vec->val[0]+vec->viewing->traits.nrowspadded*col*vec->elSize;
            }

            
        } else {

            char *oldval = vec->val[0];
            free(vec->val); 
            vec->val = NULL;
            GHOST_CALL_RETURN(ghost_malloc((void **)&vec->val,vec->traits.ncolspadded*sizeof(char *)));
            vec->val[0] = oldval;
            for (col=1; col<vec->traits.ncolspadded; col++) {
                vec->val[col] = vec->val[0]+vec->traits.nrowspadded*col*vec->elSize;
            }

            char *tmp;
            GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,vec->elSize*vec->traits.nrowspadded*vec->traits.ncolspadded));
            memcpy(tmp,vec->val[0],vec->elSize*vec->traits.nrowspadded*vec->traits.ncolspadded);

            for (row=0; row<vec->traits.nrows; row++) {
                for (col=0; col<vec->traits.ncols; col++) {
                    memcpy(vec->val[0]+col*vec->traits.nrowspadded*vec->elSize+row*vec->elSize,
                            tmp+vec->traits.ncolspadded*row*vec->elSize+col*vec->elSize,
                            vec->elSize);
                }
            }

            free(tmp);
        }
    }

    return GHOST_SUCCESS; 
}

static ghost_error_t vec_rm_uploadHalo(ghost_densemat_t *vec)
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Uploading %"PRLIDX" rows of vector",vec->traits.nrowshalo);
#ifdef GHOST_HAVE_CUDA
        GHOST_CALL_RETURN(ghost_cu_upload(CUVECVAL_RM(vec,vec->cu_val,vec->traits.nrows,0),VECVAL_RM(vec,vec->val,vec->traits.nrows,0),vec->traits.ncolspadded*vec->context->halo_elements*vec->elSize));
        //GHOST_CALL_RETURN(ghost_cu_upload(&vec->cu_val[vec->traits.nrows*vec->traits.ncolspadded*vec->elSize],vec->val[vec->traits.nrows],vec->traits.ncolspadded*vec->context->halo_elements*vec->elSize));
        /*ghost_lidx_t x,c,r;
        for (x=vec->traits.nrowsorig, r=0; x<vec->traits.nrowshalo; x++) {
            if (ghost_bitmap_isset(vec->trmask,x)) {
                for (c=0; c<vec->traits.ncolsorig; c++) {
                    if (ghost_bitmap_isset(vec->ldmask,c)) {
                        ghost_cu_upload(&vec->cu_val[(vec->traits.ncolspadded*x+c)*vec->elSize],VECVAL_RM(vec,vec->val,r,c), vec->elSize);
                    }
                }
                r++;
            }
        }*/
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_downloadHalo(ghost_densemat_t *vec)
{

    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Downloading halo elements of vector");
        WARNING_LOG("Not yet implemented!");
    }
    return GHOST_SUCCESS;
}
static ghost_error_t vec_rm_uploadNonHalo(ghost_densemat_t *vec)
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Uploading %"PRLIDX" rows of vector",vec->traits.nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t r;
        for (r=0; r<vec->traits.nrows; r++) {
            ghost_cu_upload(CUVECVAL_RM(vec,vec->cu_val,r,0),
                    VECVAL_RM(vec,vec->val,r,0), 
                    vec->traits.ncols*vec->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_downloadNonHalo(ghost_densemat_t *vec)
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Downloading vector");
#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t r;
        for (r=0; r<vec->traits.nrows; r++) {
            ghost_cu_download(VECVAL_RM(vec,vec->val,r,0),
                    CUVECVAL_RM(vec,vec->cu_val,r,0),
                    vec->traits.ncols*vec->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_upload(ghost_densemat_t *vec) 
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Uploading %"PRLIDX" rows of vector",vec->traits.nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t x,c,r;
        for (x=0, r=0; x<vec->traits.nrowsorig; x++) {
            if (ghost_bitmap_isset(vec->trmask,x)) {
                for (c=0; c<vec->traits.ncolsorig; c++) {
                    if (ghost_bitmap_isset(vec->ldmask,c)) {
                        ghost_cu_upload(&vec->cu_val[(vec->traits.ncolspadded*x+c)*vec->elSize],VECVAL_RM(vec,vec->val,r,c), vec->elSize);
                    }
                }
                r++;
            }
        }
#endif
    }
    
    vec->uploadHalo(vec);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_download(ghost_densemat_t *vec)
{
    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && (vec->traits.flags & GHOST_DENSEMAT_DEVICE)) {
        DEBUG_LOG(1,"Downloading vector");
#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t r;
        for (r=0; r<vec->traits.nrowshalo; r++) {
            ghost_cu_download(VECVAL_RM(vec,vec->val,r,0),
                    CUVECVAL_RM(vec,vec->cu_val,r,0),
                    vec->traits.ncols*vec->elSize);
        }
#endif
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_view (ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nr, ghost_lidx_t roffs, ghost_lidx_t nc, ghost_lidx_t coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" dense matrix with col offset %"PRLIDX,src->traits.nrows,nc,coffs);
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc;
    newTraits.nrows = nr;
    newTraits.flags |= GHOST_DENSEMAT_VIEW;

    ghost_densemat_create(new,src->context,newTraits);
    ghost_bitmap_copy((*new)->ldmask,src->ldmask);
    ghost_bitmap_copy((*new)->trmask,src->trmask);
    ghost_densemat_rm_malloc(*new);
    ghost_lidx_t r,c,viewedcol;
    
    for (viewedcol=0, c=0; c<src->traits.ncolsorig; c++) {
        if (viewedcol<coffs || (viewedcol >= coffs+nc)) {
            ghost_bitmap_clr((*new)->ldmask,c);
        }
        if (ghost_bitmap_isset(src->ldmask,c)) {
            viewedcol++;
        }
    }
    
    if ((*new)->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t viewedrow;
        (*new)->cu_val = src->cu_val;
        for (viewedrow=0, r=0; r<src->traits.nrowsorig; r++) {
            if (viewedrow<roffs || (viewedrow >= roffs+nr)) {
                ghost_bitmap_clr((*new)->trmask,r);
            }
            if (ghost_bitmap_isset(src->trmask,r)) {
                viewedrow++;
            }
        }
#endif
    } 

    if ((*new)->traits.flags & GHOST_DENSEMAT_HOST) {
        for (r=0; r<(*new)->traits.nrowspadded; r++) {
            (*new)->val[r] = VECVAL_RM(src,src->val,roffs+r,0);
        }
    }

    (*new)->viewing = src;
    (*new)->viewing_col = coffs;
    (*new)->viewing_row = roffs;
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_viewPlain (ghost_densemat_t *vec, void *data, ghost_lidx_t roffs, ghost_lidx_t coffs, ghost_lidx_t lda)
{
    WARNING_LOG("Possibly broken!");
    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" dense matrix from plain data with offset %"PRLIDX"x%"PRLIDX,vec->traits.nrows,vec->traits.ncols,roffs,coffs);

    ghost_lidx_t v;

    for (v=0; v<vec->traits.nrowspadded; v++) {
        vec->val[v] = &((char *)data)[(lda*(coffs+v)+roffs)*vec->elSize];
    }
    vec->traits.flags |= GHOST_DENSEMAT_VIEW;

    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_viewScatteredVec (ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nr, ghost_lidx_t *roffs, ghost_lidx_t nc, ghost_lidx_t *coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" scattered dense matrix",src->traits.nrows,nc);
    ghost_lidx_t c,r,i,viewedcol;
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc; 
    newTraits.nrows = nr;
    newTraits.flags |= GHOST_DENSEMAT_VIEW;
    newTraits.flags |= GHOST_DENSEMAT_SCATTERED;

    ghost_densemat_create(new,src->context,newTraits);
    ghost_bitmap_copy((*new)->ldmask,src->ldmask);
    ghost_bitmap_copy((*new)->trmask,src->trmask);
    ghost_densemat_rm_malloc(*new);

    for (c=0,i=0,viewedcol=-1; c<(*new)->traits.ncolsorig; c++) {
        if (ghost_bitmap_isset(src->ldmask,c)) {
            viewedcol++;
        }
        if (coffs[i] != viewedcol) {
            ghost_bitmap_clr((*new)->ldmask,c);
        } else {
            i++;
        }
    }
    
    if ((*new)->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t viewedrow;
        (*new)->cu_val = src->cu_val;
        for (r=0,i=0,viewedrow=-1; r<(*new)->traits.nrowsorig; r++) {
            if (ghost_bitmap_isset(src->trmask,r)) {
                viewedrow++;
            }
            if (roffs[i] != viewedrow) {
                ghost_bitmap_clr((*new)->trmask,r);
            } else {
                i++;
            }
        }
#endif
    } 
    
    if ((*new)->traits.flags & GHOST_DENSEMAT_HOST) {
        for (r=0; r<newTraits.nrows; r++) {
            (*new)->val[r] = VECVAL_RM(src,src->val,roffs[r],0);
        }
    }

    (*new)->viewing = src;
    (*new)->viewing_col = -1;
    (*new)->viewing_row = -1;

    return GHOST_SUCCESS;
}


static ghost_error_t vec_rm_viewCols (ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nc, ghost_lidx_t coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" contiguous dense matrix",src->traits.nrows,nc);
    ghost_lidx_t r,c,viewedcol;
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc; 
    newTraits.flags |= GHOST_DENSEMAT_VIEW;

    ghost_densemat_create(new,src->context,newTraits);
    ghost_bitmap_copy((*new)->ldmask,src->ldmask);
    ghost_bitmap_copy((*new)->trmask,src->trmask);
    ghost_densemat_rm_malloc(*new);

#ifdef GHOST_HAVE_CUDA
    if ((*new)->traits.flags & GHOST_DENSEMAT_DEVICE) {
        (*new)->cu_val = src->cu_val;
    }
#endif
    
    for (viewedcol=0, c=0; c<src->traits.ncolsorig; c++) {
        if (viewedcol<coffs || (viewedcol >= coffs+nc)) {
            ghost_bitmap_clr((*new)->ldmask,c);
        }
        if (ghost_bitmap_isset(src->ldmask,c)) {
            viewedcol++;
        }
    }
    
    for (r=0; r<newTraits.nrowspadded; r++) {
        (*new)->val[r] = VECVAL_RM(src,src->val,r,0);
    }

    (*new)->viewing = src;
    (*new)->viewing_col = coffs;
    (*new)->viewing_row = 0;
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_viewScatteredCols (ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nc, ghost_lidx_t *coffs)
{
    if (!array_strictly_ascending(coffs,nc)) {
        ERROR_LOG("Can only view sctrictly ascending scattered columns for row-major densemats!");
        return GHOST_ERR_INVALID_ARG;
    }

    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" scattered dense matrix",src->traits.nrows,nc);
    ghost_lidx_t c,i,r,viewedcol;
    ghost_densemat_traits_t newTraits = src->traits;
    newTraits.ncols = nc; 
    newTraits.flags |= GHOST_DENSEMAT_VIEW;
    newTraits.flags |= GHOST_DENSEMAT_SCATTERED;

    ghost_densemat_create(new,src->context,newTraits);
    ghost_bitmap_copy((*new)->ldmask,src->ldmask);
    ghost_densemat_rm_malloc(*new);

#ifdef GHOST_HAVE_CUDA
    if ((*new)->traits.flags & GHOST_DENSEMAT_DEVICE) {
        (*new)->cu_val = src->cu_val;
    }
#endif
   
//    char *bm;
//    ghost_bitmap_list_asprintf(&bm,src->ldmask);
//    WARNING_LOG("src->ldmask: %s",bm);
    for (viewedcol=-1, c=0, i=0; c<src->traits.ncolsorig; c++) {
//        INFO_LOG("c %d viewedcol %d i %d coffs[i] %d isset[c] %d",c,viewedcol,i,coffs[i],ghost_bitmap_isset(src->ldmask,c));
        if (ghost_bitmap_isset(src->ldmask,c)) {
            viewedcol++;
        }
        if (coffs[i] != viewedcol) {
            ghost_bitmap_clr((*new)->ldmask,c);
        } else {
            i++;
        }
    }
    for (r=0; r<newTraits.nrowspadded; r++) {
        (*new)->val[r] = VECVAL_RM(src,src->val,r,0);
    }

    (*new)->viewing = src;
    (*new)->viewing_col = -1;
    (*new)->viewing_row = -1;
    return GHOST_SUCCESS;
}

static ghost_error_t ghost_densemat_rm_normalize( ghost_densemat_t *vec)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,vec->traits.datatype));
    ghost_densemat_rm_normalize_funcs[dtIdx](vec);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_print(ghost_densemat_t *vec, char **str)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,vec->traits.datatype));
    return ghost_densemat_rm_string_funcs[dtIdx](str,vec);
}

ghost_error_t ghost_densemat_rm_malloc(ghost_densemat_t *vec)
{
    ghost_lidx_t v;
    if (vec->val == NULL) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&vec->val,vec->traits.nrowspadded*sizeof(char *)));

        for (v=0; v<vec->traits.nrowspadded; v++) {
            vec->val[v] = NULL;
        }
    }

    if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
        return GHOST_SUCCESS;
    }


    if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
        if (vec->val[0] == NULL) {
            DEBUG_LOG(2,"Allocating host side of vector");
            GHOST_CALL_RETURN(ghost_malloc_align((void **)&vec->val[0],vec->traits.ncolspadded*vec->traits.nrowspadded*vec->elSize,GHOST_DATA_ALIGNMENT));
            for (v=1; v<vec->traits.nrowspadded; v++) {
                vec->val[v] = vec->val[0]+v*vec->traits.ncolspadded*vec->elSize;
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
            GHOST_CALL_RETURN(ghost_cu_malloc(&vec->cu_val,vec->traits.nrowspadded*vec->traits.ncolspadded*vec->elSize));
#else
            //ghost_cu_safecall(cudaMallocPitch(&(void *)vec->cu_val,&vec->traits.nrowspadded,vec->traits.nrowshalo*sizeofdt,vec->traits.ncols));
            GHOST_CALL_RETURN(ghost_cu_malloc(&vec->cu_val,vec->traits.nrowspadded*vec->traits.ncolspadded*vec->elSize));
#endif
        }
#endif
    }   

    return GHOST_SUCCESS; 
}

static ghost_error_t vec_rm_fromVec(ghost_densemat_t *vec, ghost_densemat_t *vec2, ghost_lidx_t roffs, ghost_lidx_t coffs)
{
    ghost_densemat_rm_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from vector w/ col offset %"PRLIDX,coffs);
    ghost_lidx_t r;
    coffs += ghost_bitmap_first(vec2->ldmask);
    
    if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE) {
        roffs += ghost_bitmap_first(vec2->trmask);
    }

    if (ghost_bitmap_weight(vec->ldmask) != vec->traits.ncolsorig || 
            ghost_bitmap_weight(vec->trmask) != vec->traits.nrowsorig ||
            ghost_bitmap_weight(vec2->ldmask) != vec2->traits.ncolsorig ||
            ghost_bitmap_weight(vec2->trmask) != vec2->traits.nrowsorig || roffs || coffs) { 
        WARNING_LOG("Potentially slow fromVec operation because some rows or columns are masked out!");

        for (r=0; r<vec->traits.nrows; r++) {
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
            {
                if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE)
                {
#ifdef GHOST_HAVE_CUDA
                    ghost_cu_memcpy(CUVECVAL_RM(vec,vec->cu_val,r,ghost_bitmap_first(vec->ldmask)),CUVECVAL_RM(vec2,vec2->cu_val,roffs+r,coffs),vec->traits.ncols*vec->elSize);

#endif
                }
                else
                {
#ifdef GHOST_HAVE_CUDA
                    ghost_cu_upload(CUVECVAL_RM(vec,vec->cu_val,r,ghost_bitmap_first(vec->ldmask)),VECVAL_RM(vec2,vec2->val,roffs+r,coffs),vec->traits.ncols*vec->elSize);
#endif
                }
            }
            else
            {
                if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE)
                {
#ifdef GHOST_HAVE_CUDA
                    ghost_cu_download(VECVAL_RM(vec,vec->val,r,ghost_bitmap_first(vec->ldmask)),CUVECVAL_RM(vec2,vec2->cu_val,roffs+r,coffs),vec->traits.ncols*vec->elSize);
#endif
                }
                else
                {
                    memcpy(VECVAL_RM(vec,vec->val,r,ghost_bitmap_first(vec->ldmask)),VECVAL_RM(vec2,vec2->val,roffs+r,coffs),vec->traits.ncols*vec->elSize);
                }
            }

        }
    } else {
        if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
            if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
                ghost_cu_memcpy(vec->cu_val,vec2->cu_val,vec2->traits.ncolspadded*vec2->traits.nrows*vec->elSize);
#endif
            } else {
#ifdef GHOST_HAVE_CUDA
                ghost_cu_upload(vec->cu_val,vec2->val,vec2->traits.ncolspadded*vec2->traits.nrows*vec->elSize);
#endif
            }
        } else {
            if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
                ghost_cu_download(vec->val,vec2->cu_val,vec2->traits.ncolspadded*vec2->traits.nrows*vec->elSize);
#endif
            } else {
                memcpy(vec->val,vec2->val,vec2->traits.ncolspadded*vec2->traits.nrows*vec->elSize);
            }
        }
    }


    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale)
{
    GHOST_INSTR_START(axpy);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_lidx_t nc = MIN(vec->traits.ncols,vec2->traits.ncols);
    char *s = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->elSize),err,ret);

    ghost_lidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->elSize],scale,vec->elSize);
    }

    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_rm_vaxpy_funcs[dtIdx](vec,vec2,s),err,ret);

    goto out;
err:

out:
    free(s);
    GHOST_INSTR_STOP(axpy);
    return ret;
}

static ghost_error_t vec_rm_axpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *_b)
{
    GHOST_INSTR_START(axpby);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_lidx_t nc = MIN(vec->traits.ncols,vec2->traits.ncols);
    char *s = NULL;
    char *b = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->elSize),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&b,nc*vec->elSize),err,ret);

    ghost_lidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->elSize],scale,vec->elSize);
        memcpy(&b[i*vec->elSize],_b,vec->elSize);
    }
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_rm_vaxpby_funcs[dtIdx](vec,vec2,s,b),err,ret);

    goto out;
err:

out:
    free(s);
    free(b);
    GHOST_INSTR_STOP(axpby);
    return ret;
}

static ghost_error_t vec_rm_vaxpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_INSTR_START(vaxpy);
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    ret = ghost_densemat_rm_vaxpy_funcs[dtIdx](vec,vec2,scale);
    GHOST_INSTR_STOP(vaxpy);

    goto out;
err:
out:
    return ret;
}

static ghost_error_t vec_rm_vaxpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_INSTR_START(vaxpby);
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    ret = ghost_densemat_rm_vaxpby_funcs[dtIdx](vec,vec2,scale,b);
    GHOST_INSTR_STOP(vaxpby);
    goto out;
err:
out:
    return ret;
}

static ghost_error_t vec_rm_scale(ghost_densemat_t *vec, void *scale)
{
    GHOST_INSTR_START(scale);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_lidx_t nc = vec->traits.ncols;
    char *s;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->elSize),err,ret);

    ghost_lidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->elSize],scale,vec->elSize);
    }
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_rm_vscale_funcs[dtIdx](vec,s),err,ret);

    goto out;
err:

out:
    free(s);
    GHOST_INSTR_STOP(scale);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_vscale(ghost_densemat_t *vec, void *scale)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_INSTR_START(vscale);
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    ret = ghost_densemat_rm_vscale_funcs[dtIdx](vec,scale);
    GHOST_INSTR_STOP(vscale);

    goto out;
err:
out:
    return ret;
}

static ghost_error_t vec_rm_dotprod(ghost_densemat_t *vec, void *res, ghost_densemat_t *vec2)
{
    ghost_error_t ret = GHOST_SUCCESS;
    GHOST_INSTR_START(dot);
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_GOTO(ghost_datatype_idx(&dtIdx,vec->traits.datatype),err,ret);
    ret = ghost_densemat_rm_dotprod_funcs[dtIdx](vec,res,vec2);
    GHOST_INSTR_STOP(dot);
 
    goto out;
err:
out:
    return ret;
}

static ghost_error_t vec_rm_entry(ghost_densemat_t * vec, void *val, ghost_lidx_t r, ghost_lidx_t c) 
{
    int i = 0;
    int idx = ghost_bitmap_first(vec->ldmask);
    for (i=0; i<c; i++) {
        idx = ghost_bitmap_next(vec->ldmask,idx);
    }


    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
    {
#ifdef GHOST_HAVE_CUDA
        int ridx = ghost_bitmap_first(vec->trmask);
        for (i=0; i<r; i++) {
            ridx = ghost_bitmap_next(vec->trmask,ridx);
        }
        ghost_cu_download(val,&vec->cu_val[(ridx*vec->traits.ncolspadded+idx)*vec->elSize],vec->elSize);
#endif
    }
    else if (vec->traits.flags & GHOST_DENSEMAT_HOST)
    {
        memcpy(val,VECVAL_RM(vec,vec->val,r,idx),vec->elSize);
    }

    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_fromRand(ghost_densemat_t *vec)
{
    ghost_datatype_idx_t dtIdx;
    GHOST_CALL_RETURN(ghost_datatype_idx(&dtIdx,vec->traits.datatype));
    return ghost_densemat_rm_fromRand_funcs[dtIdx](vec);
}

static ghost_error_t vec_rm_fromScalar(ghost_densemat_t *vec, void *val)
{
    ghost_densemat_rm_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from scalar value with %"PRLIDX" rows",vec->traits.nrows);


    if (ghost_bitmap_weight(vec->ldmask) != vec->traits.ncolsorig || 
            ghost_bitmap_weight(vec->trmask) != vec->traits.nrowsorig) {
        ghost_lidx_t col,row,colidx;
        WARNING_LOG("Potentially slow fromScalar operation because some rows or columns are masked out!");
        ITER_BEGIN_RM(vec,col,row,colidx)
        memcpy(VECVAL_RM(vec,vec->val,row,col),val,vec->elSize);
        ITER_END_RM(colidx)
        vec->upload(vec);
    } else {
        ghost_lidx_t i;
        for (i=0; i<vec->traits.ncolspadded*vec->traits.nrows; i++) {
            memcpy(&vec->val[0][i*vec->elSize],val,vec->elSize);
        }
    }
        

    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_toFile(ghost_densemat_t *vec, char *path)
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
    ghost_lidx_t v;
    ghost_mpi_datatype_t mpidt;
    GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits.datatype));
    MPI_CALL_RETURN(MPI_File_set_view(fileh,4*sizeof(int32_t)+2*sizeof(int64_t),mpidt,mpidt,"native",MPI_INFO_NULL));
    MPI_Offset fileoffset = vec->context->lfRow[rank];
    ghost_lidx_t vecoffset = 0;
    for (v=0; v<vec->traits.ncols; v++) {
        char *val = NULL;
        int copied = 0;
        if (vec->traits.flags & GHOST_DENSEMAT_HOST)
        {
            vec->download(vec);
            val = VECVAL_RM(vec,vec->val,v,0);
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

    ghost_lidx_t v;
    for (v=0; v<vec->traits.ncols; v++) {
        char *val = NULL;
        int copied = 0;
        if (vec->traits.flags & GHOST_DENSEMAT_HOST)
        {
            vec->download(vec);
            val = VECVAL_RM(vec,vec->val,v,0);
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

static ghost_error_t vec_rm_fromFile(ghost_densemat_t *vec, char *path)
{
    int rank;
    GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));

    off_t offset;
    if ((vec->context == NULL) || !(vec->context->flags & GHOST_CONTEXT_DISTRIBUTED)) {
        offset = 0;
    } else {
        offset = vec->context->lfRow[rank];
    }

    ghost_densemat_rm_malloc(vec);
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
            if ((ghost_lidx_t)(ret = fread(VECVAL_RM(vec,vec->val,v,0), vec->elSize, vec->traits.nrows,filed)) != vec->traits.nrows) {
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

static ghost_error_t vec_rm_fromFunc(ghost_densemat_t *vec, void (*fp)(ghost_gidx_t, ghost_lidx_t, void *))
{
    int rank;
    ghost_gidx_t offset;
    if (vec->context) {
        GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));
        offset = vec->context->lfRow[rank];
    } else {
        rank = 0;
        offset = 0;
    }
    GHOST_CALL_RETURN(ghost_densemat_rm_malloc(vec));
    DEBUG_LOG(1,"Filling vector via function");

    ghost_lidx_t col,row,colidx;

    if (vec->traits.flags & GHOST_DENSEMAT_HOST) { // vector is stored _at least_ at host
        ITER_BEGIN_RM(vec,col,row,colidx)
        fp(offset+row,colidx,VECVAL_RM(vec,vec->val,row,col));
        ITER_END_RM(colidx)
        vec->upload(vec);
    } else {
        ghost_densemat_t *hostVec;
        ghost_densemat_traits_t htraits = vec->traits;
        htraits.flags &= ~GHOST_DENSEMAT_DEVICE;
        htraits.flags |= GHOST_DENSEMAT_HOST;
        GHOST_CALL_RETURN(ghost_densemat_create(&hostVec,vec->context,htraits));
        GHOST_CALL_RETURN(hostVec->fromFunc(hostVec,fp));

        char *foo;
        hostVec->string(hostVec,&foo);
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
    
    bool uniformstorage;
    GHOST_CALL_RETURN(ghost_densemat_uniformstorage(&uniformstorage,vec));
    if (!uniformstorage) {
        ERROR_LOG("Cannot collect vectors of different storage order");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_lidx_t c;
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
                MPI_CALL_RETURN(MPI_Isend(VECVAL_RM(vec,vec->val,c,nodeVec->context->lfRow[i]),nodeVec->context->lnrows[i],mpidt,i,i,nodeVec->context->mpicomm,&req[msgcount]));
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
    ghost_lidx_t c;
#ifdef GHOST_HAVE_MPI
    int me;
    int nprocs;
    ghost_mpi_datatype_t mpidt;
    GHOST_CALL_RETURN(ghost_rank(&me, vec->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nprocs, vec->context->mpicomm));
    GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits.datatype));
    
    bool uniformstorage;
    GHOST_CALL_RETURN(ghost_densemat_uniformstorage(&uniformstorage,vec));
    if (!uniformstorage) {
        ERROR_LOG("Cannot collect vectors of different storage order");
        return GHOST_ERR_INVALID_ARG;
    }

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
                MPI_CALL_RETURN(MPI_Irecv(VECVAL_RM(totalVec,totalVec->val,c,vec->context->lfRow[i]),vec->context->lnrows[i],mpidt,i,i,vec->context->mpicomm,&req[msgcount]));
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
            ghost_lidx_t v;
#ifdef GHOST_HAVE_CUDA_PINNEDMEM
            WARNING_LOG("CUDA pinned memory is disabled");
            /*if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
              for (v=0; v<vec->traits.ncols; v++) { 
              ghost_cu_safecall(cudaFreeHost(vec->val[v]));
              }
              }*/
            if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
                for (v=0; v<vec->traits.nrows; v++) {
                    free(vec->val[v]); vec->val[v] = NULL;
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
                for (v=0; v<vec->traits.nrows; v++) {
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
        ghost_bitmap_free(vec->ldmask);
        ghost_bitmap_free(vec->trmask);
        free(vec);
        // TODO free traits ???
    }
}
static ghost_error_t ghost_permuteVector( ghost_densemat_t* vec, ghost_permutation_t *permutation, ghost_permutation_direction_t dir) 
{
    // TODO enhance performance
    
    if (!permutation) {
        return GHOST_SUCCESS;
    }

    ghost_lidx_t i;
    ghost_lidx_t len, c;
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

    ghost_gidx_t *perm = NULL;
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
                ERROR_LOG("Permutation index out of bounds: %"PRGIDX" > %"PRLIDX,perm[i],len);
                free(tmp);
                return GHOST_ERR_UNKNOWN;
            }

            memcpy(&tmp[vec->elSize*perm[i]],VECVAL_RM(permvec,permvec->val,i,c),permvec->elSize);
        }
        for(i=0; i < len; ++i) {
            memcpy(VECVAL_RM(permvec,permvec->val,i,c),&tmp[permvec->elSize*i],permvec->elSize);
        }
        free(tmp);
    }
    
    if (permutation->scope == GHOST_PERMUTATION_GLOBAL && vec->traits.nrows != permutation->len) {
        permvec->distribute(permvec,vec);
        permvec->destroy(permvec);
    }

    return GHOST_SUCCESS;
}

static ghost_error_t ghost_cloneVector(ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nr, ghost_lidx_t roffs, ghost_lidx_t nc, ghost_lidx_t coffs)
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

static ghost_error_t vec_rm_compress(ghost_densemat_t *vec)
{
    if (!(vec->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        return GHOST_SUCCESS;
    }

    if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
        ghost_lidx_t v,i,r,c;

        char *val = NULL;
        GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits.nrowspadded*vec->traits.ncolspadded*vec->elSize));

#pragma omp parallel for schedule(runtime) private(v)
        for (i=0; i<vec->traits.nrowspadded; i++)
        {
            for (v=0; v<vec->traits.ncolspadded; v++)
            {
                val[(v*vec->traits.nrowspadded+i)*vec->elSize] = 0;
            }
        }


        for (r=0; r<vec->traits.nrows; r++)
        {
            for (v=0,c=0; v<vec->traits.ncolsorig; v++)
            {
                if (ghost_bitmap_isset(vec->ldmask,v)) {
                    memcpy(&val[(r*vec->traits.ncolspadded+c)*vec->elSize],
                            VECVAL_RM(vec,vec->val,r,v),vec->elSize);
                    c++;

                }
            }
            if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW))
            {
                free(vec->val[r]);
            }
            vec->val[r] = &val[(r*vec->traits.ncolspadded)*vec->elSize];
        }
    }
    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        ghost_lidx_t v,i,r,j;

        char *cu_val;
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_val,vec->traits.nrowspadded*vec->traits.ncols*vec->elSize));

        for (v=0, i=0; v<vec->traits.ncolsorig; v++) {
            if (ghost_bitmap_isset(vec->ldmask,v)) {
                for (r=0, j=0; r<vec->traits.nrowsorig; r++) {
                    if (ghost_bitmap_isset(vec->trmask,r)) {
                        ghost_cu_memcpy(&cu_val[(j*vec->traits.ncolspadded+i)*vec->elSize],
                                &vec->cu_val[(r*vec->traits.ncolspadded+v)*vec->elSize],
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
    ghost_bitmap_set_range(vec->ldmask,0,vec->traits.ncols-1);
    ghost_bitmap_set_range(vec->trmask,0,vec->traits.nrows-1);
    vec->traits.ncolsorig = vec->traits.ncols;
    vec->traits.nrowsorig = vec->traits.nrows;
    vec->traits.flags &= ~GHOST_DENSEMAT_VIEW;
    vec->traits.flags &= ~GHOST_DENSEMAT_SCATTERED;
    vec->viewing = NULL;

    return GHOST_SUCCESS;
}
/*
ghost_error_t ghost_densemat_traits_clone(ghost_densemat_traits_t *t1, ghost_densemat_traits_t **t2)
{
    GHOST_CALL_RETURN(ghost_malloc((void **)t2,sizeof(ghost_densemat_traits_t)));
    memcpy(*t2,t1,sizeof(ghost_densemat_traits_t));

    return GHOST_SUCCESS;
}*/
