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

#define ROWMAJOR
#include "ghost/densemat_iter_macros.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

static ghost_error_t vec_rm_scale(ghost_densemat_t *vec, void *scale);
static ghost_error_t vec_rm_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale);
static ghost_error_t vec_rm_axpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *b);
static ghost_error_t vec_rm_fromFunc(ghost_densemat_t *vec, void (*fp)(ghost_gidx_t, ghost_lidx_t, void *));
static ghost_error_t vec_rm_fromVec(ghost_densemat_t *vec, ghost_densemat_t *vec2, ghost_lidx_t roffs, ghost_lidx_t coffs);
static ghost_error_t vec_rm_fromScalar(ghost_densemat_t *vec, void *val);
static ghost_error_t vec_rm_fromFile(ghost_densemat_t *vec, char *path, bool singleFile);
static ghost_error_t vec_rm_toFile(ghost_densemat_t *vec, char *path, bool singleFile);
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
        vec->dot = &ghost_densemat_rm_dotprod_selector;
        vec->vaxpy = &ghost_densemat_rm_vaxpy_selector;
        vec->vaxpby = &ghost_densemat_rm_vaxpby_selector;
        vec->axpy = &vec_rm_axpy;
        vec->axpby = &vec_rm_axpby;
        vec->scale = &vec_rm_scale;
        vec->vscale = &ghost_densemat_rm_vscale_selector;
        vec->fromScalar = &vec_rm_fromScalar;
        vec->fromRand = &ghost_densemat_rm_fromRand_selector;
    }

    vec->memtranspose = &vec_rm_memtranspose;
    vec->compress = &vec_rm_compress;
    vec->string = &ghost_densemat_rm_string_selector;
    vec->fromFunc = &vec_rm_fromFunc;
    vec->fromVec = &vec_rm_fromVec;
    vec->fromFile = &vec_rm_fromFile;
    vec->toFile = &vec_rm_toFile;
    vec->distribute = &ghost_distributeVector;
    vec->collect = &ghost_collectVectors;
    vec->normalize = &ghost_densemat_rm_normalize_selector;
    vec->destroy = &ghost_freeVector;
    vec->permute = &ghost_permuteVector;
    vec->clone = &ghost_cloneVector;
    vec->entry = &vec_rm_entry;
    vec->viewVec = &vec_rm_view;
    vec->viewPlain = &vec_rm_viewPlain;
    vec->viewScatteredVec = &vec_rm_viewScatteredVec;
    vec->viewScatteredCols = &vec_rm_viewScatteredCols;
    vec->viewCols = &vec_rm_viewCols;
    
    vec->averageHalo = &ghost_densemat_rm_averagehalo_selector;

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
        ghost_bitmap_set_range(vec->ldmask,0,vec->traits.nrows-1);
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
            ghost_bitmap_set_range(vec->ldmask,0,vec->traits.nrows-1);

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
    if (!((vec->traits.flags & GHOST_DENSEMAT_HOST) && 
                (vec->traits.flags & GHOST_DENSEMAT_DEVICE))) {
        return GHOST_SUCCESS;
    }
    if (DENSEMAT_COMPACT(vec)) {
        GHOST_CALL_RETURN(ghost_cu_upload(
                    DENSEMAT_CUVAL(vec,vec->traits.nrows,0),
                    DENSEMAT_VAL(vec,vec->traits.nrows,0), 
                    (vec->traits.nrowshalo-vec->traits.nrows)*
                    vec->traits.ncolspadded*vec->elSize));
    } else {
        int col, memcol = -1;

        for (col=0; col<vec->traits.ncols; col++) {
            memcol = ghost_bitmap_next(vec->ldmask,memcol);
            GHOST_CALL_RETURN(ghost_cu_upload2d(
                        DENSEMAT_CUVAL(vec,vec->traits.nrowsorig,memcol),
                        vec->traits.ncolspadded*vec->elSize,
                        DENSEMAT_VAL(vec,vec->traits.nrows,memcol),
                        vec->traits.ncolspadded*vec->elSize,vec->elSize,
                        vec->context->halo_elements));
        }
    }
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_downloadHalo(ghost_densemat_t *vec)
{
    if (!((vec->traits.flags & GHOST_DENSEMAT_HOST) && 
                (vec->traits.flags & GHOST_DENSEMAT_DEVICE))) {
        return GHOST_SUCCESS;
    }
    if (DENSEMAT_COMPACT(vec)) {
        GHOST_CALL_RETURN(ghost_cu_download(
                    DENSEMAT_VAL(vec,vec->traits.nrows,0), 
                    DENSEMAT_CUVAL(vec,vec->traits.nrows,0),
                    (vec->traits.nrowshalo-vec->traits.nrows)*
                    vec->traits.ncolspadded*vec->elSize));
    } else {
        int col, memcol = -1;

        for (col=0; col<vec->traits.ncols; col++) {
            memcol = ghost_bitmap_next(vec->ldmask,memcol);
            GHOST_CALL_RETURN(ghost_cu_download2d(
                        DENSEMAT_VAL(vec,vec->traits.nrows,memcol),
                        vec->traits.ncolspadded*vec->elSize,
                        DENSEMAT_CUVAL(vec,vec->traits.nrowsorig,memcol),
                        vec->traits.ncolspadded*vec->elSize,
                        vec->elSize,vec->context->halo_elements));
        }
    }
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_uploadNonHalo(ghost_densemat_t *vec)
{
    if (!((vec->traits.flags & GHOST_DENSEMAT_HOST) && 
                (vec->traits.flags & GHOST_DENSEMAT_DEVICE))) {
        return GHOST_SUCCESS;
    }
    if (DENSEMAT_COMPACT(vec)) {
        GHOST_CALL_RETURN(ghost_cu_upload(
                    DENSEMAT_CUVAL(vec,0,0),
                    DENSEMAT_VAL(vec,0,0), 
                    vec->traits.nrows*vec->traits.ncolspadded*vec->elSize));
    } else {
        DENSEMAT_ITER(vec,ghost_cu_upload(
                    DENSEMAT_CUVAL(vec,memrow,memcol),
                    DENSEMAT_VAL(vec,row,memcol),
                    vec->elSize));
    }
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_downloadNonHalo(ghost_densemat_t *vec)
{
    if (!((vec->traits.flags & GHOST_DENSEMAT_HOST) && 
                (vec->traits.flags & GHOST_DENSEMAT_DEVICE))) {
        return GHOST_SUCCESS;
    }
    if (DENSEMAT_COMPACT(vec)) {
        GHOST_CALL_RETURN(ghost_cu_download(
                    DENSEMAT_VAL(vec,0,0), 
                    DENSEMAT_CUVAL(vec,0,0),
                    vec->traits.nrows*vec->traits.ncolspadded*vec->elSize));
    } else {
        DENSEMAT_ITER(vec,ghost_cu_download(
                    DENSEMAT_VAL(vec,row,memcol),
                    DENSEMAT_CUVAL(vec,memrow,memcol),
                    vec->elSize));
    }
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_upload(ghost_densemat_t *vec) 
{
    GHOST_CALL_RETURN(vec->uploadNonHalo(vec));
    GHOST_CALL_RETURN(vec->uploadHalo(vec));
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_download(ghost_densemat_t *vec)
{
    GHOST_CALL_RETURN(vec->downloadNonHalo(vec));
    GHOST_CALL_RETURN(vec->downloadHalo(vec));
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_view (ghost_densemat_t *src, ghost_densemat_t **new, 
        ghost_lidx_t nr, ghost_lidx_t roffs, ghost_lidx_t nc, ghost_lidx_t coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" dense matrix with col offset %"
            PRLIDX,src->traits.nrows,nc,coffs);

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
            (*new)->val[r] = DENSEMAT_VAL(src,roffs+r,0);
        }
    }

    (*new)->viewing = src;
    (*new)->viewing_col = coffs;
    (*new)->viewing_row = roffs;
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_viewPlain (ghost_densemat_t *vec, void *data, 
        ghost_lidx_t roffs, ghost_lidx_t coffs, ghost_lidx_t lda)
{
    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" dense matrix from plain data "
            "with offset %"PRLIDX"x%"PRLIDX,
            vec->traits.nrows,vec->traits.ncols,roffs,coffs);

    ghost_lidx_t v;
    ghost_densemat_rm_malloc(vec);

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        INFO_LOG("The plain memory has to be valid CUDA device memory!");
        INFO_LOG("The column offset is being ignored!");
        vec->cu_val = &((char *)data)[lda*roffs*vec->elSize];
        vec->traits.ncolspadded = vec->traits.ncols;
#endif
    } else {
        for (v=0; v<vec->traits.nrows; v++) {
            vec->val[v] = &((char *)data)[(lda*(roffs+v)+coffs)*vec->elSize];
        }
    }
    vec->traits.flags |= GHOST_DENSEMAT_VIEW;
    ghost_bitmap_set_range(vec->ldmask,0,vec->traits.ncolsorig);
    ghost_bitmap_set_range(vec->trmask,0,vec->traits.nrowsorig);
    vec->traits.ncolsorig = vec->traits.ncols;
    vec->traits.nrowsorig = vec->traits.nrows;

    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_viewScatteredVec (ghost_densemat_t *src, 
        ghost_densemat_t **new, ghost_lidx_t nr, ghost_lidx_t *roffs, 
        ghost_lidx_t nc, ghost_lidx_t *coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" scattered dense matrix",
            src->traits.nrows,nc);

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
            (*new)->val[r] = DENSEMAT_VAL(src,roffs[r],0);
        }
    }

    (*new)->viewing = src;
    (*new)->viewing_col = -1;
    (*new)->viewing_row = -1;

    return GHOST_SUCCESS;
}


static ghost_error_t vec_rm_viewCols (ghost_densemat_t *src, 
        ghost_densemat_t **new, ghost_lidx_t nc, ghost_lidx_t coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" contiguous dense matrix",
            src->traits.nrows,nc);

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
        (*new)->val[r] = DENSEMAT_VAL(src,r,0);
    }

    (*new)->viewing = src;
    (*new)->viewing_col = coffs;
    (*new)->viewing_row = 0;
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_viewScatteredCols (ghost_densemat_t *src, 
        ghost_densemat_t **new, ghost_lidx_t nc, ghost_lidx_t *coffs)
{
    if (!array_strictly_ascending(coffs,nc)) {
        ERROR_LOG("Can only view sctrictly ascending scattered columns "
                "for row-major densemats!");
        return GHOST_ERR_INVALID_ARG;
    }

    DEBUG_LOG(1,"Viewing a %"PRLIDX"x%"PRLIDX" scattered dense matrix",
            src->traits.nrows,nc);

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
   
    for (viewedcol=-1, c=0, i=0; c<src->traits.ncolsorig; c++) {
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
        (*new)->val[r] = DENSEMAT_VAL(src,r,0);
    }

    (*new)->viewing = src;
    (*new)->viewing_col = -1;
    (*new)->viewing_row = -1;
    return GHOST_SUCCESS;
}

ghost_error_t ghost_densemat_rm_malloc(ghost_densemat_t *vec)
{
    ghost_lidx_t v;
    if (vec->val == NULL) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&vec->val,
                    vec->traits.nrowspadded*sizeof(char *)));

        for (v=0; v<vec->traits.nrowspadded; v++) {
            vec->val[v] = NULL;
        }
    }

    if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
        return GHOST_SUCCESS;
    }


    if ((vec->traits.flags & GHOST_DENSEMAT_HOST) && !vec->val[0]) {
        DEBUG_LOG(2,"Allocating host side of vector");
        if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
            GHOST_CALL_RETURN(ghost_malloc_pinned((void **)&vec->val[0],
                        (size_t)vec->traits.ncolspadded*vec->traits.nrowspadded*
                        vec->elSize));
        } else {
            GHOST_CALL_RETURN(ghost_malloc_align((void **)&vec->val[0],
                        (size_t)vec->traits.ncolspadded*vec->traits.nrowspadded*
                        vec->elSize,GHOST_DATA_ALIGNMENT));
        }

        for (v=1; v<vec->traits.nrowspadded; v++) {
            vec->val[v] = vec->val[0]+v*(size_t)vec->traits.ncolspadded*
                vec->elSize;
        }
    }

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        DEBUG_LOG(2,"Allocating device side of vector");
#ifdef GHOST_HAVE_CUDA
        if (vec->cu_val == NULL) {
            GHOST_CALL_RETURN(ghost_cu_malloc(&vec->cu_val,
                        vec->traits.nrowspadded*vec->traits.ncolspadded*
                        vec->elSize));
        }
#endif
    }   

    return GHOST_SUCCESS; 
}

static ghost_error_t vec_rm_fromVec(ghost_densemat_t *vec, 
        ghost_densemat_t *vec2, ghost_lidx_t roffs, ghost_lidx_t coffs)
{
    ghost_densemat_rm_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from vector w/ col offset %"PRLIDX,coffs);

    if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
        if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE) {
            DENSEMAT_ITER2_OFFS(vec,vec2,roffs,coffs,ghost_cu_memcpy(
                        DENSEMAT_CUVAL(vec,memrow1,memcol1),
                        DENSEMAT_CUVAL(vec2,memrow2,memcol2),vec->elSize));
        } else {
            DENSEMAT_ITER2_OFFS(vec,vec2,roffs,coffs,ghost_cu_upload(
                        DENSEMAT_CUVAL(vec,memrow1,memcol1),
                        DENSEMAT_VAL(vec2,row+roffs,memcol2),vec->elSize));
        }
    } else {
        if (vec2->traits.flags & GHOST_DENSEMAT_DEVICE) {
            DENSEMAT_ITER2_OFFS(vec,vec2,roffs,coffs,ghost_cu_download(
                        DENSEMAT_VAL(vec,row,memcol1),
                        DENSEMAT_CUVAL(vec2,memrow2,memcol2),vec->elSize));
        } else {
            DENSEMAT_ITER2_OFFS(vec,vec2,roffs,coffs,memcpy(
                        DENSEMAT_VAL(vec,row,memcol1),
                        DENSEMAT_VAL(vec2,row+roffs,memcol2),vec->elSize));
        }
    }

    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_axpy(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_lidx_t nc = MIN(vec->traits.ncols,vec2->traits.ncols);
    char *s = NULL;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->elSize),err,ret);

    ghost_lidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->elSize],scale,vec->elSize);
    }

    ret = vec->vaxpy(vec,vec2,s);

    goto out;
err:

out:
    free(s);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}

static ghost_error_t vec_rm_axpby(ghost_densemat_t *vec, ghost_densemat_t *vec2, void *scale, void *_b)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
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

    ret = vec->vaxpby(vec,vec2,s,b);

    goto out;
err:

out:
    free(s);
    free(b);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return ret;
}

static ghost_error_t vec_rm_scale(ghost_densemat_t *vec, void *scale)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_MATH);
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_lidx_t nc = vec->traits.ncols;
    char *s;
    GHOST_CALL_GOTO(ghost_malloc((void **)&s,nc*vec->elSize),err,ret);

    ghost_lidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*vec->elSize],scale,vec->elSize);
    }

    ret = vec->vscale(vec,s);

    goto out;
err:

out:
    free(s);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_MATH);
    return GHOST_SUCCESS;
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
        memcpy(val,DENSEMAT_VAL(vec,r,idx),vec->elSize);
    }

    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_fromScalar(ghost_densemat_t *vec, void *val)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    
    ghost_densemat_rm_malloc(vec);

    DENSEMAT_ITER(vec,memcpy(DENSEMAT_VAL(vec,row,memcol),val,vec->elSize));
    
    vec->upload(vec);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_toFile(ghost_densemat_t *vec, char *path, bool singleFile)
{ 

#ifndef GHOST_HAVE_MPI
    singleFile = false;
#endif

    if (singleFile) {
#ifdef GHOST_HAVE_MPI
        int rank;
        GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));

        int32_t endianess = ghost_machine_bigendian();
        int32_t version = 1;
        int32_t order = GHOST_BINDENSEMAT_ORDER_ROW_FIRST;
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
        MPI_Offset fileoffset = vec->context->lfRow[rank]*vec->traits.ncols;
        ghost_lidx_t vecoffset = 0;
        for (v=0; v<vec->traits.nrows; v++) {
            char *val = NULL;
            int copied = 0;
            if (vec->traits.flags & GHOST_DENSEMAT_HOST)
            {
                vec->download(vec);
                val = DENSEMAT_VAL(vec,v,0);
            }
            else if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
            {
#ifdef GHOST_HAVE_CUDA
                GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits.nrows*vec->elSize));
                copied = 1;
                ghost_cu_download(val,&vec->cu_val[v*vec->traits.nrowspadded*vec->elSize],vec->traits.nrows*vec->elSize);
#endif
            }
            MPI_CALL_RETURN(MPI_File_write_at(fileh,fileoffset,val,vec->traits.ncols,mpidt,&status));
            fileoffset += ncols;
            vecoffset += vec->traits.ncolspadded*vec->elSize;
            if (copied) {
                free(val);
            }
        }
        MPI_CALL_RETURN(MPI_File_close(&fileh));


#endif
    } else {

        DEBUG_LOG(1,"Writing (local) vector to file %s",path);
        size_t ret;

        int32_t endianess = ghost_machine_bigendian();
        int32_t version = 1;
        int32_t order = GHOST_BINDENSEMAT_ORDER_ROW_FIRST;
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
        for (v=0; v<vec->traits.nrows; v++) {
            char *val = NULL;
            int copied = 0;
            if (vec->traits.flags & GHOST_DENSEMAT_HOST)
            {
                vec->download(vec);
                val = DENSEMAT_VAL(vec,v,0);
            }
            else if (vec->traits.flags & GHOST_DENSEMAT_DEVICE)
            {
#ifdef GHOST_HAVE_CUDA
                GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits.ncols*vec->elSize));
                copied = 1;
                ghost_cu_download(val,&vec->cu_val[v*vec->traits.ncolspadded*vec->elSize],vec->traits.ncols*vec->elSize);
#endif
            }

            if ((ret = fwrite(val, vec->elSize, vec->traits.ncols,filed)) != (size_t)vec->traits.ncols) {
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
    }

    return GHOST_SUCCESS;

}

static ghost_error_t vec_rm_fromFile(ghost_densemat_t *vec, char *path, bool singleFile)
{
    
#ifndef GHOST_HAVE_MPI
    singleFile = false;
#endif

    int rank;
    GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));

    off_t offset;
    if ((vec->context == NULL) || !(vec->context->flags & GHOST_CONTEXT_DISTRIBUTED) || !singleFile) {
        offset = 0;
    } else {
        offset = vec->context->lfRow[rank]*vec->traits.ncols;
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
    if (order != GHOST_BINDENSEMAT_ORDER_ROW_FIRST) {
        ERROR_LOG("Can only read row-major files!");
        return GHOST_ERR_IO;
    }

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

    if (!singleFile && (vec->traits.nrows != nrows)) {
        ERROR_LOG("The number of rows does not match between the file and the densemat!");
        return GHOST_ERR_IO;
    }
    if (singleFile && (vec->context->gnrows != nrows)) {
        ERROR_LOG("The number of rows does not match between the file and the densemat's context!");
        return GHOST_ERR_IO;
    }

    if (fseeko(filed,offset*vec->elSize,SEEK_CUR)) {
        ERROR_LOG("seek failed");
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    int v;
    for (v=0; v<vec->traits.nrows; v++) {
        if (vec->traits.flags & GHOST_DENSEMAT_HOST)
        {
            if ((ghost_lidx_t)(ret = fread(DENSEMAT_VAL(vec,v,0), vec->elSize, vec->traits.ncols,filed)) != vec->traits.ncols) {
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
            if ((ret = fread(val, vec->elSize, vec->traits.ncols,filed)) != vec->traits.ncols) {
                ERROR_LOG("fread failed: %zu",ret);
                vec->destroy(vec);
                return GHOST_ERR_IO;
            }
            ghost_cu_upload(&vec->cu_val[v*vec->traits.ncolspadded*vec->elSize],val,vec->traits.ncols*vec->elSize);
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
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
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


    if (vec->traits.flags & GHOST_DENSEMAT_HOST) { // vector is stored _at least_ at host
        DENSEMAT_ITER(vec,fp(offset+row,col,DENSEMAT_VAL(vec,row,memcol)));
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
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
                MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VAL(vec,c,nodeVec->context->lfRow[i]),nodeVec->context->lnrows[i],mpidt,i,i,nodeVec->context->mpicomm,&req[msgcount]));
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
                MPI_CALL_RETURN(MPI_Irecv(DENSEMAT_VAL(totalVec,c,vec->context->lfRow[i]),vec->context->lnrows[i],mpidt,i,i,vec->context->mpicomm,&req[msgcount]));
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
                if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
#ifdef GHOST_HAVE_CUDA
                    ghost_cu_free_host(vec->val[0]);
#endif
                } else {
                    free(vec->val[0]); vec->val[0] = NULL;
                }
            }
#ifdef GHOST_HAVE_CUDA
            if (vec->traits.flags & GHOST_DENSEMAT_DEVICE) {
                ghost_cu_free(vec->cu_val);
            }
#endif
        }
        free(vec->val); vec->val = NULL;
        ghost_bitmap_free(vec->ldmask);
        ghost_bitmap_free(vec->trmask);
#ifdef GHOST_HAVE_MPI
        MPI_Type_free(&vec->row_mpidt);
#endif
        free(vec);
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

            memcpy(&tmp[vec->elSize*perm[i]],DENSEMAT_VAL(permvec,i,c),permvec->elSize);
        }
        for(i=0; i < len; ++i) {
            memcpy(DENSEMAT_VAL(permvec,i,c),&tmp[permvec->elSize*i],permvec->elSize);
        }
        free(tmp);
    }
    
    if (permutation->scope == GHOST_PERMUTATION_GLOBAL && vec->traits.nrows != permutation->len) {
        permvec->distribute(permvec,vec);
        permvec->destroy(permvec);
    }
    
    if (dir == GHOST_PERMUTATION_ORIG2PERM) {
        vec->traits.flags |= GHOST_DENSEMAT_PERMUTED;
    } else {
        vec->traits.flags &= ~GHOST_DENSEMAT_PERMUTED;
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
    (*new)->traits.flags &= ~GHOST_DENSEMAT_SCATTERED;

    (*new)->fromVec(*new,src,roffs,coffs);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_compress(ghost_densemat_t *vec)
{
    if (!(vec->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        return GHOST_SUCCESS;
    }

    if (vec->traits.flags & GHOST_DENSEMAT_HOST) {
        ghost_lidx_t v,i;

        char *val = NULL;
        GHOST_CALL_RETURN(ghost_malloc((void **)&val,vec->traits.nrowspadded*vec->traits.ncolspadded*vec->elSize));

#pragma omp parallel for schedule(runtime) private(v)
        for (i=0; i<vec->traits.nrowspadded; i++) {
            for (v=0; v<vec->traits.ncolspadded; v++) {
                val[(v*vec->traits.nrowspadded+i)*vec->elSize] = 0;
            }
        }
        
        DENSEMAT_ITER(vec,memcpy(&val[(row*vec->traits.ncolspadded+col)*vec->elSize],
                    DENSEMAT_VAL(vec,row,memcol),vec->elSize));
      
        for (row=0; row<vec->traits.nrows; row++) {
            if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
                free(vec->val[row]);
            }
            vec->val[row] = &val[(row*vec->traits.ncolspadded)*vec->elSize];
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
