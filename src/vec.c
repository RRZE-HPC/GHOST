#define _XOPEN_SOURCE 500 
#include <ghost_config.h>
#include <ghost_types.h>
#include <ghost_vec.h>
#include <ghost_util.h>
#include <ghost_constants.h>
#include <ghost_affinity.h>

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

void (*ghost_normalizeVector_funcs[4]) (ghost_vec_t *) = 
{&s_ghost_normalizeVector, &d_ghost_normalizeVector, &c_ghost_normalizeVector, &z_ghost_normalizeVector};

void (*ghost_vec_dotprod_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*) = 
{&s_ghost_vec_dotprod, &d_ghost_vec_dotprod, &c_ghost_vec_dotprod, &z_ghost_vec_dotprod};

void (*ghost_vec_vscale_funcs[4]) (ghost_vec_t *, void*) = 
{&s_ghost_vec_vscale, &d_ghost_vec_vscale, &c_ghost_vec_vscale, &z_ghost_vec_vscale};

void (*ghost_vec_vaxpy_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*) = 
{&s_ghost_vec_vaxpy, &d_ghost_vec_vaxpy, &c_ghost_vec_vaxpy, &z_ghost_vec_vaxpy};

void (*ghost_vec_vaxpby_funcs[4]) (ghost_vec_t *, ghost_vec_t *, void*, void*) = 
{&s_ghost_vec_vaxpby, &d_ghost_vec_vaxpby, &c_ghost_vec_vaxpby, &z_ghost_vec_vaxpby};

void (*ghost_vec_fromRand_funcs[4]) (ghost_vec_t *) = 
{&s_ghost_vec_fromRand, &d_ghost_vec_fromRand, &c_ghost_vec_fromRand, &z_ghost_vec_fromRand};

void (*ghost_vec_print_funcs[4]) (ghost_vec_t *) = 
{&s_ghost_printVector, &d_ghost_printVector, &c_ghost_printVector, &z_ghost_printVector};

static void vec_print(ghost_vec_t *vec);
static void vec_scale(ghost_vec_t *vec, void *scale);
static void vec_vscale(ghost_vec_t *vec, void *scale);
static void vec_vaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale);
static void vec_vaxpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b);
static void vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale);
static void vec_axpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b);
static void vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res);
static void vec_fromFunc(ghost_vec_t *vec, void (*fp)(int,int,void *));
static void vec_fromVec(ghost_vec_t *vec, ghost_vec_t *vec2, ghost_vidx_t coffs);
static void vec_fromRand(ghost_vec_t *vec);
static void vec_fromScalar(ghost_vec_t *vec, void *val);
static void vec_fromFile(ghost_vec_t *vec, char *path);
static void vec_toFile(ghost_vec_t *vec, char *path);
static void ghost_zeroVector(ghost_vec_t *vec);
static void ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2);
static void ghost_normalizeVector( ghost_vec_t *vec);
static void ghost_distributeVector(ghost_vec_t *vec, ghost_vec_t *nodeVec);
static void ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec); 
static void ghost_freeVector( ghost_vec_t* const vec );
static void ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm); 
static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src, ghost_vidx_t, ghost_vidx_t);
static void vec_entry(ghost_vec_t *, ghost_vidx_t, ghost_vidx_t, void *);
static ghost_vec_t * vec_view (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t coffs);
static ghost_vec_t * vec_viewScatteredVec (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t *coffs);
static void vec_viewPlain (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda);
static void vec_upload(ghost_vec_t *vec);
static void vec_download(ghost_vec_t *vec);
static void vec_uploadHalo(ghost_vec_t *vec);
static void vec_downloadHalo(ghost_vec_t *vec);
static void vec_uploadNonHalo(ghost_vec_t *vec);
static void vec_downloadNonHalo(ghost_vec_t *vec);

ghost_vec_t *ghost_createVector(ghost_context_t *ctx, ghost_vtraits_t *traits)
{
    ghost_vidx_t v;
    ghost_vec_t *vec = (ghost_vec_t *)ghost_malloc(sizeof(ghost_vec_t));
    vec->context = ctx;
    vec->traits = traits;
    getNrowsFromContext(vec);

    DEBUG_LOG(1,"The vector has %"PRvecIDX" sub-vectors with %"PRvecIDX" rows and %zu bytes per entry",traits->nvecs,traits->nrows,ghost_sizeofDataType(vec->traits->datatype));
    DEBUG_LOG(1,"Initializing vector");

    if (!(vec->traits->flags & (GHOST_VEC_HOST | GHOST_VEC_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting vector placement");
        vec->traits->flags |= GHOST_VEC_HOST;
#if GHOST_HAVE_CUDA
        vec->traits->flags |= GHOST_VEC_DEVICE;
#endif
    }

    if (vec->traits->flags & GHOST_VEC_DEVICE)
    {
#if GHOST_HAVE_CUDA
        vec->dotProduct = &ghost_vec_cu_dotprod;
        vec->vaxpy = &ghost_vec_cu_vaxpy;
        vec->vaxpby = &ghost_vec_cu_vaxpby;
        vec->axpy = &ghost_vec_cu_axpy;
        vec->axpby = &ghost_vec_cu_axpby;
        vec->scale = &ghost_vec_cu_scale;
        vec->vscale = &ghost_vec_cu_vscale;
        vec->fromScalar = &ghost_vec_cu_fromScalar;
        vec->fromRand = &ghost_vec_cu_fromRand;
#endif
    }
    else
    {
        vec->dotProduct = &vec_dotprod;
        vec->vaxpy = &vec_vaxpy;
        vec->vaxpby = &vec_vaxpby;
        vec->axpy = &vec_axpy;
        vec->axpby = &vec_axpby;
        vec->scale = &vec_scale;
        vec->vscale = &vec_vscale;
        vec->fromScalar = &vec_fromScalar;
        vec->fromRand = &vec_fromRand;
    }


    vec->print = &vec_print;
    vec->fromFunc = &vec_fromFunc;
    vec->fromVec = &vec_fromVec;
    vec->fromFile = &vec_fromFile;
    vec->toFile = &vec_toFile;
    vec->zero = &ghost_zeroVector;
    vec->distribute = &ghost_distributeVector;
    vec->collect = &ghost_collectVectors;
    vec->swap = &ghost_swapVectors;
    vec->normalize = &ghost_normalizeVector;
    vec->destroy = &ghost_freeVector;
    vec->permute = &ghost_permuteVector;
    vec->clone = &ghost_cloneVector;
    vec->entry = &vec_entry;
    vec->viewVec = &vec_view;
    vec->viewPlain = &vec_viewPlain;
    vec->viewScatteredVec = &vec_viewScatteredVec;

    vec->upload = &vec_upload;
    vec->download = &vec_download;
    vec->uploadHalo = &vec_uploadHalo;
    vec->downloadHalo = &vec_downloadHalo;
    vec->uploadNonHalo = &vec_uploadNonHalo;
    vec->downloadNonHalo = &vec_downloadNonHalo;
#ifdef GHOST_HAVE_CUDA
    if (vec->traits->flags & GHOST_VEC_DEVICE) {
        vec->CU_val = NULL;
    }
#elif defined(OPENCL)
    vec->CL_val_gpu = NULL;
    if (!(vec->traits->flags & (GHOST_VEC_HOST | GHOST_VEC_DEVICE))) { // no storage specified
        vec->traits->flags |= (GHOST_VEC_HOST | GHOST_VEC_DEVICE);
    }
#endif

    // TODO free val of vec only if scattered (but do not free val[0] of course!)
    vec->val = (char **)ghost_malloc(vec->traits->nvecs*sizeof(char *));

    for (v=0; v<vec->traits->nvecs; v++) {
        vec->val[v] = NULL;
    }



    return vec;
}

static void vec_uploadHalo(ghost_vec_t *vec)
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Uploading halo elements of vector");
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CU_copyHostToDeviceOffset(&vec->CU_val[vec->traits->nrowspadded*v*ghost_sizeofDataType(vec->traits->datatype)],
                    VECVAL(vec,vec->val,v,vec->traits->nrows), 
                    vec->context->communicator->halo_elements*ghost_sizeofDataType(vec->traits->datatype),
                    vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
#ifdef GHOST_HAVE_OPENCL
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CL_copyHostToDeviceOffset(VECVAL(vec,vec->CL_val_gpu,v,0),VECVAL(vec,vec->val,v,vec->traits->nrows), vec->context->communicator->halo_elements*ghost_sizeofDataType(vec->traits->datatype),    vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
    }
}

static void vec_downloadHalo(ghost_vec_t *vec)
{

    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Downloading halo elements of vector");
        WARNING_LOG("Not yet implemented!");
    }
}
static void vec_uploadNonHalo(ghost_vec_t *vec)
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Uploading %d rows of vector",vec->traits->nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CU_copyHostToDevice(&vec->CU_val[vec->traits->nrowspadded*v*ghost_sizeofDataType(vec->traits->datatype)],VECVAL(vec,vec->val,v,0), vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
#ifdef GHOST_HAVE_OPENCL
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CL_copyHostToDevice(VECVAL(vec,vec->CL_val_gpu,v,0),VECVAL(vec,vec->val,v,0), vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
    }
}

static void vec_downloadNonHalo(ghost_vec_t *vec)
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Downloading vector");
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CU_copyDeviceToHost(VECVAL(vec,vec->val,v,0),&vec->CU_val[vec->traits->nrowspadded*v*ghost_sizeofDataType(vec->traits->datatype)],vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
#ifdef GHOST_HAVE_OPENCL
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CL_copyDeviceToHost(VECVAL(vec,vec->val,v,0),VECVAL(vec,vec->CL_val_gpu,v,0), vec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
    }
}

static void vec_upload(ghost_vec_t *vec) 
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Uploading %d rows of vector",vec->traits->nrowshalo);
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CU_copyHostToDevice(&vec->CU_val[vec->traits->nrowspadded*v*ghost_sizeofDataType(vec->traits->datatype)],VECVAL(vec,vec->val,v,0), vec->traits->nrowshalo*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
#ifdef GHOST_HAVE_OPENCL
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CL_copyHostToDevice(VECVAL(vec,vec->CL_val_gpu,v,0),VECVAL(vec,vec->val,v,0), vec->traits->nrowshalo*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
    }
}

static void vec_download(ghost_vec_t *vec)
{
    if ((vec->traits->flags & GHOST_VEC_HOST) && (vec->traits->flags & GHOST_VEC_DEVICE)) {
        DEBUG_LOG(1,"Downloading vector");
#ifdef GHOST_HAVE_CUDA
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CU_copyDeviceToHost(VECVAL(vec,vec->val,v,0),&vec->CU_val[vec->traits->nrowspadded*v*ghost_sizeofDataType(vec->traits->datatype)],vec->traits->nrowshalo*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
#ifdef GHOST_HAVE_OPENCL
        ghost_vidx_t v;
        for (v=0; v<vec->traits->nvecs; v++) {
            CL_copyDeviceToHost(VECVAL(vec,vec->val,v,0),VECVAL(vec,vec->CL_val_gpu,v,0), vec->traits->nrowshalo*ghost_sizeofDataType(vec->traits->datatype));
        }
#endif
    }
}

static ghost_vec_t * vec_view (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRvecIDX"x%"PRvecIDX" dense matrix with col offset %"PRvecIDX,src->traits->nrows,nc,coffs);
    ghost_vec_t *new;
    ghost_vtraits_t *newTraits = ghost_cloneVtraits(src->traits);
    newTraits->nvecs = nc;

    new = ghost_createVector(src->context,newTraits);
    ghost_vidx_t v;

    for (v=0; v<new->traits->nvecs; v++) {
        new->val[v] = VECVAL(src,src->val,coffs+v,0);
    }

    new->traits->flags |= GHOST_VEC_VIEW;
    return new;
}

static void vec_viewPlain (ghost_vec_t *vec, void *data, ghost_vidx_t nr, ghost_vidx_t nc, ghost_vidx_t roffs, ghost_vidx_t coffs, ghost_vidx_t lda)
{
    DEBUG_LOG(1,"Viewing a %"PRvecIDX"x%"PRvecIDX" dense matrix from plain data with offset %"PRvecIDX"x%"PRvecIDX,nr,nc,roffs,coffs);

    ghost_vidx_t v;

    for (v=0; v<vec->traits->nvecs; v++) {
        vec->val[v] = &((char *)data)[(lda*(coffs+v)+roffs)*ghost_sizeofDataType(vec->traits->datatype)];
    }
    vec->traits->flags |= GHOST_VEC_VIEW;
}

static ghost_vec_t* vec_viewScatteredVec (ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t *coffs)
{
    DEBUG_LOG(1,"Viewing a %"PRvecIDX"x%"PRvecIDX" scattered dense matrix",src->traits->nrows,nc);
    ghost_vec_t *new;
    ghost_vidx_t v;
    ghost_vtraits_t *newTraits = ghost_cloneVtraits(src->traits);
    newTraits->nvecs = nc;

    new = ghost_createVector(src->context,newTraits);

    for (v=0; v<nc; v++) {
        new->val[v] = VECVAL(src,src->val,coffs[v],0);
    }    

    new->traits->flags |= GHOST_VEC_VIEW;
    new->traits->flags |= GHOST_VEC_SCATTERED;
    return new;
}

static void ghost_normalizeVector( ghost_vec_t *vec)
{
    ghost_normalizeVector_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec);
}

static void vec_print(ghost_vec_t *vec)
{
    ghost_vec_print_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec);
    
}

void ghost_vec_malloc(ghost_vec_t *vec)
{

    ghost_vidx_t v;
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
    if (vec->traits->flags & GHOST_VEC_HOST) {
        if (vec->val[0] == NULL) {
            DEBUG_LOG(2,"Allocating host side of vector");
            vec->val[0] = ghost_malloc_align(vec->traits->nvecs*vec->traits->nrowspadded*sizeofdt,GHOST_DATA_ALIGNMENT);
            for (v=1; v<vec->traits->nvecs; v++) {
                vec->val[v] = vec->val[0]+v*vec->traits->nrowspadded*ghost_sizeofDataType(vec->traits->datatype);
            }
        }
    }

    if (vec->traits->flags & GHOST_VEC_DEVICE) {
        DEBUG_LOG(2,"Allocating device side of vector");
#ifdef GHOST_HAVE_CUDA
        if (vec->CU_val == NULL) {
#ifdef GHOST_HAVE_CUDA_PINNEDMEM
            CU_safecall(cudaHostGetDevicePointer((void **)&vec->CU_val,vec->val,0));
#else
            //CU_safecall(cudaMallocPitch(&(void *)vec->CU_val,&vec->traits->nrowspadded,vec->traits->nrowshalo*sizeofdt,vec->traits->nvecs));
            vec->CU_val = CU_allocDeviceMemory(vec->traits->nrowspadded*vec->traits->nvecs*sizeofdt);
#endif
        }
#endif
#ifdef GHOST_HAVE_OPENCL
        if (vec->CL_val_gpu[0] == NULL) {
            vec->CL_val_gpu[0] = CL_allocDeviceMemoryMapped(vec->traits->nvecs*vec->traits->nrowspadded*sizeofdt,vec->val,CL_MEM_READ_WRITE );
            for (v=1; v<vec->traits->nvecs; v++) {
                vec->CL_val_gpu[v] = vec->CL_val_gpu[0]+vec->traits->nrowspadded*ghost_sizeofDataType(vec->traits->datatype);
            }
        }
#endif
    }    
}

void getNrowsFromContext(ghost_vec_t *vec)
{
    DEBUG_LOG(1,"Computing the number of vector rows from the context");

    if (vec->context != NULL) {
        if (vec->traits->nrows == 0) {
            DEBUG_LOG(2,"nrows for vector not given. determining it from the context");
            if (vec->traits->flags & GHOST_VEC_DUMMY) {
                vec->traits->nrows = 0;
            } else if ((vec->context->flags & GHOST_CONTEXT_GLOBAL) || (vec->traits->flags & GHOST_VEC_GLOBAL))
            {
                if (vec->traits->flags & GHOST_VEC_LHS) {
                    vec->traits->nrows = vec->context->gnrows;
                } else if (vec->traits->flags & GHOST_VEC_RHS) {
                    vec->traits->nrows = vec->context->gncols;
                }
            } 
            else 
            {
                vec->traits->nrows = vec->context->communicator->lnrows[ghost_getRank(vec->context->mpicomm)];
            }
        }
        if (vec->traits->nrowshalo == 0) {
            DEBUG_LOG(2,"nrowshalo for vector not given. determining it from the context");
            if (vec->traits->flags & GHOST_VEC_DUMMY) {
                vec->traits->nrowshalo = 0;
            } else if ((vec->context->flags & GHOST_CONTEXT_GLOBAL) || (vec->traits->flags & GHOST_VEC_GLOBAL))
            {
                vec->traits->nrowshalo = vec->traits->nrows;
            } 
            else 
            {
                if (!(vec->traits->flags & GHOST_VEC_GLOBAL) && vec->traits->flags & GHOST_VEC_RHS) {
                    if (vec->context->communicator->halo_elements == -1) {
                        ABORT("You have to make sure to read in the matrix _before_ creating the right hand side vector in a distributed context! This is because we have to know the number of halo elements of the vector.");
                    }
                    vec->traits->nrowshalo = vec->traits->nrows+vec->context->communicator->halo_elements;
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
}


static void vec_fromVec(ghost_vec_t *vec, ghost_vec_t *vec2, ghost_vidx_t coffs)
{
    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from vector w/ col offset %"PRvecIDX,coffs);
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
    ghost_vidx_t i,v;

    for (v=0; v<vec->traits->nvecs; v++) {
#pragma omp parallel for 
        for (i=0; i<vec->traits->nrows; i++) {
            memcpy(VECVAL(vec,vec->val,v,i),VECVAL(vec2,vec2->val,coffs+v,i),sizeofdt);
        }
    }
    vec->upload(vec);
}

static void vec_axpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
    ghost_vidx_t nc = MIN(vec->traits->nvecs,vec2->traits->nvecs);
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
    char *s = (char *)ghost_malloc(nc*sizeofdt);

    ghost_vidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*sizeofdt],scale,sizeofdt);
    }

    ghost_vec_vaxpy_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,s);

    free(s);
}

static void vec_axpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *_b)
{
    ghost_vidx_t nc = MIN(vec->traits->nvecs,vec2->traits->nvecs);
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
    char *s = (char *)ghost_malloc(nc*sizeofdt);
    char *b = (char *)ghost_malloc(nc*sizeofdt);

    ghost_vidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*sizeofdt],scale,sizeofdt);
        memcpy(&b[i*sizeofdt],_b,sizeofdt);
    }
    ghost_vec_vaxpby_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,s,b);

    free(s);
    free(b);
}

static void vec_vaxpy(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale)
{
    ghost_vec_vaxpy_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,scale);
}

static void vec_vaxpby(ghost_vec_t *vec, ghost_vec_t *vec2, void *scale, void *b)
{
    ghost_vec_vaxpby_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,scale,b);
}

static void vec_scale(ghost_vec_t *vec, void *scale)
{
    ghost_vidx_t nc = vec->traits->nvecs;
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
    char *s = (char *)ghost_malloc(nc*sizeofdt);

    ghost_vidx_t i;
    for (i=0; i<nc; i++) {
        memcpy(&s[i*sizeofdt],scale,sizeofdt);
    }
    ghost_vec_vscale_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,s);
}

static void vec_vscale(ghost_vec_t *vec, void *scale)
{
    ghost_vec_vscale_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,scale);
}

static void vec_dotprod(ghost_vec_t *vec, ghost_vec_t *vec2, void *res)
{
    ghost_vec_dotprod_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec,vec2,res);
}

static void vec_entry(ghost_vec_t * vec, ghost_vidx_t r, ghost_vidx_t c, void *val) 
{
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
    if (vec->traits->flags & GHOST_VEC_DEVICE)
    {
#if GHOST_HAVE_CUDA
        CU_copyDeviceToHost(val,&vec->CU_val[(c*vec->traits->nrowspadded+r)*sizeofdt],sizeofdt);
#endif
    }
    else if (vec->traits->flags & GHOST_VEC_HOST)
    {
        memcpy(val,VECVAL(vec,vec->val,c,r),sizeofdt);
    }
}

static void vec_fromRand(ghost_vec_t *vec)
{
    ghost_vec_fromRand_funcs[ghost_dataTypeIdx(vec->traits->datatype)](vec);
}

static void vec_fromScalar(ghost_vec_t *vec, void *val)
{
    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Initializing vector from scalar value with %"PRvecIDX" rows",vec->traits->nrows);
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

    int i,v;
    for (v=0; v<vec->traits->nvecs; v++) {
#pragma omp parallel for schedule(runtime) private(i)
        for (i=0; i<vec->traits->nrows; i++) {
            memcpy(VECVAL(vec,vec->val,v,i),val,sizeofdt);
        }
    }
    vec->upload(vec);
}

static void vec_toFile(ghost_vec_t *vec, char *path)
{
#ifdef GHOST_HAVE_MPI
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

    int32_t endianess = ghost_archIsBigEndian();
    int32_t version = 1;
    int32_t order = GHOST_BINVEC_ORDER_COL_FIRST;
    int32_t datatype = vec->traits->datatype;
    int64_t nrows = (int64_t)vec->context->gnrows;
    int64_t ncols = (int64_t)vec->traits->nvecs;
    MPI_File fileh;
    MPI_Status status;
    MPI_safecall(MPI_File_open(vec->context->mpicomm,path,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&fileh));

    if (ghost_getRank(vec->context->mpicomm) == 0) 
    { // write header AND portion
        MPI_safecall(MPI_File_write(fileh,&endianess,1,MPI_INT,&status));
        MPI_safecall(MPI_File_write(fileh,&version,1,MPI_INT,&status));
        MPI_safecall(MPI_File_write(fileh,&order,1,MPI_INT,&status));
        MPI_safecall(MPI_File_write(fileh,&datatype,1,MPI_INT,&status));
        MPI_safecall(MPI_File_write(fileh,&nrows,1,MPI_LONG_LONG,&status));
        MPI_safecall(MPI_File_write(fileh,&ncols,1,MPI_LONG_LONG,&status));

    }    
    ghost_vidx_t v;
    MPI_Datatype mpidt = ghost_mpi_dataType(vec->traits->datatype);
    MPI_safecall(MPI_File_set_view(fileh,4*sizeof(int32_t)+2*sizeof(int64_t),mpidt,mpidt,"native",MPI_INFO_NULL));
    MPI_Offset fileoffset = vec->context->communicator->lfRow[ghost_getRank(vec->context->mpicomm)];
    ghost_vidx_t vecoffset = 0;
    for (v=0; v<vec->traits->nvecs; v++) {
        MPI_safecall(MPI_File_write_at(fileh,fileoffset,VECVAL(vec,vec->val,v,0),vec->traits->nrows,mpidt,&status));
        fileoffset += nrows;
        vecoffset += vec->traits->nrowspadded*sizeofdt;
    }
    MPI_safecall(MPI_File_close(&fileh));


#else
    DEBUG_LOG(1,"Writing (local) vector to file %s",path);
    size_t ret;
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

    int32_t endianess = ghost_archIsBigEndian();
    int32_t version = 1;
    int32_t order = GHOST_BINVEC_ORDER_COL_FIRST;
    int32_t datatype = vec->traits->datatype;
    int64_t nrows = (int64_t)vec->traits->nrows;
    int64_t ncols = (int64_t)vec->traits->nvecs;

    FILE *filed;

    if ((filed = fopen64(path, "w")) == NULL){
        ABORT("Could not vector file %s",path);
    }

    if ((ret = fwrite(&endianess,sizeof(endianess),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&version,sizeof(version),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&order,sizeof(order),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&datatype,sizeof(datatype),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&nrows,sizeof(nrows),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));
    if ((ret = fwrite(&ncols,sizeof(ncols),1,filed)) != 1) ABORT("fwrite failed (%zu): %s",ret,strerror(errno));

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
            val = ghost_malloc(vec->traits->nrows*sizeofdt);
            copied = 1;
            CU_copyDeviceToHost(val,&vec->CU_val[v*vec->traits->nrowspadded*sizeofdt],vec->traits->nrows*sizeofdt);
        }
        else 
        {
            WARNING_LOG("Invalid vector placement, not writing vector");
            fclose(filed);
        }

        if ((ret = fwrite(val, sizeofdt, vec->traits->nrows,filed)) != vec->traits->nrows)
            ABORT("fwrite failed (%zu): %s",ret,strerror(errno));

        if (copied)
            free(val);
    }
    fclose(filed);
#endif

}

static void vec_fromFile(ghost_vec_t *vec, char *path)
{
    off_t offset;
    if ((vec->context == NULL) || !(vec->context->flags & GHOST_CONTEXT_DISTRIBUTED))
        offset = 0;
    else
        offset = vec->context->communicator->lfRow[ghost_getRank(vec->context->mpicomm)];

    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Reading vector from file %s",path);
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

    FILE *filed;
    size_t ret;

    if ((filed = fopen64(path, "r")) == NULL){
        ABORT("Could not vector file %s",path);
    }

    int32_t endianess;
    int32_t version;
    int32_t order;
    int32_t datatype;

    int64_t nrows;
    int64_t ncols;


    if ((ret = fread(&endianess, sizeof(endianess), 1,filed)) != 1)
        ABORT("fread failed: %zu",ret);

    if (endianess != GHOST_BINCRS_LITTLE_ENDIAN)
        ABORT("Cannot read big endian vectors");

    if ((ret = fread(&version, sizeof(version), 1,filed)) != 1)
        ABORT("fread failed");

    if (version != 1)
        ABORT("Cannot read vector files with format != 1 (is %d)",version);

    if ((ret = fread(&order, sizeof(order), 1,filed)) != 1)
        ABORT("fread failed");
    // Order does not matter for vectors

    if ((ret = fread(&datatype, sizeof(datatype), 1,filed)) != 1)
        ABORT("fread failed");
    if (datatype != vec->traits->datatype)
        ABORT("The data types don't match! Cast-while-read is not yet implemented for vectors.");

    if ((ret = fread(&nrows, sizeof(nrows), 1,filed)) != 1)
        ABORT("fread failed");
    // I will read as many rows as the vector has

    if ((ret = fread(&ncols, sizeof(ncols), 1,filed)) != 1)
        ABORT("fread failed");
    //if (ncols != 1)
    //    ABORT("The number of columns has to be 1!");

    int v;
    for (v=0; v<vec->traits->nvecs; v++) {
        if (fseeko(filed,offset*sizeofdt,SEEK_CUR))
            ABORT("Seek failed");
        if (vec->traits->flags & GHOST_VEC_HOST)
        {
            if ((ret = fread(VECVAL(vec,vec->val,v,0), sizeofdt, vec->traits->nrows,filed)) != vec->traits->nrows)
                ABORT("fread failed");
            vec->upload(vec);
        }
        else if (vec->traits->flags & GHOST_VEC_DEVICE)
        {
            char * val = ghost_malloc(vec->traits->nrows*sizeofdt);
            if ((ret = fread(val, sizeofdt, vec->traits->nrows,filed)) != vec->traits->nrows)
                ABORT("fread failed");
            CU_copyHostToDevice(&vec->CU_val[v*vec->traits->nrowspadded*sizeofdt],val,vec->traits->nrows*sizeofdt);
            free(val);
        }
        else
        {
            WARNING_LOG("Invalid vector placement, not writing vector");
            fclose(filed);
        }

    }

    fclose(filed);


}

static void vec_fromFunc(ghost_vec_t *vec, void (*fp)(int,int,void *))
{
    ghost_vec_malloc(vec);
    DEBUG_LOG(1,"Filling vector via function");

    int i,v;

    for (v=0; v<vec->traits->nvecs; v++) {
#pragma omp parallel for schedule(runtime) private(i)
        for (i=0; i<vec->traits->nrows; i++) {
            fp(i,v,VECVAL(vec,vec->val,v,i));
        }
    }

    vec->upload(vec);
}

static void ghost_zeroVector(ghost_vec_t *vec) 
{
    DEBUG_LOG(1,"Zeroing vector");
    ghost_vidx_t v;

    for (v=0; v<vec->traits->nvecs; v++) {
        memset(VECVAL(vec,vec->val,v,0),0,vec->traits->nrowspadded*ghost_sizeofDataType(vec->traits->datatype));
    }

    vec->upload(vec);
}

static void ghost_distributeVector(ghost_vec_t *vec, ghost_vec_t *nodeVec)
{
    DEBUG_LOG(1,"Distributing vector");
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
    ghost_vidx_t c;
#ifdef GHOST_HAVE_MPI
    int me = ghost_getRank(nodeVec->context->mpicomm);
    DEBUG_LOG(2,"Scattering global vector to local vectors");

    ghost_comm_t *comm = nodeVec->context->communicator;
    MPI_Datatype mpidt = ghost_mpi_dataType(vec->traits->datatype);

    int nprocs = ghost_getNumberOfRanks(nodeVec->context->mpicomm);
    int i;

    MPI_Request req[vec->traits->nvecs*2*(nprocs-1)];
    MPI_Status stat[vec->traits->nvecs*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits->nvecs*2*(nprocs-1);i++) 
        req[i] = MPI_REQUEST_NULL;

    if (ghost_getRank(nodeVec->context->mpicomm) != 0) {
        for (c=0; c<vec->traits->nvecs; c++) {
            MPI_safecall(MPI_Irecv(nodeVec->val[c],comm->lnrows[me],mpidt,0,me,nodeVec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits->nvecs; c++) {
            memcpy(nodeVec->val[c],vec->val[c],sizeofdt*comm->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_safecall(MPI_Isend(VECVAL(vec,vec->val,c,comm->lfRow[i]),comm->lnrows[i],mpidt,i,i,nodeVec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_safecall(MPI_Waitall(msgcount,req,stat));
#else

    for (c=0; c<vec->traits->nvecs; c++) {
        memcpy(nodeVec->val[c],vec->val[c],vec->traits->nrowspadded*sizeofdt);
    }
    //    *nodeVec = vec->clone(vec);
#endif

    nodeVec->upload(nodeVec);

    DEBUG_LOG(1,"Vector distributed successfully");
}

static void ghost_collectVectors(ghost_vec_t *vec, ghost_vec_t *totalVec) 
{
    ghost_vidx_t c;
#ifdef GHOST_HAVE_MPI
    MPI_Datatype mpidt;

    if (vec->traits->datatype & GHOST_BINCRS_DT_COMPLEX) {
        if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
            mpidt = GHOST_MPI_DT_C;
        } else {
            mpidt = GHOST_MPI_DT_Z;
        }
    } else {
        if (vec->traits->datatype & GHOST_BINCRS_DT_FLOAT) {
            mpidt = MPI_FLOAT;
        } else {
            mpidt = MPI_DOUBLE;
        }
    }
    int me = ghost_getRank(vec->context->mpicomm);
    if (vec->context != NULL)
        vec->permute(vec,vec->context->invRowPerm); 

    int nprocs = ghost_getNumberOfRanks(vec->context->mpicomm);
    int i;
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);

    ghost_comm_t *comm = vec->context->communicator;
    MPI_Request req[vec->traits->nvecs*2*(nprocs-1)];
    MPI_Status stat[vec->traits->nvecs*2*(nprocs-1)];
    int msgcount = 0;

    for (i=0;i<vec->traits->nvecs*2*(nprocs-1);i++) 
        req[i] = MPI_REQUEST_NULL;

    if (ghost_getRank(vec->context->mpicomm) != 0) {
        for (c=0; c<vec->traits->nvecs; c++) {
            MPI_safecall(MPI_Isend(vec->val[c],comm->lnrows[me],mpidt,0,me,vec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits->nvecs; c++) {
            memcpy(totalVec->val[c],vec->val[c],sizeofdt*comm->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_safecall(MPI_Irecv(VECVAL(totalVec,totalVec->val,c,comm->lfRow[i]),comm->lnrows[i],mpidt,i,i,vec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_safecall(MPI_Waitall(msgcount,req,stat));
#else
    if (vec->context != NULL) {
        vec->permute(vec,vec->context->invRowPerm);
        for (c=0; c<vec->traits->nvecs; c++) {
            memcpy(totalVec->val[c],vec->val[c],totalVec->traits->nrows*ghost_sizeofDataType(vec->traits->datatype));
        }
    }
#endif
}

static void ghost_swapVectors(ghost_vec_t *v1, ghost_vec_t *v2) 
{
    if ((v1->traits->nvecs > 1) || (v2->traits->nvecs > 1)) {
        WARNING_LOG("Multi-column vector swapping not yet implemented");
    }
    char *dtmp;

    dtmp = v1->val[0];
    v1->val[0] = v2->val[0];
    v2->val[0] = dtmp;
#ifdef GHOST_HAVE_OPENCL
    cl_mem tmp;
    tmp = v1->CL_val_gpu[0];
    v1->CL_val_gpu[0] = v2->CL_val_gpu[0];
    v2->CL_val_gpu[0] = tmp;
#endif
#ifdef GHOST_HAVE_CUDA
    dtmp = v1->CU_val;
    v1->CU_val = v2->CU_val;
    v2->CU_val = dtmp;
#endif

}


static void ghost_freeVector( ghost_vec_t* vec ) 
{
    if( vec ) {
        if (!(vec->traits->flags & GHOST_VEC_VIEW)) {
            ghost_vidx_t v;
#ifdef GHOST_HAVE_CUDA_PINNEDMEM
            if (vec->traits->flags & GHOST_VEC_DEVICE) {
                for (v=0; v<vec->traits->nvecs) { 
                    CU_safecall(cudaFreeHost(vec->val[v]));
                }
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
#ifdef GHOST_HAVE_OPENCL
            if (vec->traits->flags & GHOST_VEC_DEVICE) {
                for (v=0; v<vec->traits->nvecs; v++) { 
                    CL_freeDeviceMemory( vec->CL_val_gpu[v] );
                }
            }
#endif
#ifdef GHOST_HAVE_CUDA
            if (vec->traits->flags & GHOST_VEC_DEVICE) {
                CU_freeDeviceMemory( vec->CU_val );
            }
#endif
        }
        free(vec->val);
        free(vec);
        // TODO free traits ???
    }
}
static void ghost_permuteVector( ghost_vec_t* vec, ghost_vidx_t* perm) 
{
    // TODO enhance performance
    /* permutes values in vector so that i-th entry is mapped to position perm[i] */
    size_t sizeofdt = ghost_sizeofDataType(vec->traits->datatype);
    ghost_midx_t i;
    ghost_vidx_t len = vec->traits->nrows, c;
    char* tmp;

    if (perm == NULL) {
        DEBUG_LOG(1,"Permutation vector is NULL, returning.");
        return;
    } else {
        DEBUG_LOG(1,"Permuting vector");
    }


    for (c=0; c<vec->traits->nvecs; c++) {
        tmp = ghost_malloc(sizeofdt*len);
        for(i = 0; i < len; ++i) {
            if( perm[i] >= len ) {
                ABORT("Permutation index out of bounds: %"PRmatIDX" > %"PRmatIDX,perm[i],len);
            }

            memcpy(&tmp[sizeofdt*perm[i]],VECVAL(vec,vec->val,c,i),sizeofdt);
        }
        for(i=0; i < len; ++i) {
            memcpy(VECVAL(vec,vec->val,c,i),&tmp[sizeofdt*i],sizeofdt);
        }
        free(tmp);
    }
}

static ghost_vec_t * ghost_cloneVector(ghost_vec_t *src, ghost_vidx_t nc, ghost_vidx_t coffs)
{
    ghost_vec_t *new = ghost_createVector(src->context,ghost_cloneVtraits(src->traits));
    new->traits->nvecs = nc;

    // copy the data even if the input vector is itself a view
    // (bitwise NAND operation to unset the view flag if set)
    new->traits->flags &= ~GHOST_VEC_VIEW;

    new->fromVec(new,src,coffs);
    return new;
}
