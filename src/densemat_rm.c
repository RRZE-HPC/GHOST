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
#ifdef GHOST_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#define ROWMAJOR
#include "ghost/densemat_iter_macros.h"
#include "ghost/densemat_common.c.def"

static ghost_error_t vec_rm_fromFunc(ghost_densemat_t *vec, void (*fp)(ghost_gidx_t, ghost_lidx_t, void *));
static ghost_error_t vec_rm_fromFile(ghost_densemat_t *vec, char *path, bool singleFile);
static ghost_error_t vec_rm_toFile(ghost_densemat_t *vec, char *path, bool singleFile);
static ghost_error_t ghost_distributeVector(ghost_densemat_t *vec, ghost_densemat_t *nodeVec);
static ghost_error_t ghost_collectVectors(ghost_densemat_t *vec, ghost_densemat_t *totalVec); 
static ghost_error_t ghost_cloneVector(ghost_densemat_t *src, ghost_densemat_t **new, ghost_lidx_t nr, ghost_lidx_t roffs, ghost_lidx_t nc, ghost_lidx_t coffs);
static ghost_error_t vec_rm_compress(ghost_densemat_t *vec);
static ghost_error_t vec_rm_upload(ghost_densemat_t *vec);
static ghost_error_t vec_rm_download(ghost_densemat_t *vec);
static ghost_error_t vec_rm_uploadHalo(ghost_densemat_t *vec);
static ghost_error_t vec_rm_downloadHalo(ghost_densemat_t *vec);
static ghost_error_t vec_rm_uploadNonHalo(ghost_densemat_t *vec);
static ghost_error_t vec_rm_downloadNonHalo(ghost_densemat_t *vec);
static ghost_error_t vec_rm_equalize(ghost_densemat_t *vec, ghost_mpi_comm_t comm, int root);
static ghost_error_t densemat_rm_halocommInit(ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm);
static ghost_error_t densemat_rm_halocommFinalize(ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm);

ghost_error_t ghost_densemat_rm_setfuncs(ghost_densemat_t *vec)
{
    ghost_error_t ret = GHOST_SUCCESS;

    if (vec->traits.location == GHOST_LOCATION_DEVICE || vec->traits.location == GHOST_LOCATION_HOSTDEVICE)
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
    else if (vec->traits.location == GHOST_LOCATION_HOST)
    {
        vec->dot = &ghost_densemat_rm_dotprod_selector;
        vec->vaxpy = &ghost_densemat_rm_vaxpy_selector;
        vec->vaxpby = &ghost_densemat_rm_vaxpby_selector;
        vec->axpy = &ghost_densemat_rm_axpy;
        vec->axpby = &ghost_densemat_rm_axpby;
        vec->scale = &ghost_densemat_rm_scale;
        vec->vscale = &ghost_densemat_rm_vscale_selector;
        vec->fromScalar = &ghost_densemat_rm_fromScalar_selector;
        vec->fromRand = &ghost_densemat_rm_fromRand_selector;
    }

    vec->compress = &vec_rm_compress;
    vec->string = &ghost_densemat_rm_string_selector;
    vec->fromFunc = &vec_rm_fromFunc;
    vec->fromVec = &ghost_densemat_rm_fromVec_selector;
    vec->fromFile = &vec_rm_fromFile;
    vec->toFile = &vec_rm_toFile;
    vec->distribute = &ghost_distributeVector;
    vec->collect = &ghost_collectVectors;
    vec->normalize = &ghost_densemat_rm_normalize_selector;
    vec->destroy = &ghost_densemat_destroy;
    vec->permute = &ghost_densemat_rm_permute_selector;
    vec->clone = &ghost_cloneVector;
    vec->entry = &ghost_densemat_rm_entry;
    vec->viewVec = &ghost_densemat_rm_view;
    vec->viewPlain = &ghost_densemat_rm_viewPlain;
    vec->viewScatteredVec = &ghost_densemat_rm_viewScatteredVec;
    vec->viewScatteredCols = &ghost_densemat_rm_viewScatteredCols;
    vec->viewCols = &ghost_densemat_rm_viewCols;
    vec->syncValues = &vec_rm_equalize;
    vec->halocommInit = &densemat_rm_halocommInit;
    vec->halocommFinalize = &densemat_rm_halocommFinalize;
    vec->halocommStart = &ghost_densemat_halocommStart_common;
    
    vec->averageHalo = &ghost_densemat_rm_averagehalo_selector;

    vec->upload = &vec_rm_upload;
    vec->download = &vec_rm_download;
    vec->uploadHalo = &vec_rm_uploadHalo;
    vec->downloadHalo = &vec_rm_downloadHalo;
    vec->uploadNonHalo = &vec_rm_uploadNonHalo;
    vec->downloadNonHalo = &vec_rm_downloadNonHalo;
#ifdef GHOST_HAVE_CUDA
    if (vec->traits.location == GHOST_LOCATION_DEVICE) {
        vec->cu_val = NULL;
    }
#endif

    return ret;
}

static ghost_error_t vec_rm_uploadHalo(ghost_densemat_t *vec)
{
    if (!((vec->traits.location == GHOST_LOCATION_HOST) && 
                (vec->traits.location == GHOST_LOCATION_DEVICE))) {
        return GHOST_SUCCESS;
    }
    if (vec->traits.flags & GHOST_DENSEMAT_NO_HALO) {
        ERROR_LOG("Cannot upload halo in NO_HALO densemat!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (DENSEMAT_COMPACT(vec)) {
        if (vec->traits.ncolsorig != vec->traits.ncols) {
            ghost_lidx_t row;
            for (row=vec->traits.nrowsorig; row<vec->traits.nrowshalo; row++) {
                GHOST_CALL_RETURN(ghost_cu_upload(
                            DENSEMAT_CUVALPTR(vec,row,0),
                            DENSEMAT_VALPTR(vec,row,0), 
                            vec->traits.ncols*vec->elSize));
            }
        } else { 
            GHOST_CALL_RETURN(ghost_cu_upload(
                        DENSEMAT_CUVALPTR(vec,vec->traits.nrows,0),
                        DENSEMAT_VALPTR(vec,vec->traits.nrows,0), 
                        (vec->traits.nrowshalo-vec->traits.nrows)*
                        vec->traits.ncolspadded*vec->elSize));
        }
    } else {
        int col, memcol = -1;

        for (col=0; col<vec->traits.ncols; col++) {
            memcol = ghost_bitmap_next(vec->colmask,memcol);
            GHOST_CALL_RETURN(ghost_cu_upload2d(
                        DENSEMAT_CUVALPTR(vec,vec->traits.nrowsorig,memcol),
                        vec->traits.ncolspadded*vec->elSize,
                        DENSEMAT_VALPTR(vec,vec->traits.nrows,memcol),
                        vec->traits.ncolspadded*vec->elSize,vec->elSize,
                        vec->context->halo_elements));
        }
    }
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_downloadHalo(ghost_densemat_t *vec)
{
    if (!((vec->traits.location == GHOST_LOCATION_HOST) && 
                (vec->traits.location == GHOST_LOCATION_DEVICE))) {
        return GHOST_SUCCESS;
    }
    if (vec->traits.flags & GHOST_DENSEMAT_NO_HALO) {
        ERROR_LOG("Cannot download halo in NO_HALO densemat!");
        return GHOST_ERR_INVALID_ARG;
    }
    if (DENSEMAT_COMPACT(vec)) {
        if (vec->traits.ncolsorig != vec->traits.ncols) {
            ghost_lidx_t row;
            for (row=vec->traits.nrowsorig; row<vec->traits.nrowshalo; row++) {
                GHOST_CALL_RETURN(ghost_cu_download(
                            DENSEMAT_VALPTR(vec,row,0), 
                            DENSEMAT_CUVALPTR(vec,row,0),
                            vec->traits.ncols*vec->elSize));
            }
        } else { 
            GHOST_CALL_RETURN(ghost_cu_download(
                        DENSEMAT_VALPTR(vec,vec->traits.nrows,0), 
                        DENSEMAT_CUVALPTR(vec,vec->traits.nrows,0),
                        (vec->traits.nrowshalo-vec->traits.nrows)*
                        vec->traits.ncolspadded*vec->elSize));
        }
    } else {
        int col, memcol = -1;

        for (col=0; col<vec->traits.ncols; col++) {
            memcol = ghost_bitmap_next(vec->colmask,memcol);
            GHOST_CALL_RETURN(ghost_cu_download2d(
                        DENSEMAT_VALPTR(vec,vec->traits.nrows,memcol),
                        vec->traits.ncolspadded*vec->elSize,
                        DENSEMAT_CUVALPTR(vec,vec->traits.nrowsorig,memcol),
                        vec->traits.ncolspadded*vec->elSize,
                        vec->elSize,vec->context->halo_elements));
        }
    }
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_uploadNonHalo(ghost_densemat_t *vec)
{
    if (!((vec->traits.location == GHOST_LOCATION_HOST) && 
                (vec->traits.location == GHOST_LOCATION_DEVICE))) {
        return GHOST_SUCCESS;
    }
    if (DENSEMAT_COMPACT(vec)) {
        if (vec->traits.ncolsorig != vec->traits.ncols) {
            ghost_lidx_t row;
            for (row=0; row<vec->traits.nrows; row++) {
                GHOST_CALL_RETURN(ghost_cu_upload(
                            DENSEMAT_CUVALPTR(vec,row,0),
                            DENSEMAT_VALPTR(vec,row,0), 
                            vec->traits.ncols*vec->elSize));
            }
        } else { 
            GHOST_CALL_RETURN(ghost_cu_upload(
                        DENSEMAT_CUVALPTR(vec,0,0),
                        DENSEMAT_VALPTR(vec,0,0), 
                        vec->traits.nrows*vec->stride*vec->elSize));
        }
    } else {
        DENSEMAT_ITER(vec,ghost_cu_upload(
                    DENSEMAT_CUVALPTR(vec,memrow,memcol),
                    DENSEMAT_VALPTR(vec,row,memcol),
                    vec->elSize));
    }
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_downloadNonHalo(ghost_densemat_t *vec)
{
    if (!((vec->traits.location == GHOST_LOCATION_HOST) && 
                (vec->traits.location == GHOST_LOCATION_DEVICE))) {
        return GHOST_SUCCESS;
    }
    if (DENSEMAT_COMPACT(vec)) {
        if (vec->traits.ncolsorig != vec->traits.ncols) {
            ghost_lidx_t row;
            for (row=0; row<vec->traits.nrows; row++) {
                GHOST_CALL_RETURN(ghost_cu_download(
                            DENSEMAT_VALPTR(vec,row,0), 
                            DENSEMAT_CUVALPTR(vec,row,0),
                            vec->traits.ncols*vec->elSize));
            }
        } else { 
            GHOST_CALL_RETURN(ghost_cu_download(
                        DENSEMAT_VALPTR(vec,0,0), 
                        DENSEMAT_CUVALPTR(vec,0,0),
                        vec->traits.nrows*vec->stride*vec->elSize));
        }
    } else {
        DENSEMAT_ITER(vec,ghost_cu_download(valptr,cuvalptr,vec->elSize));
    }
    
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_upload(ghost_densemat_t *vec) 
{
    GHOST_CALL_RETURN(vec->uploadNonHalo(vec));
    if (!(vec->traits.flags & GHOST_DENSEMAT_NO_HALO)) {
        GHOST_CALL_RETURN(vec->uploadHalo(vec));
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_download(ghost_densemat_t *vec)
{
    GHOST_CALL_RETURN(vec->downloadNonHalo(vec));
    if (!(vec->traits.flags & GHOST_DENSEMAT_NO_HALO)) {
        GHOST_CALL_RETURN(vec->downloadHalo(vec));
    }
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_equalize(ghost_densemat_t *vec, ghost_mpi_comm_t comm, int root)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_mpi_datatype_t vecdt;
    ghost_mpi_datatype(&vecdt,vec->traits.datatype);

    vec->downloadNonHalo(vec);

    if (vec->traits.flags & GHOST_DENSEMAT_SCATTERED) {
        ghost_lidx_t row,col;
        for (row=0; row<vec->traits.nrows; row++) {
            for (col=0; col<vec->traits.ncols; col++) {
                MPI_CALL_RETURN(MPI_Bcast(DENSEMAT_VALPTR(vec,row,col),1,vecdt,root,comm));
            }
        }
    } else if (vec->traits.flags & GHOST_DENSEMAT_VIEW) {
        ghost_lidx_t row;
        for (row=0; row<vec->traits.nrows; row++) {
            MPI_CALL_RETURN(MPI_Bcast(DENSEMAT_VALPTR(vec,row,0),vec->traits.nrows,vecdt,root,comm));
        }
    } else {
        MPI_CALL_RETURN(MPI_Bcast(vec->val,vec->traits.ncolspadded*vec->traits.nrows,vecdt,root,comm));
    }
    
    vec->uploadNonHalo(vec);
     
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
#else
    UNUSED(vec);
    UNUSED(comm);
    UNUSED(root);
#endif
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_toFile(ghost_densemat_t *vec, char *path, bool singleFile)
{ 

#ifndef GHOST_HAVE_MPI
    singleFile = false;
#endif

    if (singleFile && vec->context) {
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
        ghost_mpi_datatype_t mpidt;
        GHOST_CALL_RETURN(ghost_mpi_datatype(&mpidt,vec->traits.datatype));
        MPI_CALL_RETURN(MPI_File_set_view(fileh,4*sizeof(int32_t)+2*sizeof(int64_t),mpidt,mpidt,"native",MPI_INFO_NULL));
        MPI_Offset fileoffset = vec->context->lfRow[rank]*vec->traits.ncols;
        
        GHOST_SINGLETHREAD(DENSEMAT_ITER(vec,MPI_File_write_at(fileh,fileoffset++,valptr,1,mpidt,&status)));
        /*
        ghost_lidx_t vecoffset = 0;
         for (v=0; v<vec->traits.nrows; v++) {
            char *val = NULL;
            int copied = 0;
            if (vec->traits.location == GHOST_LOCATION_HOST)
            {
                vec->download(vec);
                val = DENSEMAT_VALPTR(vec,v,0);
            }
            else if (vec->traits.location == GHOST_LOCATION_DEVICE)
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
        }*/
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

        GHOST_SINGLETHREAD(DENSEMAT_ITER(vec,fwrite(valptr, vec->elSize, 1, filed)));
        /*ghost_lidx_t v;
        for (v=0; v<vec->traits.nrows; v++) {
            char *val = NULL;
            int copied = 0;
            if (vec->traits.location == GHOST_LOCATION_HOST)
            {
                vec->download(vec);
                val = DENSEMAT_VALPTR(vec,v,0);
            }
            else if (vec->traits.location == GHOST_LOCATION_DEVICE)
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
        }*/
        fclose(filed);
    }

    return GHOST_SUCCESS;

}

static ghost_error_t vec_rm_fromFile(ghost_densemat_t *vec, char *path, bool singleFile)
{
    
#ifndef GHOST_HAVE_MPI
    singleFile = false;
#endif


    off_t offset;
    if ((vec->context == NULL) || !singleFile) {
        offset = 0;
    } else {
        int rank;
        GHOST_CALL_RETURN(ghost_rank(&rank, vec->context->mpicomm));
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
    if (singleFile && vec->context && (vec->context->gnrows != nrows)) {
        ERROR_LOG("The number of rows does not match between the file and the densemat's context!");
        return GHOST_ERR_IO;
    }

    if (fseeko(filed,offset*vec->elSize,SEEK_CUR)) {
        ERROR_LOG("seek failed");
        vec->destroy(vec);
        return GHOST_ERR_IO;
    }

    GHOST_SINGLETHREAD(DENSEMAT_ITER(vec,fread(valptr, vec->elSize, 1,filed)));
    /*int v;
    for (v=0; v<vec->traits.nrows; v++) {
        if (vec->traits.location == GHOST_LOCATION_HOST)
        {
            if ((ghost_lidx_t)(ret = fread(DENSEMAT_VALPTR(vec,v,0), vec->elSize, vec->traits.ncols,filed)) != vec->traits.ncols) {
                ERROR_LOG("fread failed: %zu",ret);
                vec->destroy(vec);
                return GHOST_ERR_IO;
            }
            vec->upload(vec);
        }
        else if (vec->traits.location == GHOST_LOCATION_DEVICE)
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

    }*/

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


    if (vec->traits.location == GHOST_LOCATION_HOST || vec->traits.location == GHOST_LOCATION_HOSTDEVICE) { // vector is stored on host
        DENSEMAT_ITER(vec,fp(offset+row,col,valptr));
        vec->uploadNonHalo(vec);
    } else {
        INFO_LOG("Need to create dummy HOST densemat!");
        ghost_densemat_t *hostVec;
        ghost_densemat_traits_t htraits = vec->traits;
        htraits.location = GHOST_LOCATION_HOST;
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
            MPI_CALL_RETURN(MPI_Irecv(DENSEMAT_VALPTR(nodeVec,0,c),nodeVec->context->lnrows[me],mpidt,0,me,nodeVec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(nodeVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->elSize*nodeVec->context->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VALPTR(vec,c,nodeVec->context->lfRow[i]),nodeVec->context->lnrows[i],mpidt,i,i,nodeVec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else

    for (c=0; c<vec->traits.ncols; c++) {
        memcpy(DENSEMAT_VALPTR(nodeVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->traits.nrows*vec->elSize);
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
            MPI_CALL_RETURN(MPI_Isend(DENSEMAT_VALPTR(vec,0,c),vec->context->lnrows[me],mpidt,0,me,vec->context->mpicomm,&req[msgcount]));
            msgcount++;
        }
    } else {
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(totalVec,0,c),DENSEMAT_VALPTR(vec,0,c),vec->elSize*vec->context->lnrows[0]);
            for (i=1;i<nprocs;i++) {
                MPI_CALL_RETURN(MPI_Irecv(DENSEMAT_VALPTR(totalVec,c,vec->context->lfRow[i]),vec->context->lnrows[i],mpidt,i,i,vec->context->mpicomm,&req[msgcount]));
                msgcount++;
            }
        }
    }
    MPI_CALL_RETURN(MPI_Waitall(msgcount,req,stat));
#else
    if (vec->context != NULL) {
//        vec->permute(vec,vec->context->invRowPerm);
        for (c=0; c<vec->traits.ncols; c++) {
            memcpy(DENSEMAT_VALPTR(totalVec,0,c),DENSEMAT_VALPTR(vec,0,c),totalVec->traits.nrows*vec->elSize);
        }
    }
#endif

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
    (*new)->traits.flags &= ~(ghost_densemat_flags_t)GHOST_DENSEMAT_VIEW;
    (*new)->traits.flags &= ~(ghost_densemat_flags_t)GHOST_DENSEMAT_SCATTERED;
   
    (*new)->fromVec(*new,src,roffs,coffs);
    return GHOST_SUCCESS;
}

static ghost_error_t vec_rm_compress(ghost_densemat_t *vec)
{
    if (!(vec->traits.flags & GHOST_DENSEMAT_SCATTERED)) {
        return GHOST_SUCCESS;
    }

    if (vec->traits.location == GHOST_LOCATION_HOST) {
        ghost_lidx_t v,i;

        char *val = NULL;
        if (vec->traits.location == GHOST_LOCATION_DEVICE) {
            GHOST_CALL_RETURN(ghost_malloc_pinned((void **)&val,
                        (size_t)vec->traits.ncolspadded*vec->traits.nrowspadded*
                        vec->elSize));
        } else {
            GHOST_CALL_RETURN(ghost_malloc_align((void **)&val,
                        (size_t)vec->traits.ncolspadded*vec->traits.nrowspadded*
                        vec->elSize,GHOST_DATA_ALIGNMENT));
        }

#pragma omp parallel for schedule(runtime) private(v)
        for (i=0; i<vec->traits.nrowspadded; i++) {
            for (v=0; v<vec->traits.ncolspadded; v++) {
                val[(v*vec->traits.nrowspadded+i)*vec->elSize] = 0;
            }
        }
        
        DENSEMAT_ITER(vec,memcpy(&val[(row*vec->traits.ncolspadded+col)*vec->elSize],
                    valptr,vec->elSize));
        
        vec->val = val;
      
    }
    if (vec->traits.location == GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA

        char *cu_val;
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&cu_val,vec->traits.nrowspadded*vec->traits.ncolspadded*vec->elSize));
        
        DENSEMAT_ITER(vec,ghost_cu_memcpy(&cu_val[(row*vec->traits.ncolspadded+col)*vec->elSize],
                    DENSEMAT_CUVALPTR(vec,memrow,memcol),vec->elSize));

        if (!(vec->traits.flags & GHOST_DENSEMAT_VIEW)) {
            ghost_cu_free(vec->cu_val);
        }
        vec->cu_val = cu_val;
#endif 
    }
    ghost_bitmap_set_range(vec->colmask,0,vec->traits.ncols-1);
    ghost_bitmap_set_range(vec->rowmask,0,vec->traits.nrows-1);
    vec->traits.ncolsorig = vec->traits.ncols;
    vec->traits.nrowsorig = vec->traits.nrows;
    vec->traits.flags &= ~(ghost_densemat_flags_t)GHOST_DENSEMAT_VIEW;
    vec->traits.flags &= ~(ghost_densemat_flags_t)GHOST_DENSEMAT_SCATTERED;
    vec->stride = vec->traits.ncolspadded;

    return GHOST_SUCCESS;
}

static ghost_error_t densemat_rm_halocommInit(ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;
    int i, to_PE, from_PE;
    int nprocs;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_densemat_halocommInit_common(vec,comm),err,ret);
        
    GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv,nprocs*sizeof(char *)),err,ret);

    if ((vec->stride != vec->traits.ncols) || (vec->traits.location == GHOST_LOCATION_DEVICE)) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&comm->tmprecv_mem,vec->traits.ncols*vec->elSize*comm->acc_wishes),err,ret);

        for (from_PE=0; from_PE<nprocs; from_PE++){
            comm->tmprecv[from_PE] = &comm->tmprecv_mem[comm->wishptr[from_PE]*vec->traits.ncols*vec->elSize];
        }
    } else {
        for (from_PE=0; from_PE<nprocs; from_PE++) {
            comm->tmprecv[from_PE] = DENSEMAT_VALPTR(vec,vec->context->hput_pos[from_PE],0);
        }
        comm->tmprecv_mem = NULL;
    }
        
    if (vec->context->perm_local) {
#ifdef GHOST_HAVE_CUDA
        if (vec->traits.location == GHOST_LOCATION_DEVICE) {
            ghost_densemat_rm_cu_communicationassembly(comm->cu_work,comm->dueptr,vec,vec->context->perm_local->cu_perm);
        } else 
#endif
            if (vec->traits.location == GHOST_LOCATION_HOST) {
#pragma omp parallel private(to_PE,i)
                for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                    for (i=0; i<vec->context->dues[to_PE]; i++){
                        memcpy(comm->work + (comm->dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols,DENSEMAT_VALPTR(vec,vec->context->perm_local->perm[vec->context->duelist[to_PE][i]],0),vec->elSize*vec->traits.ncols);
                    }
                }
            }
    } else {
#ifdef GHOST_HAVE_CUDA
        if (vec->traits.location == GHOST_LOCATION_DEVICE) {
            ghost_densemat_rm_cu_communicationassembly(comm->cu_work,comm->dueptr,vec,NULL);
        } else 
#endif
            if (vec->traits.location == GHOST_LOCATION_HOST) {
#pragma omp parallel private(to_PE,i)
                for (to_PE=0 ; to_PE<nprocs ; to_PE++){
#pragma omp for 
                    for (i=0; i<vec->context->dues[to_PE]; i++){
                        memcpy(comm->work + (comm->dueptr[to_PE]+i)*vec->elSize*vec->traits.ncols,DENSEMAT_VALPTR(vec,vec->context->duelist[to_PE][i],0),vec->elSize*vec->traits.ncols);
                    }
                }
            }
    }
#ifdef GHOST_HAVE_CUDA
    if (vec->traits.location == GHOST_LOCATION_DEVICE) {
        GHOST_INSTR_START("downloadcomm->work");
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
        ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_IN,GHOST_DATATRANSFER_RANK_GPU,vec->traits.ncols*comm->acc_dues*vec->elSize);

#endif
        INFO_LOG("download work");
        ghost_cu_download(comm->work,comm->cu_work,vec->traits.ncols*comm->acc_dues*vec->elSize);
        GHOST_INSTR_STOP("downloadcomm->work");
    }
#endif
    

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
#else
    UNUSED(vec);
    UNUSED(comm);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif


}

static ghost_error_t densemat_rm_halocommFinalize(ghost_densemat_t *vec, ghost_densemat_halo_comm_t *comm)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;
    
    int nprocs;
    int i, from_PE;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, vec->context->mpicomm),err,ret);

    GHOST_CALL_GOTO(ghost_densemat_halocommFinalize_common(comm),err,ret);
    if ((nprocs > 1) && (vec->stride != vec->traits.ncols) && (vec->traits.location == GHOST_LOCATION_DEVICE)) {
        ERROR_LOG("Assemble row-major view not yet implemented for device densemats!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }
    if ((vec->stride != vec->traits.ncols) && (vec->traits.location == GHOST_LOCATION_HOST)) {
        GHOST_INSTR_START("Assemble row-major view");
        for (from_PE=0; from_PE<nprocs; from_PE++){
            for (i=0; i<vec->context->wishes[from_PE]; i++){
                memcpy(DENSEMAT_VALPTR(vec,vec->context->hput_pos[from_PE]+i,0),&comm->tmprecv[from_PE][(i*vec->traits.ncols)*vec->elSize],vec->elSize*vec->traits.ncols);
            }
        }
        GHOST_INSTR_STOP("Assemble row-major view");
    }

#ifdef GHOST_HAVE_CUDA 
    GHOST_INSTR_START("upload")
    if (vec->traits.location == GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_TRACK_DATATRANSFERS
        ghost_datatransfer_register("spmv_halo",GHOST_DATATRANSFER_OUT,GHOST_DATATRANSFER_RANK_GPU,vec->context->halo_elements*vec->traits.ncols*vec->elSize);
#endif
        ghost_cu_upload(DENSEMAT_CUVALPTR(vec,vec->traits.nrows,0),comm->tmprecv_mem,vec->traits.ncols*comm->acc_wishes*vec->elSize);
        INFO_LOG("upload halo");
    }
    GHOST_INSTR_STOP("upload");
#endif


    if (vec->traits.location == GHOST_LOCATION_DEVICE) {
#ifdef GHOST_HAVE_CUDA
        ghost_cu_free(comm->cu_work);
        cudaFreeHost(comm->work); comm->work = NULL;
#endif
    } else {
        free(comm->work); comm->work = NULL;
    }
    free(comm->tmprecv_mem); comm->tmprecv_mem = NULL;
    free(comm->tmprecv); comm->tmprecv = NULL;
    free(comm->request); comm->request = NULL;
    free(comm->status); comm->status = NULL;
    free(comm->dueptr); comm->dueptr = NULL;
    free(comm->wishptr); comm->wishptr = NULL;
    
    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;

#else
    UNUSED(vec);
    UNUSED(comm);
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif





}
