#define _XOPEN_SOURCE 500
#include <stdlib.h>
#include <stdio.h>
#include "ghost/util.h"
#include "ghost/bincrs.h"
#include "ghost/machine.h"
#include "ghost/locality.h"

static inline uint32_t bswap_32(uint32_t val)
{
    return ((val & (uint32_t)0x000000ffUL) << 24)
        | ((val & (uint32_t)0x0000ff00UL) <<  8)
        | ((val & (uint32_t)0x00ff0000UL) >>  8)
        | ((val & (uint32_t)0xff000000UL) >> 24);
}
static inline uint64_t bswap_64(uint64_t val)
{
    return ((val & (uint64_t)0x00000000000000ffULL) << 56)
        | ((val & (uint64_t)0x000000000000ff00ULL) << 40)
        | ((val & (uint64_t)0x0000000000ff0000ULL) << 24)
        | ((val & (uint64_t)0x00000000ff000000ULL) <<  8)
        | ((val & (uint64_t)0x000000ff00000000ULL) >>  8)
        | ((val & (uint64_t)0x0000ff0000000000ULL) >> 24)
        | ((val & (uint64_t)0x00ff000000000000ULL) >> 40)
        | ((val & (uint64_t)0xff00000000000000ULL) >> 56);
}

#define SWAPREQ(header) (header.endianess == GHOST_BINCRS_LITTLE_ENDIAN)?ghost_machine_bigendian()?1:0:ghost_machine_bigendian()?0:1


int ghost_sparsemat_rowfunc_bincrs(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    UNUSED(arg);

    static ghost_gidx *colInd = NULL, *globalRowPtr = NULL, *rowPtr = NULL;
    static char *values = NULL;
    static size_t dtsize = 0;
    static ghost_gidx firstrow = 0;
    static ghost_lidx nrows = 0;

    if (row == GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETDIM) {
        ghost_bincrs_header_t header;
        ghost_sparsemat_rowfunc_bincrs_initargs args = 
            *(ghost_sparsemat_rowfunc_bincrs_initargs *)val;
        
        char *filename = args.filename;

        ghost_bincrs_header_read(&header,filename);

        col[0] = header.nrows;
        col[1] = header.ncols;

        if(rowlen) *rowlen = header.datatype;
    } else if ((row == GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETRPT) || (row == GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_INIT)) {

        ghost_sparsemat_rowfunc_bincrs_initargs args = 
            *(ghost_sparsemat_rowfunc_bincrs_initargs *)val;
        char *filename = args.filename;
        ghost_datatype matdt = args.dt;
        ghost_datatype_size(&dtsize,matdt);
        ghost_bincrs_header_t header;
        ghost_bincrs_header_read(&header,filename);

        FILE *f;
        ghost_gidx i;
        size_t ret;

        if ((f = fopen64(filename,"r")) == NULL) {
            ERROR_LOG("fopen with %s failed!",filename);
            return 1;
        }
        

        if ((ghost_datatype)(header.datatype) != matdt) { 
            ERROR_LOG("Value casting not implemented! Adjust your sparsemat datatype to match the file!");
            return 1;
        }
        if (header.symmetry != GHOST_BINCRS_SYMM_GENERAL) {
            ERROR_LOG("Only general matrices supported at the moment!");
            return 1;
        }

        if (fseeko(f,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
            ERROR_LOG("Seek failed");
            return GHOST_ERR_IO;
        }
        
        if (row == GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETRPT) {
            ghost_malloc((void **)&globalRowPtr,(header.nrows + 1) * sizeof(ghost_gidx));

#pragma omp parallel for
            for(i=0; i < header.nrows+1; ++i){
                globalRowPtr[i] = 0;
            }
            if ((ret = fread(globalRowPtr, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),f)) != (size_t)(header.nrows+1)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
            col = globalRowPtr;
        } else {
            int me;
            ghost_sparsemat *mat = (ghost_sparsemat *)arg;
            ghost_rank(&me,mat->context->mpicomm);
            firstrow = mat->context->row_map->goffs[me];
            nrows = mat->context->row_map->ldim[me];
        
            ghost_malloc((void **)&rowPtr,(nrows + 1) * sizeof(ghost_gidx));
            
            if (fseeko(f,firstrow*GHOST_BINCRS_SIZE_RPT_EL,SEEK_CUR)) {
                ERROR_LOG("Seek failed");
                return GHOST_ERR_IO;
            }
            if ((ret = fread(rowPtr, GHOST_BINCRS_SIZE_RPT_EL, nrows+1,f)) != (size_t)(nrows+1)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
            
            ghost_lidx nnz = (ghost_lidx)(rowPtr[nrows]-rowPtr[0]);
            ghost_malloc((void **)&colInd,nnz * sizeof(ghost_gidx));
            ghost_malloc((void **)&values,nnz * dtsize);

#pragma omp parallel for
            for(i=0; i < nrows; ++i){
                values[rowPtr[i]-rowPtr[0]] = 0;
                colInd[rowPtr[i]-rowPtr[0]] = 0;
            }
            
            
            if (fseeko(f,GHOST_BINCRS_SIZE_HEADER+(header.nrows+1)*GHOST_BINCRS_SIZE_RPT_EL+rowPtr[0]*GHOST_BINCRS_SIZE_COL_EL,SEEK_SET)) {
                ERROR_LOG("Seek failed");
                return GHOST_ERR_IO;
            }
            
            if ((ret = fread(colInd, GHOST_BINCRS_SIZE_COL_EL, nnz,f)) != (size_t)(nnz)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
            
            if (fseeko(f,GHOST_BINCRS_SIZE_HEADER+(header.nrows+1)*GHOST_BINCRS_SIZE_RPT_EL+header.nnz*GHOST_BINCRS_SIZE_COL_EL+rowPtr[0]*dtsize,SEEK_SET)) {
                ERROR_LOG("Seek failed");
                return GHOST_ERR_IO;
            }
            
            if ((ret = fread(values, dtsize, nnz,f)) != (size_t)(nnz)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
        }

        fclose(f);

    } else if (row == GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_FINALIZE) {
        free(colInd);
        free(rowPtr);
        free(globalRowPtr);
        free(values);
    } else {
        *rowlen = rowPtr[row-firstrow+1]-rowPtr[row-firstrow];
        memcpy(col,&colInd[rowPtr[row-firstrow]-rowPtr[0]],(*rowlen)*sizeof(ghost_gidx));
        memcpy(val,&values[(rowPtr[row-firstrow]-rowPtr[0])*dtsize],(*rowlen)*dtsize);
    }


    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    return 0;


}

ghost_error ghost_bincrs_header_read(ghost_bincrs_header_t *header, char *matrixPath)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_IO);
    FILE* file;
    long filesize;
    int swapReq = 0;

    DEBUG_LOG(1,"Reading header from %s",matrixPath);

    if ((file = fopen(matrixPath, "rb"))==NULL){
        ERROR_LOG("Could not open binary CRS file %s: %s",matrixPath,strerror(errno));
        return GHOST_ERR_IO;
    }

    fseek(file,0L,SEEK_END);
    filesize = ftell(file);
    fseek(file,0L,SEEK_SET);

    fread(&header->endianess, 4, 1, file);
    if (header->endianess == GHOST_BINCRS_LITTLE_ENDIAN && ghost_machine_bigendian()) {
        DEBUG_LOG(1,"Need to convert from little to big endian.");
        swapReq = 1;
    } else if (header->endianess != GHOST_BINCRS_LITTLE_ENDIAN && !ghost_machine_bigendian()) {
        DEBUG_LOG(1,"Need to convert from big to little endian.");
        swapReq = 1;
    } else {
        DEBUG_LOG(1,"OK, file and library have same endianess.");
    }

    fread(&header->version, 4, 1, file);
    if (swapReq) header->version = bswap_32(header->version);

    fread(&header->base, 4, 1, file);
    if (swapReq) header->base = bswap_32(header->base);

    fread(&header->symmetry, 4, 1, file);
    if (swapReq) header->symmetry = bswap_32(header->symmetry);

    fread(&header->datatype, 4, 1, file);
    if (swapReq) header->datatype = bswap_32(header->datatype);

    fread(&header->nrows, 8, 1, file);
    if (swapReq) header->nrows  = bswap_64(header->nrows);

    fread(&header->ncols, 8, 1, file);
    if (swapReq)  header->ncols  = bswap_64(header->ncols);

    fread(&header->nnz, 8, 1, file);
    if (swapReq)  header->nnz  = bswap_64(header->nnz);

    size_t valSize;
    GHOST_CALL_RETURN(ghost_datatype_size(&valSize,(ghost_datatype)header->datatype));

    long rightFilesize = GHOST_BINCRS_SIZE_HEADER +
        (long)(header->nrows+1) * GHOST_BINCRS_SIZE_RPT_EL +
        (long)header->nnz * GHOST_BINCRS_SIZE_COL_EL +
        (long)header->nnz * valSize;

    if (filesize != rightFilesize) {
        ERROR_LOG("File has invalid size! (is: %ld, should be: %ld)",filesize, rightFilesize);
        return GHOST_ERR_IO;
    }

    fclose(file);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL|GHOST_FUNCTYPE_IO);
    return GHOST_SUCCESS;
}

