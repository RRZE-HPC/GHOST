#define _XOPEN_SOURCE 500
#include "ghost/io.h"
#include "ghost/log.h"
#include "ghost/util.h"
#include "ghost/constants.h"
#include "ghost/machine.h"
#include <errno.h>
#include <limits.h>

#if GHOST_HAVE_BSWAP
#include <byteswap.h>
#else
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
#endif

void (*ghost_castArray_funcs[4][4]) (void *, void *, int) = 
{{&ss_ghost_castArray,&sd_ghost_castArray,&sc_ghost_castArray,&sz_ghost_castArray},
    {&ds_ghost_castArray,&dd_ghost_castArray,&dc_ghost_castArray,&dz_ghost_castArray},
    {&cs_ghost_castArray,&cd_ghost_castArray,&cc_ghost_castArray,&cz_ghost_castArray},
    {&zs_ghost_castArray,&zd_ghost_castArray,&zc_ghost_castArray,&zz_ghost_castArray}};


ghost_error_t ghost_readColOpen(ghost_midx_t *col, char *matrixPath, ghost_mnnz_t offsEnts, ghost_mnnz_t nEnts, FILE *filed)
{
    ghost_matfile_header_t header;
    size_t ret;
    int swapReq;
    off64_t offs;
    ghost_mnnz_t i;

    GHOST_CALL_RETURN(ghost_readMatFileHeader(matrixPath,&header));
    GHOST_CALL_RETURN(ghost_endianessDiffers(&swapReq,matrixPath));

    DEBUG_LOG(1,"Reading array with column indices");
    offs = GHOST_BINCRS_SIZE_HEADER+
        GHOST_BINCRS_SIZE_RPT_EL*(header.nrows+1)+
        GHOST_BINCRS_SIZE_COL_EL*offsEnts;
    if (fseeko(filed,offs,SEEK_SET)) {
        ERROR_LOG("Seek failed");
        return GHOST_ERR_IO;
    }

#if GHOST_HAVE_LONGIDX
    if (swapReq) {
        int64_t *tmp = (int64_t *)ghost_malloc(nEnts*8);
        if ((ret = fread(tmp, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (nEnts)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
        for( i = 0; i < nEnts; i++ ) {
            col[i] = bswap_64(tmp[i]);
        }
    } else {
        if ((ret = fread(col, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (nEnts)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
    }
#else // casting from 64 to 32 bit
    DEBUG_LOG(1,"Casting from 64 bit to 32 bit column indices");
    int64_t *tmp = (int64_t *)ghost_malloc(nEnts*8);
    if ((ghost_midx_t)(ret = fread(tmp, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (nEnts)){
        ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
        return GHOST_ERR_IO;
    }
    for(i = 0 ; i < nEnts; ++i) {
        if (tmp[i] >= (int64_t)INT_MAX) {
            ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
            return GHOST_ERR_IO;
        }
        if (swapReq) {
            col[i] = (ghost_midx_t)(bswap_64(tmp[i]));
        } else {
            col[i] = (ghost_midx_t)tmp[i];
        }
    }
    free(tmp);
#endif

    return GHOST_SUCCESS;
}


ghost_error_t ghost_readCol(ghost_midx_t *col, char *matrixPath, ghost_mnnz_t offsEnts, ghost_mnnz_t nEnts)
{
    FILE *filed;

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s",matrixPath);
        return GHOST_ERR_IO;
    }

    GHOST_CALL_RETURN(ghost_readColOpen(col,matrixPath,offsEnts,nEnts,filed));

    fclose(filed);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_readValOpen(char *val, int datatype, char *matrixPath, ghost_mnnz_t offsEnts, ghost_mnnz_t nEnts, FILE *filed)
{
    ghost_matfile_header_t header;
    size_t ret;
    int swapReq;
    off64_t offs;
    ghost_mnnz_t i;
    size_t sizeofdt;
    GHOST_CALL_RETURN(ghost_sizeofDatatype(&sizeofdt,datatype));
    GHOST_CALL_RETURN(ghost_readMatFileHeader(matrixPath,&header));
    GHOST_CALL_RETURN(ghost_endianessDiffers(&swapReq,matrixPath));

    size_t valSize;
    GHOST_CALL_RETURN(ghost_sizeofDatatype(&valSize,header.datatype));

    DEBUG_LOG(1,"Reading array with values");
    offs = GHOST_BINCRS_SIZE_HEADER+
        GHOST_BINCRS_SIZE_RPT_EL*(header.nrows+1)+
        GHOST_BINCRS_SIZE_COL_EL*header.nnz+
        valSize*offsEnts;
    if (fseeko(filed,offs,SEEK_SET)) {
        ERROR_LOG("Seek failed");
        return GHOST_ERR_IO;
    }

    if (datatype == header.datatype) {
        if (swapReq) {
            uint8_t *tmpval = (uint8_t *)ghost_malloc(nEnts*valSize);
            if ((ghost_midx_t)(ret = fread(tmpval, valSize, nEnts,filed)) != (nEnts)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
            if (datatype & GHOST_BINCRS_DT_COMPLEX) {
                if (datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t *a = (uint32_t *)tmpval;
                        uint32_t rswapped = bswap_32(a[2*i]);
                        uint32_t iswapped = bswap_32(a[2*i+1]);
                        memcpy(&(val[i]),&rswapped,4);
                        memcpy(&(val[i])+4,&iswapped,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint64_t *a = (uint64_t *)tmpval;
                        uint64_t rswapped = bswap_64(a[2*i]);
                        uint64_t iswapped = bswap_64(a[2*i+1]);
                        memcpy(&(val[i]),&rswapped,8);
                        memcpy(&(val[i])+8,&iswapped,8);
                    }
                }
            } else {
                if (datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t *a = (uint32_t *)tmpval;
                        uint32_t swapped = bswap_32(a[i]);
                        memcpy(&(val[i]),&swapped,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint64_t *a = (uint64_t *)tmpval;
                        uint64_t swapped = bswap_64(a[i]);
                        memcpy(&(val[i]),&swapped,8);
                    }
                }

            }
        } else {
            if ((ghost_midx_t)(ret = fread(val, valSize, nEnts,filed)) != (nEnts)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
        }
    } else {
        INFO_LOG("This matrix is supposed to be of %s data but"
                " the file contains %s data. Casting...",ghost_datatypeString(datatype),ghost_datatypeString(header.datatype));


        uint8_t *tmpval = (uint8_t *)ghost_malloc(nEnts*valSize);
        if ((ghost_midx_t)(ret = fread(tmpval, valSize, nEnts,filed)) != (nEnts)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }

        if (swapReq) {
            WARNING_LOG("Not yet supported!");
            if (datatype & GHOST_BINCRS_DT_COMPLEX) {
                if (datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t re = bswap_32(tmpval[i*valSize]);
                        uint32_t im = bswap_32(tmpval[i*valSize+valSize/2]);
                        memcpy(&val[i*sizeofdt],&re,4);
                        memcpy(&val[i*sizeofdt+4],&im,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t re = bswap_64(tmpval[i*valSize]);
                        uint32_t im = bswap_64(tmpval[i*valSize+valSize/2]);
                        memcpy(&val[i*sizeofdt],&re,8);
                        memcpy(&val[i*sizeofdt+8],&im,8);
                    }
                }
            } else {
                if (datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t swappedVal = bswap_32(tmpval[i*valSize]);
                        memcpy(&val[i*sizeofdt],&swappedVal,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t swappedVal = bswap_64(tmpval[i*valSize]);
                        memcpy(&val[i*sizeofdt],&swappedVal,8);
                    }
                }

            }

        } else {
            ghost_castArray_funcs[ghost_dataTypeIdx(datatype)][ghost_dataTypeIdx(header.datatype)](val,tmpval,nEnts);
        }

        free(tmpval);
    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_readVal(char *val, int datatype, char *matrixPath, ghost_mnnz_t offsEnts, ghost_mnnz_t nEnts)
{
    FILE *filed;

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s",matrixPath);
        return GHOST_ERR_IO;
    }

    ghost_readValOpen(val,datatype,matrixPath,offsEnts,nEnts,filed);

    fclose(filed);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_readRptOpen(ghost_midx_t *rpt, char *matrixPath, ghost_mnnz_t offsRows, ghost_mnnz_t nRows, FILE *filed)
{
    ghost_matfile_header_t header;
    size_t ret;
    int swapReq;
    off64_t offs;
    ghost_mnnz_t i;

    GHOST_CALL_RETURN(ghost_readMatFileHeader(matrixPath,&header));
    GHOST_CALL_RETURN(ghost_endianessDiffers(&swapReq,matrixPath));

    DEBUG_LOG(1,"Reading array with column indices");
    offs = GHOST_BINCRS_SIZE_HEADER+
        GHOST_BINCRS_SIZE_RPT_EL*offsRows;
    if (fseeko(filed,offs,SEEK_SET)) {
        ERROR_LOG("Seek failed");
        return GHOST_ERR_IO;
    }

#if GHOST_HAVE_LONGIDX
    if (swapReq) {
        int64_t *tmp = (int64_t *)ghost_malloc(nRows*8);
        if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, nRows,filed)) != (nRows)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
        for( i = 0; i < nRows; i++ ) {
            rpt[i] = bswap_64(tmp[i]);
        }
    } else {
        if ((ret = fread(rpt, GHOST_BINCRS_SIZE_RPT_EL, nRows,filed)) != (nRows)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
    }
#else // casting from 64 to 32 bit
    DEBUG_LOG(1,"Casting from 64 bit to 32 bit column indices");
    int64_t *tmp = (int64_t *)ghost_malloc(nRows*8);
    if ((ghost_mnnz_t)(ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, nRows,filed)) != (nRows)){
        ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
        return GHOST_ERR_IO;
    }
    for(i = 0 ; i < nRows; ++i) {
        if (tmp[i] >= (int64_t)INT_MAX) {
            ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
            return GHOST_ERR_IO;
        }
        if (swapReq) {
            rpt[i] = (ghost_midx_t)(bswap_64(tmp[i]));
        } else {
            rpt[i] = (ghost_midx_t)tmp[i];
        }
    }
    free(tmp);
#endif

    return GHOST_SUCCESS;
}

ghost_error_t ghost_readRpt(ghost_mnnz_t *rpt, char *matrixPath, ghost_mnnz_t offsRows, ghost_mnnz_t nRows)
{
    FILE *filed;

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s",matrixPath);
        return GHOST_ERR_IO;
    }

    GHOST_CALL_RETURN(ghost_readRptOpen(rpt,matrixPath,offsRows,nRows,filed));

    fclose(filed);

    return GHOST_SUCCESS;

}

ghost_error_t ghost_endianessDiffers(int *differs, char *matrixPath)
{
    ghost_matfile_header_t header;
    GHOST_CALL_RETURN(ghost_readMatFileHeader(matrixPath,&header));

    if (header.endianess == GHOST_BINCRS_LITTLE_ENDIAN && ghost_machineIsBigEndian()) {
        *differs = 1;
    } else if (header.endianess != GHOST_BINCRS_LITTLE_ENDIAN && !ghost_machineIsBigEndian()) {
        *differs = 1;
    } else {
        *differs = 0;
    }

    return GHOST_SUCCESS;

}

ghost_error_t ghost_readMatFileHeader(char *matrixPath, ghost_matfile_header_t *header)
{
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
    if (header->endianess == GHOST_BINCRS_LITTLE_ENDIAN && ghost_machineIsBigEndian()) {
        DEBUG_LOG(1,"Need to convert from little to big endian.");
        swapReq = 1;
    } else if (header->endianess != GHOST_BINCRS_LITTLE_ENDIAN && !ghost_machineIsBigEndian()) {
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
    GHOST_CALL_RETURN(ghost_sizeofDatatype(&valSize,header->datatype));

    long rightFilesize = GHOST_BINCRS_SIZE_HEADER +
        (long)(header->nrows+1) * GHOST_BINCRS_SIZE_RPT_EL +
        (long)header->nnz * GHOST_BINCRS_SIZE_COL_EL +
        (long)header->nnz * valSize;

    if (filesize != rightFilesize) {
        ERROR_LOG("File has invalid size! (is: %ld, should be: %ld)",filesize, rightFilesize);
        return GHOST_ERR_IO;
    }

    fclose(file);

    return GHOST_SUCCESS;
}


