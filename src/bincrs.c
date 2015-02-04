#define _XOPEN_SOURCE 500
#include "ghost/bincrs.h"
#include "ghost/log.h"
#include "ghost/util.h"
#include "ghost/machine.h"
#include <errno.h>
#include <limits.h>

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

static void (*ghost_castarray_funcs[4][4]) (void *, void *, int) = 
{{&ss_ghost_castarray,&sd_ghost_castarray,&sc_ghost_castarray,&sz_ghost_castarray},
    {&ds_ghost_castarray,&dd_ghost_castarray,&dc_ghost_castarray,&dz_ghost_castarray},
    {&cs_ghost_castarray,&cd_ghost_castarray,&cc_ghost_castarray,&cz_ghost_castarray},
    {&zs_ghost_castarray,&zd_ghost_castarray,&zc_ghost_castarray,&zz_ghost_castarray}};


ghost_error_t ghost_bincrs_col_read_opened(ghost_gidx_t *col, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm, int keepCols, FILE *filed)
{
    ghost_bincrs_header_t header;
    size_t ret;
    int swapReq;
    off64_t offs;
    ghost_gidx_t i,j,e;

    GHOST_CALL_RETURN(ghost_bincrs_header_read(&header,matrixPath));
    swapReq = SWAPREQ(header);
    
    
    if (perm) {
        if ( perm->scope == GHOST_PERMUTATION_GLOBAL) {
            int64_t *rpt_raw;
            GHOST_CALL_RETURN(ghost_malloc((void **)&rpt_raw,(header.nrows+1)*8));
            if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
                ERROR_LOG("Seek failed");
                return GHOST_ERR_IO;
            }
            if (swapReq) {
                int64_t *tmp;
                GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,(header.nrows+1)*8));
                if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),filed)) != (size_t)(header.nrows+1)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
                for( i = 0; i < (header.nrows+1); i++ ) {
                    rpt_raw[i] = bswap_64(tmp[i]);
                }
                free(tmp);
            } else {
                if ((ret = fread(rpt_raw, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),filed)) != (size_t)(header.nrows+1)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
            }
            e = 0;
            for(i = offsRows; i < offsRows+nRows; i++) {
                if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER+GHOST_BINCRS_SIZE_RPT_EL*(header.nrows+1)+perm->invPerm[i]*GHOST_BINCRS_SIZE_COL_EL,SEEK_SET)) {
                    ERROR_LOG("Seek failed");
                    return GHOST_ERR_IO;
                }
                ghost_lidx_t rowlen = rpt_raw[perm->invPerm[i]]+1-rpt_raw[perm->invPerm[i]];
                ghost_gidx_t rawcol[rowlen];
                if ((ret = fread(rawcol, GHOST_BINCRS_SIZE_COL_EL, rowlen,filed)) != (size_t)(rowlen)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }

                if (keepCols) {
                    memcpy(&col[e],rawcol,rowlen*sizeof(ghost_gidx_t));
                    e+=rowlen;
                } else {
                    for(j = 0; j < rowlen; j++) {
                        col[e++] = perm->perm[rawcol[j]];
                    }
                }
            }

            //free(col_raw);
            free(rpt_raw);
        } else {
            int64_t *rpt_raw;
            GHOST_CALL_RETURN(ghost_malloc((void **)&rpt_raw,(header.nrows+1)*8));
            if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
                ERROR_LOG("Seek failed");
                return GHOST_ERR_IO;
            }
            if (swapReq) {
                int64_t *tmp;
                GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,(header.nrows+1)*8));
                if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),filed)) != (size_t)(header.nrows+1)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
                for( i = 0; i < (header.nrows+1); i++ ) {
                    rpt_raw[i] = bswap_64(tmp[i]);
                }
                free(tmp);
            } else {
                if ((ret = fread(rpt_raw, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),filed)) != (size_t)(header.nrows+1)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
            }
            int64_t *col_raw;
            GHOST_CALL_RETURN(ghost_malloc((void **)&col_raw,header.nnz*8));
            if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER+GHOST_BINCRS_SIZE_RPT_EL*(header.nrows+1),SEEK_SET)) {
                ERROR_LOG("Seek failed");
                return GHOST_ERR_IO;
            }
            if (swapReq) {
                int64_t *tmp;
                GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,header.nnz*8));
                if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, header.nnz,filed)) != (size_t)(header.nnz)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
                for( i = 0; i < (header.nrows+1); i++ ) {
                    col_raw[i] = bswap_64(tmp[i]);
                }
                free(tmp);
            } else {
                if ((ret = fread(col_raw, GHOST_BINCRS_SIZE_COL_EL, header.nnz,filed)) != (size_t)(header.nnz)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
            }
            e = 0;
            //INFO_LOG("Read %d rows from row %d perm len %d %d %d %d %d",nRows,offsRows,perm->len,perm->perm[0],perm->perm[1],perm->perm[2],perm->perm[3]);
            for(i = offsRows; i < offsRows+nRows; i++) {
                for(j = rpt_raw[perm->invPerm[i-offsRows]]; j < rpt_raw[perm->invPerm[i-offsRows]+1]; j++) {
                    if (!keepCols) {
                        if ((col_raw[j]>=offsRows+nRows) || (col_raw[j] < offsRows)) {
                            col[e++] = col_raw[j];
                        } else {
                            col[e++] = offsRows+perm->perm[col_raw[j]-offsRows];
                        }
                    } else {
                        col[e++] = col_raw[j];
                    }

                }
            }

            free(col_raw);
            free(rpt_raw);
        }
    } else {
        int64_t fEnt, lEnt;
        if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER+GHOST_BINCRS_SIZE_RPT_EL*offsRows,SEEK_SET)) {
            ERROR_LOG("Seek failed");
            return GHOST_ERR_IO;
        }
        if ((ret = fread(&fEnt, GHOST_BINCRS_SIZE_COL_EL, 1,filed)) != 1){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
        if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER+GHOST_BINCRS_SIZE_RPT_EL*(offsRows+nRows),SEEK_SET)) {
            ERROR_LOG("Seek failed");
            return GHOST_ERR_IO;
        }
        if ((ret = fread(&lEnt, GHOST_BINCRS_SIZE_COL_EL, 1,filed)) != 1){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }

        ghost_lidx_t nEnts = lEnt-fEnt;

        DEBUG_LOG(1,"Reading array with column indices");
        offs = GHOST_BINCRS_SIZE_HEADER+
            GHOST_BINCRS_SIZE_RPT_EL*(header.nrows+1)+
            GHOST_BINCRS_SIZE_COL_EL*fEnt;
        if (fseeko(filed,offs,SEEK_SET)) {
            ERROR_LOG("Seek failed");
            return GHOST_ERR_IO;
        }

#ifdef GHOST_HAVE_LONGIDX_GLOBAL
        if (swapReq) {
            int64_t *tmp;
            GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,nEnts*8));
            if ((ret = fread(tmp, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (size_t)(nEnts)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
            for( i = 0; i < nEnts; i++ ) {
                col[i] = bswap_64(tmp[i]);
            }
        } else {
            if ((ret = fread(col, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (size_t)(nEnts)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
        }
#else // casting from 64 to 32 bit
        DEBUG_LOG(1,"Casting from 64 bit to 32 bit column indices");
        int64_t *tmp;
        GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,nEnts*8));
        if ((ret = fread(tmp, GHOST_BINCRS_SIZE_COL_EL, nEnts,filed)) != (size_t)(nEnts)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
        for(i = 0 ; i < nEnts; ++i) {
            if (tmp[i] >= (int64_t)INT_MAX) {
                ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
                return GHOST_ERR_IO;
            }
            if (swapReq) {
                col[i] = (ghost_lidx_t)(bswap_64(tmp[i]));
            } else {
                col[i] = (ghost_lidx_t)tmp[i];
            }
        }
        free(tmp);
#endif
    }

    return GHOST_SUCCESS;
}


ghost_error_t ghost_bincrs_col_read(ghost_gidx_t *col, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm, int keepCols)
{
    FILE *filed;

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s",matrixPath);
        return GHOST_ERR_IO;
    }

    GHOST_CALL_RETURN(ghost_bincrs_col_read_opened(col,matrixPath,offsRows,nRows,perm,keepCols,filed));

    fclose(filed);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_bincrs_val_read_opened(char *val, ghost_datatype_t datatype, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm, FILE *filed)
{
    ghost_bincrs_header_t header;
    size_t ret;
    int swapReq;
    off64_t offs;
    ghost_lidx_t i,j,e;
    size_t sizeofdt;
    GHOST_CALL_RETURN(ghost_datatype_size(&sizeofdt,datatype));
    GHOST_CALL_RETURN(ghost_bincrs_header_read(&header,matrixPath));
    swapReq = SWAPREQ(header);

    size_t valSize;
    GHOST_CALL_RETURN(ghost_datatype_size(&valSize,(ghost_datatype_t)header.datatype));

    int64_t *rpt_raw;
    GHOST_CALL_RETURN(ghost_malloc((void **)&rpt_raw,(header.nrows+1)*8));
    if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
        ERROR_LOG("Seek failed");
        return GHOST_ERR_IO;
    }
    if (swapReq) {
        int64_t *tmp;
        GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,(header.nrows+1)*8));
        if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),filed)) != (size_t)(header.nrows+1)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
        for( i = 0; i < (header.nrows+1); i++ ) {
            rpt_raw[i] = bswap_64(tmp[i]);
        }
        free(tmp);
    } else {
        if ((ret = fread(rpt_raw, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),filed)) != (size_t)(header.nrows+1)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
    }
        
    
    DEBUG_LOG(1,"Reading array with values");
        
    char *val_raw;
    ghost_lidx_t nEnts;

    if (perm) {
        GHOST_CALL_RETURN(ghost_malloc((void **)&val_raw,header.nnz*sizeofdt));
        nEnts = header.nnz;
        offs = GHOST_BINCRS_SIZE_HEADER+
            GHOST_BINCRS_SIZE_RPT_EL*(header.nrows+1)+
            GHOST_BINCRS_SIZE_COL_EL*header.nnz;
    } else {
        val_raw = val;
        nEnts = rpt_raw[nRows+offsRows]-rpt_raw[offsRows];
        offs = GHOST_BINCRS_SIZE_HEADER+
            GHOST_BINCRS_SIZE_RPT_EL*(header.nrows+1)+
            GHOST_BINCRS_SIZE_COL_EL*header.nnz+
            valSize*rpt_raw[offsRows];
    }
    if (fseeko(filed,offs,SEEK_SET)) {
        ERROR_LOG("Seek failed");
        return GHOST_ERR_IO;
    }

    if ((int)datatype == header.datatype) {
        if (swapReq) {
            uint8_t *tmpval;
            GHOST_CALL_RETURN(ghost_malloc((void **)&tmpval,nEnts*valSize));
            if ((ghost_lidx_t)(ret = fread(tmpval, valSize, nEnts,filed)) != (nEnts)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
            if (datatype & GHOST_BINCRS_DT_COMPLEX) {
                if (datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t *a = (uint32_t *)tmpval;
                        uint32_t rswapped = bswap_32(a[2*i]);
                        uint32_t iswapped = bswap_32(a[2*i+1]);
                        memcpy(&(val_raw[i]),&rswapped,4);
                        memcpy(&(val_raw[i])+4,&iswapped,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint64_t *a = (uint64_t *)tmpval;
                        uint64_t rswapped = bswap_64(a[2*i]);
                        uint64_t iswapped = bswap_64(a[2*i+1]);
                        memcpy(&(val_raw[i]),&rswapped,8);
                        memcpy(&(val_raw[i])+8,&iswapped,8);
                    }
                }
            } else {
                if (datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t *a = (uint32_t *)tmpval;
                        uint32_t swapped = bswap_32(a[i]);
                        memcpy(&(val_raw[i]),&swapped,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint64_t *a = (uint64_t *)tmpval;
                        uint64_t swapped = bswap_64(a[i]);
                        memcpy(&(val_raw[i]),&swapped,8);
                    }
                }

            }
        } else {
            if ((ghost_lidx_t)(ret = fread(val_raw, valSize, nEnts,filed)) != (nEnts)){
                ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                return GHOST_ERR_IO;
            }
        }
    } else {
        INFO_LOG("This matrix is supposed to be of %s data but"
                " the file contains %s data. Casting...",ghost_datatype_string(datatype),ghost_datatype_string((ghost_datatype_t)header.datatype));


        uint8_t *tmpval;
        GHOST_CALL_RETURN(ghost_malloc((void **)&tmpval,nEnts*valSize));
        if ((ghost_lidx_t)(ret = fread(tmpval, valSize, nEnts,filed)) != (nEnts)){
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
                        memcpy(&val_raw[i*sizeofdt],&re,4);
                        memcpy(&val_raw[i*sizeofdt+4],&im,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t re = bswap_64(tmpval[i*valSize]);
                        uint32_t im = bswap_64(tmpval[i*valSize+valSize/2]);
                        memcpy(&val_raw[i*sizeofdt],&re,8);
                        memcpy(&val_raw[i*sizeofdt+8],&im,8);
                    }
                }
            } else {
                if (datatype & GHOST_BINCRS_DT_FLOAT) {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t swappedVal = bswap_32(tmpval[i*valSize]);
                        memcpy(&val_raw[i*sizeofdt],&swappedVal,4);
                    }
                } else {
                    for (i = 0; i<nEnts; i++) {
                        uint32_t swappedVal = bswap_64(tmpval[i*valSize]);
                        memcpy(&val_raw[i*sizeofdt],&swappedVal,8);
                    }
                }

            }

        } else {
            ghost_datatype_idx_t matDtIdx;
            ghost_datatype_idx_t headerDtIdx;
            GHOST_CALL_RETURN(ghost_datatype_idx(&matDtIdx,datatype));
            GHOST_CALL_RETURN(ghost_datatype_idx(&headerDtIdx,(ghost_datatype_t)header.datatype));

            ghost_castarray_funcs[matDtIdx][headerDtIdx](val_raw,tmpval,nEnts);
        }

        free(tmpval);
    }

    if (perm) {
        e = 0;
        if (perm->scope == GHOST_PERMUTATION_GLOBAL) {
            for(i = offsRows; i < offsRows+nRows; i++) {
                for(j = rpt_raw[perm->invPerm[i]]; j < rpt_raw[perm->invPerm[i]+1]; j++) {
                    memcpy(val+e*sizeofdt,val_raw+j*sizeofdt,sizeofdt);
                    e++;
                }
            }
        } else {
            for(i = offsRows; i < offsRows+nRows; i++) {
                for(j = rpt_raw[offsRows+perm->invPerm[i-offsRows]]; j < rpt_raw[offsRows+perm->invPerm[i-offsRows]+1]; j++) {
                    memcpy(val+e*sizeofdt,val_raw+j*sizeofdt,sizeofdt);
                    e++;
                }
            }
        }

        free(val_raw);

    }

    return GHOST_SUCCESS;
}

ghost_error_t ghost_bincrs_val_read(char *val, ghost_datatype_t datatype, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm)
{
    FILE *filed;

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s",matrixPath);
        return GHOST_ERR_IO;
    }

    ghost_bincrs_val_read_opened(val,datatype,matrixPath,offsRows,nRows,perm,filed);

    fclose(filed);

    return GHOST_SUCCESS;
}

ghost_error_t ghost_bincrs_rpt_read_opened(ghost_gidx_t *rpt, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm, FILE *filed)
{
    ghost_bincrs_header_t header;
    size_t ret;
    int swapReq;
    off64_t offs;

    ghost_gidx_t i;

    GHOST_CALL_RETURN(ghost_bincrs_header_read(&header,matrixPath));
    swapReq = SWAPREQ(header);

    if (perm) {
        if ( perm->scope == GHOST_PERMUTATION_GLOBAL) {
            int64_t *rpt_raw;
            GHOST_CALL_RETURN(ghost_malloc((void **)&rpt_raw,(header.nrows+1)*8));
            if (fseeko(filed,GHOST_BINCRS_SIZE_HEADER,SEEK_SET)) {
                ERROR_LOG("Seek failed");
                return GHOST_ERR_IO;
            }
            if (swapReq) {
                int64_t *tmp;
                GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,(header.nrows+1)*8));
                if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),filed)) != (size_t)(header.nrows+1)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
                for( i = 0; i < (header.nrows+1); i++ ) {
                    rpt_raw[i] = bswap_64(tmp[i]);
                }
                free(tmp);
            } else {
                if ((ret = fread(rpt_raw, GHOST_BINCRS_SIZE_RPT_EL, (header.nrows+1),filed)) != (size_t)(header.nrows+1)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
            }
                rpt[0] = rpt_raw[perm->invPerm[offsRows]];
                for( i = 1; i < nRows; i++ ) {
                    rpt[i] = rpt[i-1]+(rpt_raw[perm->invPerm[i-1]+1]-rpt_raw[perm->invPerm[i-1]]);
                }

            free(rpt_raw);
            return GHOST_SUCCESS;
        } else {
            int64_t *rpt_raw;
            GHOST_CALL_RETURN(ghost_malloc((void **)&rpt_raw,nRows*8));
            offs = GHOST_BINCRS_SIZE_HEADER+
                GHOST_BINCRS_SIZE_RPT_EL*offsRows;
            
            if (fseeko(filed,offs,SEEK_SET)) {
                ERROR_LOG("Seek failed");
                return GHOST_ERR_IO;
            }
            if (swapReq) {
                int64_t *tmp;
                GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,nRows*8));
                if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, nRows,filed)) != (size_t)(nRows)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
                for( i = 0; i < (nRows); i++ ) {
                    rpt_raw[i] = bswap_64(tmp[i]);
                }
                free(tmp);
            } else {
                if ((ret = fread(rpt_raw, GHOST_BINCRS_SIZE_RPT_EL, nRows,filed)) != (size_t)(nRows)){
                    ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
                    return GHOST_ERR_IO;
                }
            }
                rpt[0] = rpt_raw[0];
                for( i = 1; i < nRows; i++ ) {
                    rpt[i] = rpt[i-1]+(rpt_raw[perm->invPerm[i-1]+1]-rpt_raw[perm->invPerm[i-1]]);
                }

            free(rpt_raw);
            return GHOST_SUCCESS;
        }

    }



    DEBUG_LOG(1,"Reading array with column indices");
    offs = GHOST_BINCRS_SIZE_HEADER+
        GHOST_BINCRS_SIZE_RPT_EL*offsRows;
    if (fseeko(filed,offs,SEEK_SET)) {
        ERROR_LOG("Seek failed");
        return GHOST_ERR_IO;
    }

#ifdef GHOST_HAVE_LONGIDX_GLOBAL
        if (swapReq) {
            int64_t *tmp;
            GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,nRows*8));
            if ((ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, nRows,filed)) != (size_t)(nRows)){
                ERROR_LOG("fread failed with %"PRLIDX" rows: %s (%zu)",nRows,strerror(errno),ret);
                return GHOST_ERR_IO;
            }
            for( i = 0; i < nRows; i++ ) {
                rpt[i] = bswap_64(tmp[i]);
            }
            free(tmp);
        } else {
            if ((ret = fread(rpt, GHOST_BINCRS_SIZE_RPT_EL, nRows,filed)) != (size_t)(nRows)){
                ERROR_LOG("fread failed with %"PRLIDX" rows: %s (%zu)",nRows,strerror(errno),ret);
                return GHOST_ERR_IO;
            }
        }
#else // casting from 64 to 32 bit
        DEBUG_LOG(1,"Casting from 64 bit to 32 bit column indices");
        int64_t *tmp;
        GHOST_CALL_RETURN(ghost_malloc((void **)&tmp,nRows*8));
        if ((ghost_lidx_t)(ret = fread(tmp, GHOST_BINCRS_SIZE_RPT_EL, nRows,filed)) != (nRows)){
            ERROR_LOG("fread failed: %s (%zu)",strerror(errno),ret);
            return GHOST_ERR_IO;
        }
        for(i = 0 ; i < nRows; ++i) {
            if (tmp[i] >= (int64_t)INT_MAX) {
                ERROR_LOG("The matrix is too big for 32-bit indices. Recompile with LONGIDX!");
                return GHOST_ERR_IO;
            }
            if (swapReq) {
                rpt[i] = (ghost_lidx_t)(bswap_64(tmp[i]));
            } else {
                rpt[i] = (ghost_lidx_t)tmp[i];
            }
        }
        free(tmp);
#endif


    return GHOST_SUCCESS;
}

ghost_error_t ghost_bincrs_rpt_read(ghost_gidx_t *rpt, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm)
{
    FILE *filed;

    if ((filed = fopen64(matrixPath, "r")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s",matrixPath);
        return GHOST_ERR_IO;
    }

    GHOST_CALL_RETURN(ghost_bincrs_rpt_read_opened(rpt,matrixPath,offsRows,nRows,perm,filed));

    fclose(filed);

    return GHOST_SUCCESS;

}

/*ghost_error_t ghost_endianessDiffers(int *differs, char *matrixPath)
{
    ghost_bincrs_header_t header;
    GHOST_CALL_RETURN(ghost_bincrs_header_read(matrixPath,&header));

    if (header.endianess == GHOST_BINCRS_LITTLE_ENDIAN && ghost_machine_bigendian()) {
        *differs = 1;
    } else if (header.endianess != GHOST_BINCRS_LITTLE_ENDIAN && !ghost_machine_bigendian()) {
        *differs = 1;
    } else {
        *differs = 0;
    }

    return GHOST_SUCCESS;

}*/

ghost_error_t ghost_bincrs_header_read(ghost_bincrs_header_t *header, char *matrixPath)
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
    GHOST_CALL_RETURN(ghost_datatype_size(&valSize,(ghost_datatype_t)header->datatype));

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


