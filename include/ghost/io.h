#ifndef GHOST_IO_H
#define GHOST_IO_H

#include "config.h"
#include "types.h"
#include "error.h"

#include <stdio.h>

#ifdef __cplusplus
template<typename m_t, typename f_t> void ghost_castArray_tmpl(void *out, void *in, int nEnts);
extern "C" {
#endif

ghost_error_t ghost_readMatFileHeader(char *, ghost_matfile_header_t *);
ghost_error_t ghost_readCol(ghost_midx_t *col, char *matrixPath, ghost_mnnz_t offs, ghost_mnnz_t nEnts);
ghost_error_t ghost_readColOpen(ghost_midx_t *col, char *matrixPath, ghost_mnnz_t offsEnts, ghost_mnnz_t nEnts, FILE *filed);
ghost_error_t ghost_readVal(char *val, int datatype, char *matrixPath, ghost_mnnz_t offs, ghost_mnnz_t nEnts);
ghost_error_t ghost_readValOpen(char *val, int datatype, char *matrixPath, ghost_mnnz_t offs, ghost_mnnz_t nEnts, FILE *filed);
ghost_error_t ghost_readRpt(ghost_mnnz_t *rpt, char *matrixPath, ghost_mnnz_t offsRows, ghost_mnnz_t nRows);
ghost_error_t ghost_readRptOpen(ghost_midx_t *rpt, char *matrixPath, ghost_mnnz_t offsRows, ghost_mnnz_t nRows, FILE *filed);

ghost_error_t ghost_endianessDiffers(int *differs, char *matrixPath);
void dd_ghost_castArray(void *, void *, int);
void ds_ghost_castArray(void *, void *, int);
void dc_ghost_castArray(void *, void *, int);
void dz_ghost_castArray(void *, void *, int);
void sd_ghost_castArray(void *, void *, int);
void ss_ghost_castArray(void *, void *, int);
void sc_ghost_castArray(void *, void *, int);
void sz_ghost_castArray(void *, void *, int);
void cd_ghost_castArray(void *, void *, int);
void cs_ghost_castArray(void *, void *, int);
void cc_ghost_castArray(void *, void *, int);
void cz_ghost_castArray(void *, void *, int);
void zd_ghost_castArray(void *, void *, int);
void zs_ghost_castArray(void *, void *, int);
void zc_ghost_castArray(void *, void *, int);
void zz_ghost_castArray(void *, void *, int);

extern void (*ghost_castArray_funcs[4][4]) (void *, void *, int); 
#ifdef __cplusplus
}
#endif
#endif
