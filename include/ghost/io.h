/**
 * @file io.h
 * @brief Types and functions for data I/O.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_IO_H
#define GHOST_IO_H

#include "config.h"
#include "types.h"
#include "error.h"
#include "sparsemat.h"

#include <stdio.h>

#define GHOST_BINCRS_SIZE_HEADER 44 /* header consumes 44 bytes */
#define GHOST_BINCRS_SIZE_RPT_EL 8 /* one rpt element is 8 bytes */
#define GHOST_BINCRS_SIZE_COL_EL 8 /* one col element is 8 bytes */

#define GHOST_BINVEC_SIZE_HEADER 32
#define GHOST_BINCRS_LITTLE_ENDIAN (0)

#define GHOST_BINCRS_SYMM_GENERAL GHOST_SPARSEMAT_SYMM_GENERAL
#define GHOST_BINCRS_SYMM_SYMMETRIC GHOST_SPARSEMAT_SYMM_SYMMETRIC
#define GHOST_BINCRS_SYMM_SKEW_SYMMETRIC GHOST_SPARSEMAT_SYMM_SKEW_SYMMETRIC
#define GHOST_BINCRS_SYMM_HERMITIAN  GHOST_SPARSEMAT_SYMM_HERMITIAN

#define GHOST_BINCRS_DT_FLOAT   GHOST_DT_FLOAT
#define GHOST_BINCRS_DT_DOUBLE  GHOST_DT_DOUBLE
#define GHOST_BINCRS_DT_REAL    GHOST_DT_REAL
#define GHOST_BINCRS_DT_COMPLEX GHOST_DT_COMPLEX

#define GHOST_BINVEC_ORDER_COL_FIRST 0
#define GHOST_BINVEC_ORDER_ROW_FIRST 1

/**
 * @brief The header of a sparse matrix file.
 */
typedef struct {
    int32_t endianess;
    int32_t version;
    int32_t base;
    int32_t symmetry;
    int32_t datatype;
    int64_t nrows;
    int64_t ncols;
    int64_t nnz;
} ghost_matfile_header_t;

#ifdef __cplusplus
template<typename m_t, typename f_t> void ghost_castArray_tmpl(void *out, void *in, int nEnts);
extern "C" {
#endif

    ghost_error_t ghost_readMatFileHeader(char *, ghost_matfile_header_t *);
    ghost_error_t ghost_readCol(ghost_idx_t *col, char *matrixPath, ghost_nnz_t offsRows, ghost_nnz_t nRows, ghost_permutation_t *perm, int keepCols);
    ghost_error_t ghost_readColOpen(ghost_idx_t *col, char *matrixPath, ghost_nnz_t offsRows, ghost_nnz_t nRows, ghost_permutation_t *perm, int keepCols, FILE *filed);
    ghost_error_t ghost_readVal(char *val, int datatype, char *matrixPath, ghost_nnz_t offsRows, ghost_nnz_t nRows, ghost_permutation_t *perm);
    ghost_error_t ghost_readValOpen(char *val, int datatype, char *matrixPath, ghost_nnz_t offsRows, ghost_nnz_t nRows, ghost_permutation_t *perm, FILE *filed);
    ghost_error_t ghost_readRpt(ghost_nnz_t *rpt, char *matrixPath, ghost_nnz_t offsRows, ghost_nnz_t nRows, ghost_permutation_t *perm);
    ghost_error_t ghost_readRptOpen(ghost_idx_t *rpt, char *matrixPath, ghost_nnz_t offsRows, ghost_nnz_t nRows, ghost_permutation_t *perm, FILE *filed);

    /**
     * @brief Check if the machine endianess differs from the sparse matrix file.
     *
     * @param differs Where to store the result.
     * @param matrixPath The sparse matrix file.
     *
     * @return GHOST_SUCCESS on success or an error indicator.
     */
    ghost_error_t ghost_endianessDiffers(int *differs, char *matrixPath);

    extern void (*ghost_castArray_funcs[4][4]) (void *, void *, int); 
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


#ifdef __cplusplus
}
#endif
#endif

