/**
 * @file bincrs.h
 * @brief Types and functions for reading binary CRS files.
 * @author Moritz Kreutzer <moritz.kreutzer@fau.de>
 */
#ifndef GHOST_BINCRS_H
#define GHOST_BINCRS_H

#include "config.h"
#include "types.h"
#include "error.h"
#include "perm.h"
#include "context.h"
#include "sparsemat.h"

#include <stdio.h>

/**
 * @brief The header consumes 44 bytes.
 */
#define GHOST_BINCRS_SIZE_HEADER 44
/**
 * @brief One rpt element is 8 bytes.
 */
#define GHOST_BINCRS_SIZE_RPT_EL 8
/**
 * @brief One col element is 8 bytes.
 */
#define GHOST_BINCRS_SIZE_COL_EL 8
/**
 * @brief Indicates that the file is stored in little endianess.
 */
#define GHOST_BINCRS_LITTLE_ENDIAN 0

#define GHOST_BINCRS_SYMM_GENERAL GHOST_SPARSEMAT_SYMM_GENERAL
#define GHOST_BINCRS_SYMM_SYMMETRIC GHOST_SPARSEMAT_SYMM_SYMMETRIC
#define GHOST_BINCRS_SYMM_SKEW_SYMMETRIC GHOST_SPARSEMAT_SYMM_SKEW_SYMMETRIC
#define GHOST_BINCRS_SYMM_HERMITIAN  GHOST_SPARSEMAT_SYMM_HERMITIAN

#define GHOST_BINCRS_DT_FLOAT   GHOST_DT_FLOAT
#define GHOST_BINCRS_DT_DOUBLE  GHOST_DT_DOUBLE
#define GHOST_BINCRS_DT_REAL    GHOST_DT_REAL
#define GHOST_BINCRS_DT_COMPLEX GHOST_DT_COMPLEX

#define GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_INIT -1
#define GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_FINALIZE -2
#define GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETRPT -3
#define GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETDIM -4

typedef struct 
{
    char *filename;
    ghost_datatype_t dt;
} ghost_sparsemat_rowfunc_bincrs_initargs;

/**
 * @brief The header of a sparse matrix file.
 */
typedef struct { 
    /**
     * @brief The endianess of the machine on which the file has been written.
     */
    int32_t endianess;
    /**
     * @brief Version of the file format.
     */
    int32_t version;
    /**
     * @brief Base index. 0 for C, 1 for Fortran.
     */
    int32_t base;
    /**
     * @brief Matrix symmetry information.
     */
    int32_t symmetry;
    /**
     * @brief The data type of the matrix data.
     */
    int32_t datatype;
    /**
     * @brief The number of matrix rows.
     */
    int64_t nrows;
    /**
     * @brief The number of matrix columns.
     */
    int64_t ncols;
    /**
     * @brief The number of nonzeros in the matrix.
     */
    int64_t nnz;
} ghost_bincrs_header_t;

#ifdef __cplusplus
template<typename m_t, typename f_t> void ghost_castarray_tmpl(void *out, void *in, int nEnts);
extern "C" {
#endif
/**
 * @brief 
 *
 * @param row
 * @param rowlen
 * @param col
 * @param val
 *
 * @return 
 *
 * If called with row #GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_INIT, the parameter
 * \p val has to be a ghost_sparsemat_rowfunc_bincrs_initargs * with the according
 * information filled in. The parameter \p col has to be a ghost_gidx_t[2] in 
 * which the number of rows and columns will be stored.
 */
int ghost_sparsemat_rowfunc_bincrs(ghost_gidx_t row, ghost_lidx_t *rowlen, ghost_gidx_t *col, void *val, void *arg);
    ghost_error_t ghost_bincrs_header_read(ghost_bincrs_header_t *header, char *path);

#if 0

    ghost_error_t ghost_bincrs_header_read(ghost_bincrs_header_t *header, char *path);
    ghost_error_t ghost_bincrs_col_read(ghost_gidx_t *col, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_context_t *context, int keepCols);
    ghost_error_t ghost_bincrs_col_read_opened(ghost_gidx_t *col, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_context_t *context, int keepCols, FILE *filed);
    ghost_error_t ghost_bincrs_val_read(char *val, ghost_datatype_t datatype, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm);
    ghost_error_t ghost_bincrs_val_read_opened(char *val, ghost_datatype_t datatype, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm, FILE *filed);
    ghost_error_t ghost_bincrs_rpt_read(ghost_gidx_t *rpt, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm);
    ghost_error_t ghost_bincrs_rpt_read_opened(ghost_gidx_t *rpt, char *matrixPath, ghost_gidx_t offsRows, ghost_lidx_t nRows, ghost_permutation_t *perm, FILE *filed);

    void dd_ghost_castarray(void *, void *, int);
    void ds_ghost_castarray(void *, void *, int);
    void dc_ghost_castarray(void *, void *, int);
    void dz_ghost_castarray(void *, void *, int);
    void sd_ghost_castarray(void *, void *, int);
    void ss_ghost_castarray(void *, void *, int);
    void sc_ghost_castarray(void *, void *, int);
    void sz_ghost_castarray(void *, void *, int);
    void cd_ghost_castarray(void *, void *, int);
    void cs_ghost_castarray(void *, void *, int);
    void cc_ghost_castarray(void *, void *, int);
    void cz_ghost_castarray(void *, void *, int);
    void zd_ghost_castarray(void *, void *, int);
    void zs_ghost_castarray(void *, void *, int);
    void zc_ghost_castarray(void *, void *, int);
    void zz_ghost_castarray(void *, void *, int);
#endif

#ifdef __cplusplus
}
#endif
#endif

