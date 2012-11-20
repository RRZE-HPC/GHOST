#ifndef _MY_ELLPACK_H_
#define _MY_ELLPACK_H_

#include "ghost_util.h"
#include <stdio.h>


#include <CL/cl.h>

typedef struct {
	int nrows, nMaxRow, padding;
	int* col;
	int* rowLen;
	mat_data_t* val;
	int *invRowPerm;
	int *rowPerm;
	int T;
} ELR_TYPE;

typedef struct {
	cl_int nrows, nMaxRow, padding;
	cl_mem col;
	cl_mem rowLen;
	cl_mem val;
} CL_ELR_TYPE;

typedef struct {
	int nrows, nMaxRow, padding, nEnts;
	int* col;
	int* colStart;
	int* rowLen;
	mat_data_t* val;
	int *invRowPerm;
	int *rowPerm;
	int T;
} PJDS_TYPE;

typedef struct {
	cl_int nrows, nMaxRow, padding, nEnts;
	cl_mem col;
	cl_mem colStart;
	cl_mem rowLen;
	cl_mem val;
} CL_PJDS_TYPE;

typedef struct {
	int row, col;
	mat_data_t val;
} MATRIX_ENTRY;

ELR_TYPE* CRStoELR( const mat_data_t*, const int*, const int*, const int ); 
ELR_TYPE* CRStoELRT(const mat_data_t*, const int*, const int*, const int, int);
ELR_TYPE* CRStoELRS(  const mat_data_t* , const int* , const int*,  const int);
ELR_TYPE* CRStoELRP(  const mat_data_t* , const int* , const int*,  const int, const int*);
ELR_TYPE* CRStoELRTP(  const mat_data_t* , const int* , const int*,  const int, const int*, int);
ELR_TYPE* MMtoELR(const char *, int);
void checkCRSToELR(	const mat_data_t* crs_val, const int* crs_col, const int* crs_row_ptr, const int nrows, const ELR_TYPE* elr);

//void checkCRStToPJDS(const mat_data_t* crs_val, const int* crs_col,const int* crs_row_ptr, const int nRow,	const PJDS_TYPE* pjds);
void resetELR( ELR_TYPE* elr );
void freeELR( ELR_TYPE* const elr );


PJDS_TYPE* CRStoPJDS( const mat_data_t*, const int*, const int*, const int); 
PJDS_TYPE* CRStoPJDST( const mat_data_t*, const int*, const int*, const int, const int); 
PJDS_TYPE* ELRStoPJDST( const ELR_TYPE*, int ); 
void checkCRStoPJDS( const mat_data_t*, const int*, const int*, const int, const PJDS_TYPE* ); // TODO
void resetPJDS( PJDS_TYPE* pjds );
void freePJDS( PJDS_TYPE* pjds );


CL_PJDS_TYPE* CL_initPJDS( const PJDS_TYPE* pjds );
void CL_uploadPJDS(CL_PJDS_TYPE* cpjds, const PJDS_TYPE* pjds );
void CL_downloadPJDS( PJDS_TYPE* pjds, const CL_PJDS_TYPE* cpjds );
void CL_freePJDS( CL_PJDS_TYPE* const cpjds );

CL_ELR_TYPE* CL_initELR( const ELR_TYPE* elr );
void CL_uploadELR(CL_ELR_TYPE* celr, const ELR_TYPE* elr );
void CL_downloadELR( ELR_TYPE* elr, const CL_ELR_TYPE* celr );
void CL_freeELR( CL_ELR_TYPE* const celr );

void CL_freeMatrix(void *matrix, int format);


void elrColIdToFortran( ELR_TYPE* elr );
void elrColIdToC( ELR_TYPE* elr );
size_t getBytesize(void *mat, int format); 
#endif
