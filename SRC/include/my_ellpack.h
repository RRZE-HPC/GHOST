#ifndef _MY_ELLPACK_H_
#define _MY_ELLPACK_H_

#include <stdio.h>
#include <CL/cl.h>

typedef struct {
	int nRows, nMaxRow, padding;
} ELR_PROPS;

typedef struct {
	int nRows, nMaxRow, padding;
	int* col;
	int* rowLen;
	double* val;
	int *invRowPerm;
	int *rowPerm;
} ELR_TYPE;

typedef struct {
	int nRows, nMaxRow, padding;
	int* col;
	int* rowLen;
	double* val;
} CUDA_ELR_TYPE;

typedef struct {
	cl_int nRows, nMaxRow, padding;
	cl_mem col;
	cl_mem rowLen;
	cl_mem val;
} CL_ELR_TYPE;


ELR_TYPE* convertCRSToELRMatrix( const double*, const int*, const int*, const int ); 
void checkCRSToELRsanity( const double*, const int*, const int*, const int, const ELR_TYPE* );
void resetELR( ELR_TYPE* elr );
void elrColIdToFortran( ELR_TYPE* elr );
void elrColIdToC( ELR_TYPE* elr );

CUDA_ELR_TYPE* cudaELRInit( const ELR_TYPE* elr );

void cudaCopyELRToDevice(CUDA_ELR_TYPE* celr, const ELR_TYPE* elr );
void cudaCopyELRBackToHost( ELR_TYPE* elr, const CUDA_ELR_TYPE* celr );

void freeELRMatrix( ELR_TYPE* const elr );

void freeCUDAELRMatrix( CUDA_ELR_TYPE* const celr );


CL_ELR_TYPE* CL_ELRInit( const ELR_TYPE* elr );

void CL_CopyELRToDevice(CL_ELR_TYPE* celr, const ELR_TYPE* elr );
void CL_CopyELRBackToHost( ELR_TYPE* elr, const CL_ELR_TYPE* celr );


void CLfreeELRMatrix( CL_ELR_TYPE* const celr );


ELR_TYPE* convertCRSToELRSortedMatrix(  const double* , const int* , const int*,  const int);
ELR_TYPE* convertCRSToELRPermutedMatrix(  const double* , const int* , const int*,  const int,	const int*, const int*);
/************ PJDS *******/

typedef struct {
	int nRows, nMaxRow, padding, nEnts;
	int* col;
	int* colStart;
	int* rowLen;
	double* val;
	int *invRowPerm;
	int *rowPerm;
} PJDS_TYPE;

typedef struct {
	int nRows, nMaxRow, padding, nEnts;
	int* col;
	int* colStart;
	int* rowLen;
	double* val;
} CUDA_PJDS_TYPE;

typedef struct {
	cl_int nRows, nMaxRow, padding, nEnts;
	cl_mem col;
	cl_mem colStart;
	cl_mem rowLen;
	cl_mem val;
} CL_PJDS_TYPE;


PJDS_TYPE* convertCRSToPJDSMatrix( const double*, const int*, const int*, const int); 
PJDS_TYPE* convertELRSortedToPJDSMatrix( const ELR_TYPE* ); 
void checkCRSToPJDSsanity( const double*, const int*, const int*, const int, const PJDS_TYPE*, const int* ); // TODO
void resetPJDS( PJDS_TYPE* pjds );

CUDA_PJDS_TYPE* cudaPJDSInit( const PJDS_TYPE* pjds );

void cudaCopyPJDSToDevice(CUDA_PJDS_TYPE* cpjds, const PJDS_TYPE* pjds );
void cudaCopyPJDSBackToHost( PJDS_TYPE* pjds, const CUDA_PJDS_TYPE* cpjds );

void freePJDSMatrix( PJDS_TYPE* const pjds );

void freeCUDAPJDSMatrix( CUDA_PJDS_TYPE* const cpjds );



CL_PJDS_TYPE* CL_PJDSInit( const PJDS_TYPE* pjds );

void CL_CopyPJDSToDevice(CL_PJDS_TYPE* cpjds, const PJDS_TYPE* pjds );
void CL_CopyPJDSBackToHost( PJDS_TYPE* pjds, const CL_PJDS_TYPE* cpjds );


void CL_freePJDSMatrix( CL_PJDS_TYPE* const cpjds );
#endif
