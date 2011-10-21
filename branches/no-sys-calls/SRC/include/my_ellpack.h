#ifndef _MY_ELLPACK_H_
#define _MY_ELLPACK_H_

typedef struct {
	int nRows, nMaxRow, padding;
	int* col;
	int* rowLen;
	double* val;
} ELR_TYPE;

typedef struct {
	int nRows, nMaxRow, padding;
	int* col;
	int* rowLen;
	double* val;
} CUDA_ELR_TYPE;


ELR_TYPE* convertCRSToELRMatrix( const double*, const int*, const int*, const int ); 
void checkCRSToELRsanity( const double*, const int*, const int*, const int, const ELR_TYPE* );
void resetELR( ELR_TYPE* elr );

CUDA_ELR_TYPE* cudaELRInit( const ELR_TYPE* elr );

void cudaCopyELRToDevice(CUDA_ELR_TYPE* celr, const ELR_TYPE* elr );
void cudaCopyELRBackToHost( ELR_TYPE* elr, const CUDA_ELR_TYPE* celr );

void freeELRMatrix( ELR_TYPE* const elr );

void freeCUDAELRMatrix( CUDA_ELR_TYPE* const celr );


#endif
