#ifndef _MATRICKS_H_
#define _MATRICKS_H_

#include "spmvm.h"


#include <stdio.h>
#include <stdlib.h>



typedef struct {
	int row, col, nThEntryInRow;
	mat_data_t val;
} NZE_TYPE;

typedef struct {
	int nRows, nCols, nEnts;
	NZE_TYPE* nze;
} MM_TYPE;


typedef struct {
	int nRows, nCols, nEnts, nDiags;
	int* rowPerm;
	int* diagOffset;
	int* col;
	mat_data_t* val;
} JD_TYPE;

typedef struct {
	int row, nEntsInRow;
} JD_SORT_TYPE;


typedef struct {
        int nEnts, nRows, totalblocks;
        int* blockdim_rows;
        int* blockdim_cols;
        int* resorted_col;
        mat_data_t* resorted_val;
        int* blockinfo;
        int* tbi;
} JD_RESORTED_TYPE;

typedef struct {
	int pagesize, cachesize;
        int vecdim;
        int ppvec, offset, numvecs, globdim;
        mat_data_t* mem;
        mat_data_t** vec;
} REVBUF_TYPE;


typedef struct {
        int nEnts, nRows, nDiags, totalblocks;
        int* blockdim_rows;
        int* blockdim_cols;
        int* rowPerm;
        int* diagOffset;
        int* col;
        int* resorted_col;
        mat_data_t* val;
        mat_data_t* resorted_val;
        int* blockinfo;
        int* tbi;
} JD_OFFDIAGONAL_TYPE;

typedef struct {
  int nEnts;
  int nRows;
} LMMP_TYPE;


typedef struct {
  int nodes, threads;
  int* on_me;
  int* pseudo_on_me;
  int* wish;
  int* wishlist;
  int* pseudo_nidx;
  int* glob2loc;
  int* loc2glob;
} CR_P_TYPE;

typedef struct {
	int offset, numelements;
        float portion;
} VIP_ENTRY_TYPE;

typedef struct {
        int items;
        VIP_ENTRY_TYPE* entry;
} VIP_TYPE;

typedef struct {
        int row, col, put;
        mat_data_t val;
} BLOCKENTRY_TYPE;
        
typedef struct {
        int offset, entries;
} TARGET_TYPE;
 
/* ########################################################################## */

typedef unsigned long long uint64;

void getMatrixPath(char *given, char *path);
int isMMfile(const char *filename);

void* allocateMemory( const size_t size, const char* desc );

void             zeroVector(VECTOR_TYPE *vec);
VECTOR_TYPE*     newVector( const int nRows );
HOSTVECTOR_TYPE* newHostVector( const int nRows, mat_data_t (*fp)(int));
void             swapVectors(VECTOR_TYPE *v1, VECTOR_TYPE *v2);
void             normalize( mat_data_t *vec, int nRows);


MM_TYPE* readMMFile( const char* filename );

CR_TYPE* convertMMToCRMatrix( const MM_TYPE* mm );
JD_TYPE* convertMMToJDMatrix( MM_TYPE* mm );

JD_RESORTED_TYPE* resort_JDS(const JD_TYPE*, const int);
VIP_TYPE* check_divide(MM_TYPE*, float);
JD_OFFDIAGONAL_TYPE* convertMMToODJDMatrix(const MM_TYPE*, const int);

void crColIdToFortran( CR_TYPE* cr );
void crColIdToC( CR_TYPE* cr );

void for_timing_start_asm_(uint64*);
void for_timing_stop_asm_(uint64*, uint64*);
void fortrancrs_(int*, int*, mat_data_t*, mat_data_t*, mat_data_t*, int*, int*);
void fortranjds_(int*, int*, int*, mat_data_t*, mat_data_t*, int*, mat_data_t*, int*, int*, int*);

void freeMMMatrix( MM_TYPE* const mm );
void freeJDMatrix( JD_TYPE* const cr );
void tmpwrite_d(int, int, mat_data_t*);
void tmpwrite_i(int, int, int*, char*);

void readCRbinFile(CR_TYPE*, const char* );
void readCRrowsBinFile(CR_TYPE* cr, const char* path);
void readJDbinFile(JD_TYPE*, const int, const char*);

void pio_write_cr_rownumbers(const CR_TYPE*, const char*);
void pio_read_cr_rownumbers(CR_TYPE*, const char*);

LCRP_TYPE* new_pio_read(char*, int);
LCRP_TYPE* parallel_MatRead(char*, int);
LCRP_TYPE* setup_communication_parRead(LMMP_TYPE* const);

void check_lcrp(int, LCRP_TYPE* const);
void pio_write_cr(const CR_TYPE*, const char*);
void myabort(char*);
void mypabort(char*);
void myaborti(char*, int);
void myabortf(char*, float);
void myaborttf(char*, int, float);
void mypaborts(const char*, const char*);
uint64 cycles4measurement, p_cycles4measurement;
double clockfreq;
double total_mem;
double RecalFrequency(uint64, double);
void freeRevBuf(REVBUF_TYPE*);

REVBUF_TYPE* revolvingBuffer(const uint64, const int, const int);
/* ########################################################################## */
float myCpuClockFrequency();
int Correctness_check( mat_data_t*, LCRP_TYPE*, mat_data_t* );
unsigned long thishost(char*); 
double my_amount_of_mem(void);
unsigned long machname(char* );
unsigned long kernelversion(char* );
unsigned long modelname(char* );
uint64 cachesize(void);

void freeMemory(size_t, const char*, void*);
int compareNZEOrgPos( const void* a, const void* b );
int compareNZEPos(const void*, const void*);
int compareNZEPerRow( const void*, const void*);
int compareNZEForJD( const void*, const void* );
BJDS_TYPE * CRStoBJDS(CR_TYPE *cr);
BJDS_TYPE * CRStoSBJDS(CR_TYPE *cr, int **rowPerm, int **invRowPerm); 
BJDS_TYPE * CRStoTBJDS(CR_TYPE *cr); 
int pad(int nRows, int padding);

#endif /* _MATRICKS_H_ */
