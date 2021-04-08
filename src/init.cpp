#include "ghost/init.h"
#include "ghost/constants.h" 
#include "ghost/util.h"
#include "ghost/config.h"
#ifdef GHOST_HAVE_RACE
#include <RACE/interface.h>
#endif



void chunkPtr(int start, int end, void *args_)
{
    ghost_sparsemat *args = (ghost_sparsemat*) args_;
    for(int i=start; i<end; ++i)
    {
        args->chunkStart[i] = 0;
    }
}

void rowLen(int start, int end, void *args_)
{
    ghost_sparsemat *args = (ghost_sparsemat*) args_;
    for(int i=start; i<end; ++i)
    {
        args->rowLen[i] = 0;
    }
}


void col(int start, int end, void *args_)
{
    ghost_sparsemat *args = (ghost_sparsemat*) args_;
    for(int i=start; i<end; ++i)
    {
        for(int j=args->chunkStart[i]; j<args->chunkStart[i+1]; j++) {
            args->col[j] = 0;
        }
    }
}


void val(int start, int end, void *args_)
{
    ghost_sparsemat *args = (ghost_sparsemat*) args_;
    for(int i=start; i<end; ++i)
    {
        for(int j=args->chunkStart[i]; j<args->chunkStart[i+1]; j++) {
            args->val[j] = 0.0;
        }
    }
}

extern "C" {
void initChunkPtr(ghost_sparsemat *mat)
{
#ifdef GHOST_HAVE_RACE
    printf("Initing chunkPtr\n");
    RACE::Interface *ce = (RACE::Interface*) (mat->context->coloringEngine);
    //NUMA INIT for coloring engine
    int fnId = ce->registerFunction(&chunkPtr, (void*)mat);
    ce->executeFunction(fnId);
    printf("Finished chunkPtr\n");
#endif
}

void initRowLen(ghost_sparsemat *mat)
{

#ifdef GHOST_HAVE_RACE
    printf("Initing rowLen\n");
    RACE::Interface *ce = (RACE::Interface*) (mat->context->coloringEngine);
    //NUMA INIT for coloring engine
    int fnId = ce->registerFunction(&rowLen, (void*)mat);
    ce->executeFunction(fnId);
    printf("Finished rowLen\n");
#endif
}


void initVal(ghost_sparsemat *mat)
{

#ifdef GHOST_HAVE_RACE
    printf("initing val\n");
    RACE::Interface *ce = (RACE::Interface*) (mat->context->coloringEngine);
    //NUMA INIT for coloring engine
    int fnId = ce->registerFunction(&val, (void*)mat);
    ce->executeFunction(fnId);
    printf("finished val\n");
#endif
}


void initCol(ghost_sparsemat *mat)
{
#ifdef GHOST_HAVE_RACE
    printf("initing col\n");
    RACE::Interface *ce = (RACE::Interface*) (mat->context->coloringEngine);
    //NUMA INIT for coloring engine
    int fnId = ce->registerFunction(&col, (void*)mat);
    ce->executeFunction(fnId);
    printf("finished col\n");
#endif
}

#define INIT_POLICY(RACE_policy, ...)\
    if(mat->context->coloringEngine)\
    {\
        printf("Initing RACE\n");\
        RACE_policy; \
    }\
    else if((mat->traits.flags & GHOST_SPARSEMAT_COLOR)&&(mat->context->color_ptr))\
    {\
        printf("Initing MC\n");\
        for(int color=0; color<mat->context->ncolors; ++color)\
        {\
_Pragma("omp parallel for schedule(static)")\
            for(int row=mat->context->color_ptr[color]; row<mat->context->color_ptr[color+1]; ++row)\
            {\
                __VA_ARGS__;\
            }\
        }\
    }\
    else if((mat->traits.flags & GHOST_SPARSEMAT_ABMC) && (mat->context->color_ptr))\
    {\
        printf("Initing ABMC\n");\
        for(int color=0; color<mat->context->ncolors; ++color)\
        {\
_Pragma("omp parallel for schedule(static)")\
            for(int part=mat->context->color_ptr[color]; part<mat->context->color_ptr[color+1]; ++part)\
            {\
                for(int row=mat->context->part_ptr[part]; row<mat->context->part_ptr[part+1]; ++row)\
                {\
                    __VA_ARGS__;\
                }\
            }\
        }\
    }\
    else\
    {\
        printf("Initing NORMAL\n");\
        _Pragma("omp parallel for schedule(static)")\
        for(int row=0; row<mat->context->row_map->dim; ++row)\
        {\
            __VA_ARGS__;\
        }\
    }\


//reinitialise for NUMA and CE compatibility
void reInit(ghost_sparsemat *mat)
{
    int *newChunkStart;
    ghost_malloc_align((void **)&newChunkStart,(mat->context->row_map->dim+1)*sizeof(ghost_lidx),GHOST_DATA_ALIGNMENT);
    int* oldChunkStart = mat->chunkStart;
    mat->chunkStart = newChunkStart;
    INIT_POLICY(initChunkPtr(mat), mat->chunkStart[row] = 0);

/*    if(mat->context->coloringEngine)
    {
        initChunkPtr(mat);
    }
    */
    int len = mat->context->row_map->dim+1;
#pragma omp parallel for schedule(static)
    for(int i=0; i<len; ++i)
    {
        mat->chunkStart[i] = oldChunkStart[i];
    }

 //   memcpy(mat->chunkStart, oldChunkStart, sizeof(int)*len);
    delete[] oldChunkStart;

    int *newRowLen;
    ghost_malloc_align((void **)&newRowLen,(mat->context->row_map->dim)*sizeof(ghost_lidx),GHOST_DATA_ALIGNMENT);
    int* oldRowLen = mat->rowLen;
    mat->rowLen = newRowLen;
    INIT_POLICY(initRowLen(mat), mat->rowLen[row]=0);

/*    if(mat->context->coloringEngine)
    {
        initRowLen(mat);
    }
    */

    len = mat->context->row_map->dim;
#pragma omp parallel for schedule(static)
    for(int i=0; i<len; ++i)
    {
        mat->rowLen[i] = oldRowLen[i];
    }

    // memcpy(mat->rowLen, oldRowLen, sizeof(int)*mat->context->row_map->dim);
    delete[] oldRowLen;


    char *newVal;
    ghost_malloc_align((void **)&newVal,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT);
    char* oldVal = mat->val;
    mat->val = newVal;

    INIT_POLICY(initVal(mat), for(int idx=mat->chunkStart[row]; idx<mat->chunkStart[row+1]; ++idx) {    ((double*)mat->val)[idx] = 0; } );

    /*

    if(mat->context->coloringEngine)
    {
        initVal(mat);
    }
    */

#pragma omp parallel for schedule(static)
    for(int i=0; i<mat->context->row_map->dim; ++i)
    {
        for(int idx=mat->chunkStart[i]; idx<mat->chunkStart[i+1]; ++idx)
        {
            ((double*)mat->val)[idx] = ((double*)oldVal)[idx];
        }
    }


    //memcpy(mat->val, oldVal, mat->elSize*(mat->nEnts));
    delete[] oldVal;

    int *newCol;
    ghost_malloc_align((void **)&newCol,sizeof(ghost_lidx)*mat->nEnts,GHOST_DATA_ALIGNMENT);
    int *oldCol = mat->col;
    mat->col = newCol;

    INIT_POLICY(initCol(mat), for(int idx=mat->chunkStart[row]; idx<mat->chunkStart[row+1]; ++idx)  { mat->col[idx] = 0; } );
/* 
    if(mat->context->coloringEngine)
    {
        initCol(mat);
    }
    */

#pragma omp parallel for schedule(static)
    for(int i=0; i<mat->context->row_map->dim; ++i)
    {
        for(int idx=mat->chunkStart[i]; idx<mat->chunkStart[i+1]; ++idx)
        {
            mat->col[idx] = oldCol[idx];
        }
    }

//    memcpy(mat->col, oldCol, sizeof(int)*(mat->nEnts));
    delete[] oldCol;
}

}
