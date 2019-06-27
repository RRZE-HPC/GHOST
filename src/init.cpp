#ifdef GHOST_HAVE_NAME
#include <NAME/interface.h>
#endif
#include "ghost/init.h"
#include "ghost/constants.h" 
#include "ghost/util.h"

#ifdef GHOST_HAVE_NAME
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
    printf("Initing chunkPtr\n");
    NAMEInterface *ce = (NAMEInterface*) (mat->context->coloringEngine);
    //NUMA INIT for coloring engine
    int fnId = ce->registerFunction(&chunkPtr, (void*)mat);
    ce->executeFunction(fnId);
    printf("Finished chunkPtr\n");
}

void initRowLen(ghost_sparsemat *mat)
{
    printf("Initing rowLen\n");
    NAMEInterface *ce = (NAMEInterface*) (mat->context->coloringEngine);
    //NUMA INIT for coloring engine
    int fnId = ce->registerFunction(&rowLen, (void*)mat);
    ce->executeFunction(fnId);
    printf("Finished rowLen\n");
}


void initVal(ghost_sparsemat *mat)
{
    printf("initing val\n");
    NAMEInterface *ce = (NAMEInterface*) (mat->context->coloringEngine);
    //NUMA INIT for coloring engine
    int fnId = ce->registerFunction(&val, (void*)mat);
    ce->executeFunction(fnId);
    printf("finished val\n");
}


void initCol(ghost_sparsemat *mat)
{
    printf("initing col\n");
    NAMEInterface *ce = (NAMEInterface*) (mat->context->coloringEngine);
    //NUMA INIT for coloring engine
    int fnId = ce->registerFunction(&col, (void*)mat);
    ce->executeFunction(fnId);
    printf("finished col\n");
}

}
#endif

extern "C" {
//reinitialise for NUMA and CE compatibility
void reInit(ghost_sparsemat *mat)
{
    int *newChunkStart;
    ghost_malloc_align((void **)&newChunkStart,(mat->context->row_map->dim+1)*sizeof(ghost_lidx),GHOST_DATA_ALIGNMENT);
    int* oldChunkStart = mat->chunkStart;
    mat->chunkStart = newChunkStart;
#ifdef GHOST_HAVE_NAME
    initChunkPtr(mat);
#endif
    int len = mat->context->row_map->dim+1;
#pragma omp parallel for
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
#ifdef GHOST_HAVE_NAME
    initRowLen(mat);
#endif
    len = mat->context->row_map->dim;
#pragma omp parallel for
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
#ifdef GHOST_HAVE_NAME
    initVal(mat);
#endif
    len = mat->nEnts*mat->elSize;
#pragma omp parallel for
    for(int i=0; i<len; ++i)
    {
        mat->val[i] = oldVal[i];
    }

    //memcpy(mat->val, oldVal, mat->elSize*(mat->nEnts));
    delete[] oldVal;

    int *newCol;
    ghost_malloc_align((void **)&newCol,sizeof(ghost_lidx)*mat->nEnts,GHOST_DATA_ALIGNMENT);
    int *oldCol = mat->col;
    mat->col = newCol;
#ifdef GHOST_HAVE_NAME
    initCol(mat);
#endif
    len = mat->nEnts;
#pragma omp parallel for
    for(int i=0; i<len; ++i)
    {
        mat->col[i] = oldCol[i];
    }

//    memcpy(mat->col, oldCol, sizeof(int)*(mat->nEnts));
    delete[] oldCol;
}

}
