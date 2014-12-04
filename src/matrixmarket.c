#include <stdlib.h>
#include "ghost/util.h"
#include "ghost/mmio.h"
#include "ghost/matrixmarket.h"

int ghost_sparsemat_rowfunc_mm(ghost_gidx_t row, ghost_lidx_t *rowlen, ghost_gidx_t *col, void *val)
{

    static ghost_gidx_t *colInd = NULL, *rowPtr = NULL;
    static char *values = NULL;
    static size_t dtsize = 0;

    if (row == GHOST_SPARSEMAT_ROWFUNC_MM_ROW_INIT) {
        ghost_sparsemat_rowfunc_mm_initargs args = 
            *(ghost_sparsemat_rowfunc_mm_initargs *)val;
        char *filename = args.filename;
        ghost_datatype_t matdt = args.dt;

        ghost_datatype_size(&dtsize,matdt);

        FILE *f;
        int ret_code;
        MM_typecode matcode;
        int M, N, nz;
        ghost_gidx_t * offset;
        ghost_gidx_t i;

        if ((f = fopen(filename,"r")) == NULL) {
            ERROR_LOG("fopen failed!");
            return 1;
        }

        if (mm_read_banner(f, &matcode) != 0)
        {
            ERROR_LOG("Could not process Matrix Market banner!");
            return 1;
        }
        if ((mm_is_complex(matcode) && !(matdt & GHOST_DT_COMPLEX)) ||
                (!mm_is_complex(matcode) && (matdt & GHOST_DT_COMPLEX))) {
            ERROR_LOG("On-the-fly casting between real and complex not implemented!");
            return 1;
        }
        if (!mm_is_general(matcode)) {
            ERROR_LOG("Only general matrices supported at the moment!");
            return 1;
        }
        if (mm_is_pattern(matcode)) {
            ERROR_LOG("Pattern matrices not supported!");
            return 1;
        }

        if((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0){
            ERROR_LOG("Could not read header!");
            return 1;
        }
        if (M < 0 || N < 0 || nz < 0) {
            ERROR_LOG("Probably integer overflow");
            return 1;
        }


        ghost_malloc((void **)&colInd,nz * sizeof(ghost_gidx_t));
        ghost_malloc((void **)&rowPtr,(M + 1) * sizeof(ghost_gidx_t));
        ghost_malloc((void **)&values,nz * dtsize);
        ghost_malloc((void **)&offset,(M + 1) * sizeof(ghost_gidx_t));


        for(i = 0; i <= M; ++i){
            rowPtr[i] = 0;
            offset[i] = 0;
        }

        ghost_gidx_t row,col;
        char value[dtsize];
        fpos_t pos;
        fgetpos(f,&pos);

        for (i = 0; i < nz; ++i){
            if (matdt & GHOST_DT_COMPLEX) {
                fscanf(f, "%"PRGIDX" %"PRGIDX" %lg %lg\n", &row,&colInd[i],(double *)value,(double *)(value+dtsize/2));
            } else {
                fscanf(f, "%"PRGIDX" %"PRGIDX" %lg\n", &row,&colInd[i],(double *)value);
            }
            colInd[i]--;
            row--;
            rowPtr[row+1]++;
        }

        for(i = 1; i<=M; ++i){
            rowPtr[i] += rowPtr[i-1];
        }

        fsetpos(f,&pos);

        for (i = 0; i < nz; ++i){
            if (matdt & GHOST_DT_COMPLEX) {
                fscanf(f, "%"PRGIDX" %"PRGIDX" %lg %lg\n", &row,&col,(double *)value,(double *)(value+dtsize/2));
            } else {
                fscanf(f, "%"PRGIDX" %"PRGIDX" %lg\n", &row,&col,(double *)value);
            }
            row--;
            memcpy(&values[(rowPtr[row] + offset[row])*dtsize],value,dtsize);
            offset[row]++;
        }


        free(offset);
        fclose(f);

    } else if (row == GHOST_SPARSEMAT_ROWFUNC_MM_ROW_FINALIZE) {
        free(colInd);
        free(rowPtr);
        free(values);
    } else {
        *rowlen = rowPtr[row+1]-rowPtr[row];
        memcpy(col,&colInd[rowPtr[row]],(*rowlen)*sizeof(ghost_gidx_t));
        memcpy(val,&values[rowPtr[row]*dtsize],(*rowlen)*dtsize);
    }


    return 0;


}
