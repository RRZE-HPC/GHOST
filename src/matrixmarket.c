#include <stdlib.h>
#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/mmio.h"
#include "ghost/matrixmarket.h"
#include "ghost/locality.h"

int ghost_sparsemat_rowfunc_mm(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    static ghost_gidx *colInd = NULL, *rowPtr = NULL;
    static char *values = NULL;
    static size_t dtsize = 0;
    bool isPattern = false;

    if (row == GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETDIM) {
        ghost_sparsemat_rowfunc_file_initargs args =
            *(ghost_sparsemat_rowfunc_file_initargs *)arg;
        char *filename = args.filename;

        FILE *f;
        int ret_code;
        int M, N, nz;
        MM_typecode matcode;

        if ((f = fopen(filename,"r")) == NULL) {
            GHOST_ERROR_LOG("fopen with %s failed!",filename);
            return 1;
        }

        if (mm_read_banner(f, &matcode) != 0){
            GHOST_ERROR_LOG("Could not process Matrix Market banner!");
            return 1;
        }

        if (rowlen){
            if (mm_is_complex(matcode)) *rowlen = (ghost_lidx)GHOST_DT_COMPLEX;
            else *rowlen = (ghost_lidx)GHOST_DT_REAL;
        }

        if((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0){
            GHOST_ERROR_LOG("Could not read header!");
            return 1;
        }
        col[0] = M;
        col[1] = N;

        fclose(f);
    } else if (row == GHOST_SPARSEMAT_ROWFUNC_INIT) {

        ghost_sparsemat_rowfunc_file_initargs args =
            *(ghost_sparsemat_rowfunc_file_initargs *)arg;
        char *filename = args.filename;
        ghost_datatype matdt = args.dt;

        ghost_datatype_size(&dtsize,matdt);

        FILE *f;
        int ret_code;
        MM_typecode matcode;
        int M, N, nz, actualnz;
        ghost_gidx * offset;
        ghost_gidx i;
        int symm = 0;

        if ((f = fopen(filename,"r")) == NULL) {
            GHOST_ERROR_LOG("fopen with %s failed!",filename);
            return 1;
        }

        if (mm_read_banner(f, &matcode) != 0)
        {
            GHOST_ERROR_LOG("Could not process Matrix Market banner!");
            return 1;
        }
        if ((mm_is_complex(matcode) && !(matdt & GHOST_DT_COMPLEX)) ||
                (!mm_is_complex(matcode) && (matdt & GHOST_DT_COMPLEX))) {
            GHOST_ERROR_LOG("On-the-fly casting between real and complex not implemented!");
            return 1;
        }
        if (!mm_is_general(matcode) && !mm_is_symmetric(matcode)) {
            GHOST_ERROR_LOG("Only general and symmetric matrices supported at the moment!");
            return 1;
        }
        //supporting pattern matrices
        if (mm_is_pattern(matcode)) {
            GHOST_WARNING_LOG("Will fill 1.0(double) as the value of pattern matrices");
            /*     GHOST_ERROR_LOG("Pattern matrices not supported!");
                   return 1;*/
            isPattern = true;
        }

        if((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0){
            GHOST_ERROR_LOG("Could not read header!");
            return 1;
        }
        if (M < 0 || N < 0 || nz < 0) {
            GHOST_ERROR_LOG("Probably integer overflow");
            return 1;
        }

        if (mm_is_symmetric(matcode)) {
            GHOST_PERFWARNING_LOG("Will create a general matrix out of a symmetric matrix!");
            args.mat->traits.symmetry = GHOST_SPARSEMAT_SYMM_SYMMETRIC;
            actualnz = nz*2;
            symm = 1;
        } else {
            actualnz = nz;
            symm = 0;
        }

        //printf("allocating total size of %f, where gidx = %d\n", (double)actualnz*sizeof(ghost_gidx)+2*(M + 1)*sizeof(ghost_gidx)+(double)actualnz*dtsize, sizeof(ghost_gidx));
        ghost_malloc((void **)&colInd,actualnz * sizeof(ghost_gidx));
        ghost_malloc((void **)&rowPtr,(M + 1) * sizeof(ghost_gidx));
        ghost_malloc((void **)&values,actualnz * dtsize);
        ghost_malloc((void **)&offset,(M + 1) * sizeof(ghost_gidx));
        printf("allocated\n");

        for(i = 0; i <= M; ++i){
            rowPtr[i] = 0;
            offset[i] = 0;
        }

        int toread = (!isPattern)?(3*nz):(2*nz);
        if (matdt & GHOST_DT_COMPLEX) {
            toread += nz;
        }
        ghost_gidx readrow,readcol;
        char value[dtsize];
        fpos_t pos;
        fgetpos(f,&pos);

        int scanned = 0;
        for (i = 0; i < nz; ++i){
            if(!isPattern) {
                if (matdt & GHOST_DT_COMPLEX) {
                    if (matdt & GHOST_DT_DOUBLE) {
                        scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %lg %lg\n", &readrow,&readcol,(double *)value,(double *)(value+dtsize/2));
                    } else {
                        scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %g %g\n", &readrow,&readcol,(float *)value,(float *)(value+dtsize/2));
                    }
                } else {
                    if (matdt & GHOST_DT_DOUBLE) {
                        scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %lg\n", &readrow,&readcol,(double *)value);
                    } else {
                        scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %g\n", &readrow,&readcol,(float *)value);
                    }
                }
            } else {
                scanned += fscanf(f, "%"PRGIDX" %"PRGIDX"\n", &readrow,&readcol);
            }

            readcol--;
            readrow--;

            rowPtr[readrow+1]++;

            if (symm) {
                if (readrow != readcol) {
                    rowPtr[readcol+1]++; // insert sibling entry
                } else {
                    actualnz--; // do not count diagonal entries twice
                }
            }
        }

        if (scanned != toread) {
            GHOST_ERROR_LOG("Error while reading filei: read %d items but was expecting %d!",scanned,toread);
            return 1;
        }

        for(i = 1; i<=M; ++i){
            rowPtr[i] += rowPtr[i-1];
        }

        if (row == GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETRPT) {
            col = rowPtr;
        } else {

            fsetpos(f,&pos);

            scanned = 0;
            for (i = 0; i < nz; ++i){
                if(!isPattern)
                {
                    if (matdt & GHOST_DT_COMPLEX) {
                        if (matdt & GHOST_DT_DOUBLE) {
                            scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %lg %lg\n", &readrow,&readcol,(double *)value,(double *)(value+dtsize/2));
                        } else {
                            scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %g %g\n", &readrow,&readcol,(float *)value,(float *)(value+dtsize/2));
                        }
                    } else {
                        if (matdt & GHOST_DT_DOUBLE) {
                            scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %lg\n", &readrow,&readcol,(double *)value);
                        } else {
                            scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %g\n", &readrow,&readcol,(float *)value);
                        }
                    }
                }
                else
                {
                    scanned += fscanf(f, "%"PRGIDX" %"PRGIDX"\n", &readrow,&readcol);
                }
                readrow--;
                readcol--;

                if(!isPattern)
                {
                    memcpy(&values[(rowPtr[readrow] + offset[readrow])*dtsize],value,dtsize);
                }
                else
                {
                  //  values[rowPtr[readrow] + offset[readrow]] = 1.0;
                    memset(&values[(rowPtr[readrow] + offset[readrow])*dtsize],1.0,dtsize);
                }

                colInd[rowPtr[readrow] + offset[readrow]] = readcol;
                offset[readrow]++;

                if (symm && (readrow != readcol)) {
                    memcpy(&values[(rowPtr[readcol] + offset[readcol])*dtsize],value,dtsize);
                    colInd[rowPtr[readcol] + offset[readcol]] = readrow;
                    offset[readcol]++;
                }

            }
            if (scanned != toread) {
                GHOST_ERROR_LOG("Error while reading file: read %d items but was expecting %d!",scanned,toread);
                return 1;
            }
        }

        free(offset);
        fclose(f);

    } else if (row == GHOST_SPARSEMAT_ROWFUNC_FINALIZE) {
        free(colInd);
        free(rowPtr);
        free(values);
    } else {
        *rowlen = rowPtr[row+1]-rowPtr[row];
        memcpy(col,&colInd[rowPtr[row]],(*rowlen)*sizeof(ghost_gidx));
        memcpy(val,&values[rowPtr[row]*dtsize],(*rowlen)*dtsize);
    }


    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return 0;


}

int ghost_sparsemat_rowfunc_mm_transpose(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    static ghost_gidx *colInd = NULL, *rowPtr = NULL;
    static char *values = NULL;
    static size_t dtsize = 0;

    if (row == GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETDIM) {
       ghost_sparsemat_rowfunc_file_initargs args =
            *(ghost_sparsemat_rowfunc_file_initargs *)arg;

        char *filename = args.filename;

        FILE *f;
        int ret_code;
        int M, N, nz;
        MM_typecode matcode;

        if ((f = fopen(filename,"r")) == NULL) {
            GHOST_ERROR_LOG("fopen with %s failed!",filename);
            return 1;
        }

        if (mm_read_banner(f, &matcode) != 0){
            GHOST_ERROR_LOG("Could not process Matrix Market banner!");
            return 1;
        }

        if (rowlen){
            if (mm_is_complex(matcode)) *rowlen = (ghost_lidx)GHOST_DT_COMPLEX;
            else *rowlen = (ghost_lidx)GHOST_DT_REAL;
        }

        if((ret_code = mm_read_mtx_crd_size(f, &N, &M, &nz)) != 0){
            GHOST_ERROR_LOG("Could not read header!");
            return 1;
        }
        col[0] = M;
        col[1] = N;

        fclose(f);
    } else if (row == GHOST_SPARSEMAT_ROWFUNC_INIT) {

       ghost_sparsemat_rowfunc_file_initargs args =
            *(ghost_sparsemat_rowfunc_file_initargs *)arg;

        char *filename = args.filename;
        ghost_datatype matdt = args.dt;

        ghost_datatype_size(&dtsize,matdt);

        FILE *f;
        int ret_code;
        MM_typecode matcode;
        int M, N, nz, actualnz;
        ghost_gidx * offset;
        ghost_gidx i;
        int symm = 0;

        if ((f = fopen(filename,"r")) == NULL) {
            GHOST_ERROR_LOG("fopen with %s failed!",filename);
            return 1;
        }

        if (mm_read_banner(f, &matcode) != 0)
        {
            GHOST_ERROR_LOG("Could not process Matrix Market banner!");
            return 1;
        }
        if ((mm_is_complex(matcode) && !(matdt & GHOST_DT_COMPLEX)) ||
                (!mm_is_complex(matcode) && (matdt & GHOST_DT_COMPLEX))) {
            GHOST_ERROR_LOG("On-the-fly casting between real and complex not implemented!");
            return 1;
        }
        if (!mm_is_general(matcode) && !mm_is_symmetric(matcode)) {
            GHOST_ERROR_LOG("Only general and symmetric matrices supported at the moment!");
            return 1;
        }
        if (mm_is_pattern(matcode)) {
            GHOST_ERROR_LOG("Pattern matrices not supported!");
            return 1;
        }

        if((ret_code = mm_read_mtx_crd_size(f, &N, &M, &nz)) != 0){
            GHOST_ERROR_LOG("Could not read header!");
            return 1;
        }
        if (M < 0 || N < 0 || nz < 0) {
            GHOST_ERROR_LOG("Probably integer overflow");
            return 1;
        }

        if (mm_is_symmetric(matcode)) {
            GHOST_PERFWARNING_LOG("Will create a general matrix out of a symmetric matrix!");
            args.mat->traits.symmetry = GHOST_SPARSEMAT_SYMM_SYMMETRIC;
            actualnz = nz*2;
            symm = 1;
        } else {
            actualnz = nz;
            symm = 0;
        }


        ghost_malloc((void **)&colInd,actualnz * sizeof(ghost_gidx));
        ghost_malloc((void **)&rowPtr,(M + 1) * sizeof(ghost_gidx));
        ghost_malloc((void **)&values,actualnz * dtsize);
        ghost_malloc((void **)&offset,(M + 1) * sizeof(ghost_gidx));

        for(i = 0; i <= M; ++i){
            rowPtr[i] = 0;
            offset[i] = 0;
        }

        int toread = 3*nz;
        if (matdt & GHOST_DT_COMPLEX) {
            toread += nz;
        }
        ghost_gidx readrow,readcol;
        char value[dtsize];
        fpos_t pos;
        fgetpos(f,&pos);
        int scanned = 0;

        for (i = 0; i < nz; ++i){
            if (matdt & GHOST_DT_COMPLEX) {
                if (matdt & GHOST_DT_DOUBLE) {
                    scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %lg %lg\n", &readcol,&readrow,(double *)value,(double *)(value+dtsize/2));
                } else {
                    scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %g %g\n", &readcol,&readrow,(float *)value,(float *)(value+dtsize/2));
                }
            } else {
                if (matdt & GHOST_DT_DOUBLE) {
                    scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %lg\n", &readcol,&readrow,(double *)value);
                } else {
                    scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %g\n", &readcol,&readrow,(float *)value);
                }
            }
            readcol--;
            readrow--;
            rowPtr[readrow+1]++;

            if (symm) {
                if (readrow != readcol) {
                    rowPtr[readcol+1]++; // insert sibling entry
                } else {
                    actualnz--; // do not count diagonal entries twice
                }
            }
        }

        if (scanned != toread) {
            GHOST_ERROR_LOG("Error while reading filei: read %d items but was expecting %d!",scanned,toread);
            return 1;
        }

        for(i = 1; i<=M; ++i){
            rowPtr[i] += rowPtr[i-1];
        }

        if (row == GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETRPT) {
            col = rowPtr;
        } else {

            fsetpos(f,&pos);

            scanned = 0;
            for (i = 0; i < nz; ++i){
                if (matdt & GHOST_DT_COMPLEX) {
                    if (matdt & GHOST_DT_DOUBLE) {
                        scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %lg %lg\n", &readcol,&readrow,(double *)value,(double *)(value+dtsize/2));
                    } else {
                        scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %g %g\n", &readcol,&readrow,(float *)value,(float *)(value+dtsize/2));
                    }
                } else {
                    if (matdt & GHOST_DT_DOUBLE) {
                        scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %lg\n", &readcol,&readrow,(double *)value);
                    } else {
                        scanned += fscanf(f, "%"PRGIDX" %"PRGIDX" %g\n", &readcol,&readrow,(float *)value);
                    }
                }
                readrow--;
                readcol--;

                memcpy(&values[(rowPtr[readrow] + offset[readrow])*dtsize],value,dtsize);
                colInd[rowPtr[readrow] + offset[readrow]] = readcol;
                offset[readrow]++;

                if (symm && (readrow != readcol)) {
                    memcpy(&values[(rowPtr[readcol] + offset[readcol])*dtsize],value,dtsize);
                    colInd[rowPtr[readcol] + offset[readcol]] = readrow;
                    offset[readcol]++;
                }

            }
            if (scanned != toread) {
                GHOST_ERROR_LOG("Error while reading filei: read %d items but was expecting %d!",scanned,toread);
                return 1;
            }
        }


        free(offset);
        fclose(f);

    } else if (row == GHOST_SPARSEMAT_ROWFUNC_FINALIZE) {
        free(colInd);
        free(rowPtr);
        free(values);
    } else {
        *rowlen = rowPtr[row+1]-rowPtr[row];
        memcpy(col,&colInd[rowPtr[row]],(*rowlen)*sizeof(ghost_gidx));
        memcpy(val,&values[rowPtr[row]*dtsize],(*rowlen)*dtsize);
    }


    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return 0;


}

ghost_error ghost_sparsemat_to_mm(char *path, ghost_sparsemat *mat)
{
    MM_typecode matcode;
    ghost_lidx row,entinrow,globrow;
    ghost_lidx sellidx;
    int nrank,rank;
    FILE *fp;

    if (mat->context->row_map->gdim > INT_MAX) {
        GHOST_ERROR_LOG("The number of matrix rows exceeds INT_MAX and I cannot write a MatrixMarket file!");
        return GHOST_ERR_INVALID_ARG;
    }

    if (mat->context->col_map->gdim > INT_MAX) {
        GHOST_ERROR_LOG("The number of matrix columns exceeds INT_MAX and I cannot write a MatrixMarket file!");
        return GHOST_ERR_INVALID_ARG;
    }

    if (mat->context->gnnz > INT_MAX) {
        GHOST_ERROR_LOG("The number of matrix entries exceeds INT_MAX and I cannot write a MatrixMarket file!");
        return GHOST_ERR_INVALID_ARG;
    }

    ghost_nrank(&nrank,mat->context->mpicomm);
    ghost_rank(&rank,mat->context->mpicomm);

    if (nrank > 1 && !(mat->traits.flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
        GHOST_WARNING_LOG("The matrix is distributed and the non-compressed columns are not saved. The output will probably be useless!");
    }

    if (rank == 0) {
        fp = fopen(path,"w");
        if (!fp) {
            GHOST_ERROR_LOG("Unable to open file %s!",path);
            return GHOST_ERR_INVALID_ARG;
        }
        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_coordinate(&matcode);
        if (mat->traits.datatype & GHOST_DT_REAL) {
            mm_set_real(&matcode);
        } else {
            mm_set_complex(&matcode);
        }

        mm_write_banner(fp,matcode);
        mm_write_mtx_crd_size(fp,(int)mat->context->row_map->gdim,(int)mat->context->col_map->gdim,(int)mat->context->gnnz);

        fclose(fp);
    }

    int i;
    for (i=0; i<nrank; i++) {
#ifdef GHOST_HAVE_MPI
        MPI_Barrier(mat->context->mpicomm);
#endif
        if (i == rank) {
            fp = fopen(path,"a");
            if (!fp) {
                GHOST_ERROR_LOG("Unable to open file %s!",path);
                return GHOST_ERR_INVALID_ARG;
            }

            globrow = mat->context->row_map->goffs[rank]+1;

            for (row=1; row<=SPM_NROWS(mat); row++, globrow++) {
                for (entinrow=0; entinrow<mat->rowLen[row-1]; entinrow++) {
                    sellidx = mat->chunkStart[(row-1)/mat->traits.C] + entinrow*mat->traits.C + (row-1)%mat->traits.C;
                    if (mat->traits.flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS) {
                        int col = (int)mat->col_orig[sellidx]+1;
                        if (mat->traits.datatype & GHOST_DT_REAL) {
                            if (mat->traits.datatype & GHOST_DT_DOUBLE) {
                                fprintf(fp,"%d %d %10.3g\n",globrow,col,((double *)mat->val)[sellidx]);
                            } else {
                                fprintf(fp,"%d %d %10.3g\n",globrow,col,((float *)mat->val)[sellidx]);
                            }
                        } else {
                            if (mat->traits.datatype & GHOST_DT_DOUBLE) {
                                fprintf(fp,"%d %d %10.3g %10.3g\n",globrow,col,creal(((complex double *)mat->val)[sellidx]),cimag(((complex double *)mat->val)[sellidx]));
                            } else {
                                fprintf(fp,"%d %d %10.3g %10.3g\n",globrow,col,crealf(((complex float *)mat->val)[sellidx]),cimagf(((complex float *)mat->val)[sellidx]));
                            }
                        }
                    } else {
                        ghost_lidx col = mat->col[sellidx]+1;
                        if (mat->traits.datatype & GHOST_DT_REAL) {
                            if (mat->traits.datatype & GHOST_DT_DOUBLE) {
                                fprintf(fp,"%d %d %10.3g\n",globrow,col,((double *)mat->val)[sellidx]);
                            } else {
                                fprintf(fp,"%d %d %10.3g\n",globrow,col,((float *)mat->val)[sellidx]);
                            }
                        } else {
                            if (mat->traits.datatype & GHOST_DT_DOUBLE) {
                                fprintf(fp,"%d %d %10.3g %10.3g\n",globrow,col,creal(((complex double *)mat->val)[sellidx]),cimag(((complex double *)mat->val)[sellidx]));
                            } else {
                                fprintf(fp,"%d %d %10.3g %10.3g\n",globrow,col,crealf(((complex float *)mat->val)[sellidx]),cimagf(((complex float *)mat->val)[sellidx]));
                            }
                        }
                    }

                }
            }
            fclose(fp);
        }
    }

    return GHOST_SUCCESS;
}
