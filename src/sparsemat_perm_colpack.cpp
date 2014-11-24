#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#ifdef GHOST_HAVE_COLPACK
#include "ColPack/ColPackHeaders.h"
#endif

extern "C" ghost_error_t ghost_sparsemat_perm_color(ghost_sparsemat_t *mat, void *matrixSource, ghost_sparsemat_src_t srcType)
{
#ifdef GHOST_HAVE_COLPACK
    INFO_LOG("Create permutation from coloring");
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_lidx_t *curcol = NULL;
    uint32_t** adolc = new uint32_t*[mat->nrows];
    std::vector<int>* colvec = NULL;
    uint32_t *adolc_data;
    ColPack::GraphColoring *GC=new ColPack::GraphColoring();

    int me, i, j;
    ghost_gidx_t *rpt = NULL;
    ghost_lidx_t *rptlocal = NULL;
    ghost_gidx_t *col = NULL;
    ghost_lidx_t *collocal = NULL;
    ghost_lidx_t nnz = 0, nnzlocal = 0;
    int64_t pos=0;

    GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(mat->context->lnrows[me]+1) * sizeof(ghost_gidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rptlocal,(mat->context->lnrows[me]+1) * sizeof(ghost_lidx_t)),err,ret);

        
    rpt[0] = 0;
    rptlocal[0] = 0;

    if (srcType == GHOST_SPARSEMAT_SRC_FILE) {
        char *matrixPath = (char *)matrixSource;
        GHOST_CALL_GOTO(ghost_bincrs_rpt_read(rpt, matrixPath, mat->context->lfRow[me], mat->context->lnrows[me]+1, NULL),err,ret);
#pragma omp parallel for
        for (i=1;i<mat->context->lnrows[me]+1;i++) {
            rpt[i] -= rpt[0];
        }
        rpt[0] = 0;
        nnz = rpt[mat->context->lnrows[me]];
        GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_lidx_t)),err,ret);
        GHOST_CALL_GOTO(ghost_bincrs_col_read(col, matrixPath, mat->context->lfRow[me], mat->context->lnrows[me], NULL,1),err,ret);
       
#pragma omp parallel for private(j) reduction(+:nnzlocal) 
        for (i=0;i<mat->context->lnrows[me]+1;i++) {
            for (j=rpt[i]; j<rpt[i+1]; j++) {
                if (col[j] >= mat->context->lfRow[me] && col[j] < mat->context->lfRow[me]+mat->context->lnrows[me]) {
                    nnzlocal++;
                }
            }
        }

    } else if (srcType == GHOST_SPARSEMAT_SRC_FUNC) {
        ghost_sparsemat_src_rowfunc_t *src = (ghost_sparsemat_src_rowfunc_t *)matrixSource;
        char * tmpval = NULL;
        ghost_gidx_t * tmpcol = NULL;

        ghost_lidx_t rowlen;

#pragma omp parallel private (tmpval,tmpcol,i,rowlen,j) reduction(+:nnz) reduction(+:nnzlocal)
        {
            ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
            ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t));
            
#pragma omp for
            for (i=0; i<mat->context->lnrows[me]; i++) {
                src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval);
                nnz += rowlen;
                for (j=0; j<rowlen; j++) {
                    if (tmpcol[j] >= mat->context->lfRow[me] && tmpcol[j] < mat->context->lnrows[me]) {
                        nnzlocal++;
                    }
                }
            }
            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_gidx_t)),err,ret);
        
#pragma omp parallel private (tmpval,tmpcol,i,rowlen) reduction(+:nnz)
        {
            ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
            ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx_t));
#pragma omp for ordered
            for (i=0; i<mat->context->lnrows[me]; i++) {
#pragma omp ordered
                {
                    src->func(mat->context->lfRow[me]+i,&rowlen,&col[rpt[i]],tmpval);
                    rpt[i+1] = rpt[i] + rowlen;
                }
            }
            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
    }
        
    GHOST_CALL_GOTO(ghost_malloc((void **)&collocal,nnzlocal * sizeof(ghost_lidx_t)),err,ret);

    for (i=0; i<mat->context->lnrows[me]; i++) {
        rptlocal[i+1] = rptlocal[i];
        for (j=rpt[i]; j<rpt[i+1]; j++) {
            if (col[j] >= mat->context->lfRow[me] && col[j] < mat->context->lfRow[me]+mat->context->lnrows[me]) {
                collocal[rptlocal[i+1]] = col[j] - mat->context->lfRow[me];
                rptlocal[i+1]++;
            }
        }
    }

    adolc_data = new uint32_t[nnzlocal+mat->nrows];

    for (int i=0;i<mat->nrows;i++)
    {
        adolc[i]=&(adolc_data[pos]);
        adolc_data[pos++]=rptlocal[i+1]-rptlocal[i];
        for (int j=rptlocal[i];j<rptlocal[i+1];j++)
        {
            adolc_data[pos++]=collocal[j];
        }
    }

    GC->BuildGraphFromRowCompressedFormat(adolc, mat->nrows);

    COLPACK_CALL_GOTO(GC->DistanceTwoColoring(),err,ret);

    if (GC->CheckDistanceTwoColoring(2)) {
        ERROR_LOG("Error in coloring!");
        ret = GHOST_ERR_COLPACK;
        goto err;
    }

    mat->ncolors = GC->GetVertexColorCount();
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation,sizeof(ghost_permutation_t)),err,ret);
    mat->context->permutation->scope = GHOST_PERMUTATION_LOCAL;
    mat->context->permutation->len = mat->nrows;
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation->perm,sizeof(ghost_gidx_t)*mat->nrows),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->permutation->invPerm,sizeof(ghost_gidx_t)*mat->nrows),err,ret);

#ifdef GHOST_HAVE_CUDA
    GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->context->permutation->cu_perm,sizeof(ghost_gidx_t)*mat->nrows),err,ret);
#endif

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->color_ptr,(mat->ncolors+1)*sizeof(ghost_lidx_t)),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(mat->ncolors)*sizeof(ghost_lidx_t)),err,ret);
    memset(curcol,0,mat->ncolors*sizeof(ghost_lidx_t));
    
    colvec = GC->GetVertexColorsPtr();

    for (int i=0;i<mat->ncolors+1;i++) {
        mat->color_ptr[i] = 0;
    }

    for (int i=0;i<mat->nrows;i++) {
        mat->color_ptr[(*colvec)[i]+1]++;
    }

    for (int i=1;i<mat->ncolors+1;i++) {
        mat->color_ptr[i] += mat->color_ptr[i-1];
    }
    
    for (int i=0;i<mat->nrows;i++) {
        mat->context->permutation->perm[i] = curcol[(*colvec)[i]] + mat->color_ptr[(*colvec)[i]];
        curcol[(*colvec)[i]]++;
    }
    
    for (int i=0;i<mat->nrows;i++) {
        mat->context->permutation->invPerm[mat->context->permutation->perm[i]] = i;
    }


    goto out;
err:

out:
    delete [] adolc_data;
    delete [] adolc;
    delete GC;
    free(curcol);
    free(rpt);
    free(col);
    free(rptlocal);
    free(collocal);

    return ret;
#else
    UNUSED(mat);
    UNUSED(matrixSource);
    UNUSED(srcType);
    ERROR_LOG("ColPack not available!");
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}

