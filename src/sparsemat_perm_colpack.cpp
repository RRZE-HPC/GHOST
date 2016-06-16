#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"
#ifdef GHOST_HAVE_COLPACK
#include "ColPack/ColPackHeaders.h"
#endif

extern "C" ghost_error ghost_sparsemat_perm_color(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType)
{
    UNUSED(srcType);
#ifdef GHOST_HAVE_COLPACK
    INFO_LOG("Create permutation from coloring");
    ghost_error ret = GHOST_SUCCESS;
    ghost_lidx *curcol = NULL;
    uint32_t** adolc = new uint32_t*[mat->nrows];
    std::vector<int>* colvec = NULL;
    uint32_t *adolc_data = NULL;
    ColPack::GraphColoring *GC=new ColPack::GraphColoring();
    ghost_permutation *oldperm = NULL;

    int me, i, j;
    ghost_gidx *rpt = NULL;
    ghost_lidx *rptlocal = NULL;
    ghost_gidx *col = NULL;
    ghost_lidx *collocal = NULL;
    ghost_lidx nnz = 0, nnzlocal = 0;
    int64_t pos=0;
    
    ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
    char * tmpval = NULL;
    ghost_gidx * tmpcol = NULL;

    GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,(mat->context->lnrows[me]+1) * sizeof(ghost_gidx)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&rptlocal,(mat->context->lnrows[me]+1) * sizeof(ghost_lidx)),err,ret);

        
    rpt[0] = 0;
    rptlocal[0] = 0;
    

    ghost_lidx rowlen;

#pragma omp parallel private (tmpval,tmpcol,i,rowlen,j) reduction(+:nnz) reduction(+:nnzlocal)
    {
        ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
        ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
        
#pragma omp for
        for (i=0; i<mat->context->lnrows[me]; i++) {
            if (mat->context->perm_global && mat->context->perm_local) {
                src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,tmpcol,tmpval,NULL);
            } else if (mat->context->perm_global) {
                src->func(mat->context->perm_global->invPerm[i],&rowlen,tmpcol,tmpval,NULL);
            } else if (mat->context->perm_local) {
                src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[i],&rowlen,tmpcol,tmpval,NULL);
            } else {
                src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,NULL);
            }
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
    GHOST_CALL_GOTO(ghost_malloc((void **)&col,nnz * sizeof(ghost_gidx)),err,ret);
    
#pragma omp parallel private (tmpval,tmpcol,i,rowlen) reduction(+:nnz)
    {
        ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
        ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
#pragma omp for ordered
        for (i=0; i<mat->context->lnrows[me]; i++) {
#pragma omp ordered
            {
                if (mat->context->perm_global && mat->context->perm_local) {
                    src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,&col[rpt[i]],tmpval,NULL);
                } else if (mat->context->perm_global) {
                    src->func(mat->context->perm_global->invPerm[i],&rowlen,&col[rpt[i]],tmpval,NULL);
                } else if (mat->context->perm_local) {
                    src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[i],&rowlen,&col[rpt[i]],tmpval,NULL);
                } else {
                    src->func(mat->context->lfRow[me]+i,&rowlen,&col[rpt[i]],tmpval,NULL);
                }
                rpt[i+1] = rpt[i] + rowlen;
            }
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
        
    GHOST_CALL_GOTO(ghost_malloc((void **)&collocal,nnzlocal * sizeof(ghost_lidx)),err,ret);
 
    for (i=0; i<mat->context->lnrows[me]; i++) {  
        rptlocal[i+1] = rptlocal[i];
        for (j=rpt[i]; j<rpt[i+1]; j++) {

            if (!mat->context->perm_local) {
                if (col[j] >= mat->context->lfRow[me] && col[j] < mat->context->lfRow[me]+mat->context->lnrows[me]) {
                    collocal[rptlocal[i+1]] = col[j] - mat->context->lfRow[me];
                    rptlocal[i+1]++;
                }
            } else {
                if (col[j] >= mat->context->lfRow[me] && col[j] < mat->context->lfRow[me]+mat->context->lnrows[me]) {
                    collocal[rptlocal[i+1]] = mat->context->perm_local->perm[col[j] - mat->context->lfRow[me]];
                    rptlocal[i+1]++;
                }
            }
        }
    }

    adolc_data = new uint32_t[nnzlocal+mat->nrows];

    for (i=0;i<mat->nrows;i++)
    {   
        adolc[i]=&(adolc_data[pos]);
        adolc_data[pos++]=rptlocal[i+1]-rptlocal[i];
        for (j=rptlocal[i];j<rptlocal[i+1];j++)
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
    

    if (!mat->context->perm_local) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local,sizeof(ghost_permutation)),err,ret);
        mat->context->perm_local->scope = GHOST_PERMUTATION_LOCAL;
        mat->context->perm_local->len = mat->nrows;
        mat->context->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC; //you can also make it symmetric
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->perm,sizeof(ghost_gidx)*mat->nrows),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->invPerm,sizeof(ghost_gidx)*mat->nrows),err,ret);   
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colPerm,sizeof(ghost_gidx)*mat->ncols),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colInvPerm,sizeof(ghost_gidx)*mat->ncols),err,ret);

        for(int i=0; i<mat->ncols; ++i) {
                mat->context->perm_local->colPerm[i] = i;
                mat->context->perm_local->colInvPerm[i] = i;
        }

#ifdef GHOST_HAVE_CUDA
        GHOST_CALL_GOTO(ghost_cu_malloc((void **)&mat->context->perm_local->cu_perm,sizeof(ghost_gidx)*mat->nrows),err,ret);
#endif

    } else if(mat->context->perm_local->method == GHOST_PERMUTATION_SYMMETRIC) {
        oldperm = mat->context->perm_local;
	mat->context->perm_local->method = GHOST_PERMUTATION_UNSYMMETRIC;//change to unsymmetric
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colPerm,sizeof(ghost_gidx)*mat->ncols),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_local->colInvPerm,sizeof(ghost_gidx)*mat->ncols),err,ret);

        for(int i=0; i<mat->ncols; ++i) {
                mat->context->perm_local->colPerm[i] = mat->context->perm_local->perm[i];
                mat->context->perm_local->colInvPerm[i] = mat->context->perm_local->invPerm[i];
        }        
    } else {
        oldperm = mat->context->perm_local;
    }

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->color_ptr,(mat->ncolors+1)*sizeof(ghost_lidx)),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(mat->ncolors)*sizeof(ghost_lidx)),err,ret);
    memset(curcol,0,mat->ncolors*sizeof(ghost_lidx));
    
    colvec = GC->GetVertexColorsPtr();


    for (i=0;i<mat->ncolors+1;i++) {
        mat->color_ptr[i] = 0;
    }

    for (i=0;i<mat->nrows;i++) {
        mat->color_ptr[(*colvec)[i]+1]++;
    }

    for (i=1;i<mat->ncolors+1;i++) {
        mat->color_ptr[i] += mat->color_ptr[i-1];
    }
    
    if (oldperm) {
        for (i=0;i<mat->nrows;i++) {
            int idx = mat->context->perm_local->invPerm[i];
            mat->context->perm_local->perm[idx]  = curcol[(*colvec)[i]] + mat->color_ptr[(*colvec)[i]];
            //mat->context->perm_local->perm[i] = mat->context->perm_local->invPerm[curcol[(*colvec)[i]] + mat->color_ptr[(*colvec)[i]]];
            curcol[(*colvec)[i]]++;
        }
    } else {
        for (i=0;i<mat->nrows;i++) {
            mat->context->perm_local->perm[i] = curcol[(*colvec)[i]] + mat->color_ptr[(*colvec)[i]];
            curcol[(*colvec)[i]]++;
        }
    }
    for (i=0;i<mat->nrows;i++) {
        mat->context->perm_local->invPerm[mat->context->perm_local->perm[i]] = i;
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

