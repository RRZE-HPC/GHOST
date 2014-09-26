#include "ghost/colpack.h"
#include "ghost/crs.h"
#include "ghost/util.h"
#ifdef GHOST_HAVE_COLPACK
#include "ColPack/ColPackHeaders.h"
#endif

extern "C" ghost_error_t ghost_sparsemat_coloring_create(ghost_sparsemat_t *mat)
{
#ifdef GHOST_HAVE_COLPACK
    ghost_error_t ret = GHOST_SUCCESS;
    ghost_crs_t *crmat = CR(mat->localPart);
    ghost_lidx_t *curcol = NULL;
    int64_t nzloc=mat->localPart->nnz;
    uint32_t** adolc = new uint32_t*[mat->nrows];
    std::vector<int>* colvec = NULL;
    uint32_t *adolc_data=new uint32_t[nzloc+mat->nrows];
    ColPack::GraphColoring *GC=new ColPack::GraphColoring();


    if (mat->traits->format != GHOST_SPARSEMAT_CRS) {
        ERROR_LOG("Coloring only working for CRS at the moment!");
        return GHOST_ERR_NOT_IMPLEMENTED;
    }

    int64_t pos=0;
    for (int i=0;i<mat->nrows;i++)
    {
        adolc[i]=&(adolc_data[pos]);
        adolc_data[pos++]=crmat->rpt[i+1]-crmat->rpt[i];
        for (int j=crmat->rpt[i];j<crmat->rpt[i+1];j++)
        {
            adolc_data[pos++]=crmat->col[j];
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

    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->colors,mat->nrows*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->color_map,mat->nrows*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->color_ptr,(mat->ncolors+1)*sizeof(ghost_lidx_t)),err,ret);

    GHOST_CALL_GOTO(ghost_malloc((void **)&curcol,(mat->ncolors)*sizeof(ghost_lidx_t)),err,ret);
    memset(curcol,0,mat->ncolors*sizeof(ghost_lidx_t));
    
    colvec = GC->GetVertexColorsPtr();

    for (int i=0;i<mat->nrows;i++) {
        mat->colors[i] = (*colvec)[i];
    }
    
    for (int i=0;i<mat->ncolors+1;i++) {
        mat->color_ptr[i] = 0;
    }

    for (int i=0;i<mat->nrows;i++) {
        mat->color_ptr[mat->colors[i]+1]++;
    }

    for (int i=1;i<mat->ncolors+1;i++) {
        mat->color_ptr[i] += mat->color_ptr[i-1];
    }
    
    for (int i=0;i<mat->nrows;i++) {
        mat->color_map[i] = curcol[mat->colors[i]] + mat->color_ptr[mat->colors[i]];
        curcol[mat->colors[i]]++;
    }

    goto out;
err:

out:
    delete [] adolc_data;
    delete [] adolc;
    delete GC;
    free(curcol);

    return ret;
#else
    UNUSED(mat);
    ERROR_LOG("ColPack not available!");
    return GHOST_ERR_NOT_IMPLEMENTED;
#endif
}

