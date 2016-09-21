// CURRENTLY BROKEN

#include <altivec.h>

static void SELL_kernel_VSX (ghost_sparsemat *mat, ghost_densemat * lhs, ghost_densemat * invec, int options)
{
    ghost_lidx c,j;
    ghost_lidx offs;
    vector double tmp;
    vector double val;
    vector double rhs;


#pragma omp parallel for schedule(runtime) private(j,tmp,val,rhs,offs)
    for (c=0; c<mat->nrowsPadded>>1; c++) 
    { // loop over chunks
        tmp = vec_splats(0.);
        offs = mat->chunkStart[c];

        for (j=0; j<(mat->chunkStart[c+1]-mat->chunkStart[c])>>1; j++) 
        { // loop inside chunk
            val = vec_xld2(offs*sizeof(ghost_dt),mat->val);                      // load values
            rhs = vec_insert(invec->val[mat->col[offs++]],rhs,0);
            rhs = vec_insert(invec->val[mat->col[offs++]],rhs,1);
            tmp = vec_madd(val,rhs,tmp);
        }
        if (options & GHOST_SPMV_AXPY) {
            vec_xstd2(vec_add(tmp,vec_xld2(c*mat->chunkHeight*sizeof(ghost_dt),lhs->val)),c*mat->chunkHeight*sizeof(ghost_dt),lhs->val);
        } else {
            vec_xstd2(tmp,c*mat->chunkHeight*sizeof(ghost_dt),lhs->val);
        }
    }
}
