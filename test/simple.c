#include <ghost.h>
#include "ghost_test.h"

#define N 4

static int diag(ghost_gidx_t row, ghost_lidx_t *rowlen, ghost_gidx_t *col, void *val, void *arg)
{
    *rowlen = 1;
    col[0] = row;
    ((mydata_t *)val)[0] = (mydata_t)(row+1);
    
    return 0;
}


int main(int argc, char **argv) {
    ghost_context_t *ctx;
    ghost_sparsemat_t *A;
    ghost_densemat_t *y, *x;
    double zero = 0.;
    
    ghost_sparsemat_traits_t mtraits = GHOST_SPARSEMAT_TRAITS_INITIALIZER;
    ghost_densemat_traits_t vtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    ghost_spmv_flags_t spmvflags = GHOST_SPMV_DEFAULT;
    ghost_sparsemat_src_rowfunc_t matsrc = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    
    GHOST_TEST_CALL(ghost_init(argc,argv));
    GHOST_TEST_CALL(ghost_context_create(&ctx,N,N,GHOST_CONTEXT_DEFAULT,diag,GHOST_SPARSEMAT_SRC_FUNC,MPI_COMM_WORLD,1.));
    GHOST_TEST_CALL(ghost_sparsemat_create(&A, ctx, &mtraits, 1));
    matsrc.func = diag;
    matsrc.maxrowlen = N;
    
    GHOST_TEST_CALL(A->fromRowFunc(A,&matsrc));
    GHOST_TEST_CALL(ghost_densemat_create(&x, ctx, vtraits));
    GHOST_TEST_CALL(ghost_densemat_create(&y, ctx, vtraits));
    GHOST_TEST_CALL(x->fromRand(x));
    GHOST_TEST_CALL(y->fromScalar(y,&zero));

    GHOST_TEST_CALL(ghost_spmv(y,A,x,&spmvflags));

    double yent = 0., xent = 0., yent_ref = 0.;

    ghost_lidx_t i;

    for (i=0; i<y->traits.nrows; i++) {
        GHOST_TEST_CALL(y->entry(y,&yent,i,0));
        GHOST_TEST_CALL(x->entry(x,&xent,i,0));
        yent_ref = (i+1)*xent;
        if (DIFFER(yent,yent_ref,1)) {
            return EXIT_FAILURE;
        }
    }
        

    
    GHOST_TEST_CALL(ghost_finalize());
    return EXIT_SUCCESS;
}
