#include <ghost.h>
#include <stdio.h>

#define N 4

static int diag(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, __attribute__((unused)) void *arg)
{
    *rowlen = 1;
    col[0] = row;
    ((double *)val)[0] = (double)(row+1);
    
    return 0;
}

int main(int argc, char **argv) 
{
    ghost_context *ctx;
    ghost_sparsemat *A;
    ghost_densemat *y, *x;
    double zero = 0.;
    char *Astr, *xstr, *ystr;
    
    ghost_sparsemat_traits mtraits = GHOST_SPARSEMAT_TRAITS_INITIALIZER;
    ghost_densemat_traits vtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;

    ghost_init(argc,argv);

    // create matrix source
    ghost_sparsemat_src_rowfunc matsrc = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    matsrc.func = diag;
    matsrc.maxrowlen = N;

    // create context
    ghost_context_create(&ctx,N,N,GHOST_CONTEXT_DEFAULT,&matsrc,GHOST_SPARSEMAT_SRC_FUNC,MPI_COMM_WORLD,1.);
   
    // create sparse matrix A from row-wise function    
    ghost_sparsemat_create(&A, ctx, &mtraits, 1);
    A->fromRowFunc(A,&matsrc);

    // create input vector x and output vector y
    ghost_densemat_create(&x, ctx, vtraits);
    ghost_densemat_create(&y, ctx, vtraits);
    x->fromRand(x); // x = random
    y->fromScalar(y,&zero); // y = 0

    // compute y = A*x
    ghost_spmv(y,A,x,GHOST_SPMV_TRAITS_INITIALIZER);
   
    // print y, A and x 
    A->string(A,&Astr,1);
    x->string(x,&xstr);
    y->string(y,&ystr);
    printf("%s\n=\n%s\n*\n%s\n",ystr,Astr,xstr);
   
    // clean up 
    free(Astr); 
    free(xstr);
    free(ystr);
    A->destroy(A);
    x->destroy(x);
    y->destroy(y);
    ghost_context_destroy(ctx);

    ghost_finalize();

    return 0;
}
