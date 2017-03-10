#include <ghost.h>
#include <ghost/util.h>
#include <stdio.h>
#include <math.h>

/* 
 * Test matrix looks as follows:
 * 1.0 ... 1.1 ...
 * 2.1 2.0 ... 2.2
 * ... 3.1 ... ...
 * ... 4.1 4.2 4.0
 *
 * Stored in SELL-2-4, the diag is:
 * 2.0 1.0 0.0 4.0
 *
 */
static int testmatrix(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *v, __attribute__((unused)) void *arg)
{
    double *val = (double *)v;
    switch (row) {
        case 0:
            *rowlen = 2;
            memcpy(val, (double[]) {1.0, 1.1}, *rowlen*sizeof(double));
            memcpy(col, (ghost_gidx[]) {0, 2}, *rowlen*sizeof(ghost_gidx));
            break;
        case 1:
            *rowlen = 3;
            memcpy(val, (double[]) {2.1, 2.0, 2.2}, *rowlen*sizeof(double));
            memcpy(col, (ghost_gidx[]) {0, 1, 3}, *rowlen*sizeof(ghost_gidx));
            break;
        case 2:
            *rowlen = 1;
            memcpy(val, (double[]) {3.1}, *rowlen*sizeof(double));
            memcpy(col, (ghost_gidx[]) {1}, *rowlen*sizeof(ghost_gidx));
            break;
        case 3:
            *rowlen = 3;
            memcpy(val, (double[]) {4.1, 4.2, 4.0}, *rowlen*sizeof(double));
            memcpy(col, (ghost_gidx[]) {1, 2, 3}, *rowlen*sizeof(ghost_gidx));
            break;
    }
    
    return 0;
}

int main(int argc, char **argv) 
{
    ghost_sparsemat *A;
    char *Astr;
    int me,nrank;
    
    ghost_sparsemat_traits mtraits = GHOST_SPARSEMAT_TRAITS_INITIALIZER;
    mtraits.datatype = (ghost_datatype)(GHOST_DT_REAL|GHOST_DT_DOUBLE);
    mtraits.flags = (ghost_sparsemat_flags)(GHOST_SPARSEMAT_SAVE_ORIG_COLS|GHOST_SPARSEMAT_DIAG_FIRST); // needed for printing if more than one process
    mtraits.sortScope = 4;
    mtraits.C = 2;

    GHOST_CALL_RETURN(ghost_init(argc,argv));
    GHOST_CALL_RETURN(ghost_rank(&me,MPI_COMM_WORLD));
    GHOST_CALL_RETURN(ghost_nrank(&nrank,MPI_COMM_WORLD));

    // create matrix source
    ghost_sparsemat_src_rowfunc matsrc = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    matsrc.func = testmatrix;
    matsrc.maxrowlen = 3;
    matsrc.gnrows = 4;

    // create sparse matrix A from row-wise source function    
    GHOST_CALL_RETURN(ghost_sparsemat_create(&A, NULL, &mtraits, 1));
    GHOST_CALL_RETURN(ghost_sparsemat_init_rowfunc(A,&matsrc,MPI_COMM_WORLD,1.));

    // print y, A and x 
    GHOST_CALL_RETURN(ghost_sparsemat_string(&Astr,A,1));
    printf("%s\n",Astr);

    double *diag;
    GHOST_CALL_RETURN(ghost_malloc((void **)&diag,A->context->row_map->dim*sizeof(double)));
    if (nrank == 0) {
        memcpy(diag, (double []) {2.0, 4.0, 1.0, 0.0}, 4*sizeof(double));
    } else if (nrank == 1) {
        if (me == 0) {
           memcpy(diag, (double []) {2.0, 1.0}, 2*sizeof(double));
        } else {
           memcpy(diag, (double []) {4.0, 0.0}, 2*sizeof(double));
        }
    }
    int chunk;
    int err = 0;
    for (chunk=0; chunk<A->nchunks; chunk++) {

        int i;
        for (i=0; i<mtraits.C; i++) {
            if (fabs(diag[A->context->row_map->goffs[me]+mtraits.C*chunk+i]-((double *)A->val)[A->chunkStart[chunk]+i] > 1.e-14)) {
                printf("Error in row %d! [should is] %f %f\n",mtraits.C*chunk+i,diag[A->context->row_map->goffs[me]+mtraits.C*chunk+i],((double *)A->val)[A->chunkStart[chunk]+i]);
                err = 1;
            }
        }
    }
   
    // clean up 
    free(Astr); 
    ghost_sparsemat_destroy(A);

    ghost_finalize();

    return err;
}
