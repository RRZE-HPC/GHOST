#include <ghost.h>
#include <stdio.h>
#include <mpi.h>

#include "scamac.h"


struct my_work_st {
 ScamacGenerator * gen;
 ScamacWorkspace * ws;
};

static int my_func(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg)
{
    struct my_work_st * my_work = arg;
    ScamacErrorCode err=scamac_generate_row(my_work->gen, my_work->ws, (ScamacIdx) row, SCAMAC_DEFAULT, (ScamacIdx*) rowlen, (ScamacIdx *) col, (double *) val); 
    if (err) { return 1; }
    return 0;
}

static int my_funcinit(void *arg, void **work) {
  ScamacGenerator * gen = (ScamacGenerator *) arg;
  if (*work) {// free
    struct my_work_st * my_work = *work;
    scamac_workspace_free( my_work->ws);
    free(my_work);
    *work=NULL;
  } else {//alloc
    struct my_work_st * my_work = malloc(sizeof *my_work);
    my_work->gen = (ScamacGenerator * ) arg;
    scamac_workspace_alloc(my_work->gen, &(my_work->ws));
    *work = my_work;
  } 
}

int main(int argc, char **argv) 
{
    ghost_sparsemat *A;
    ghost_densemat *y, *x, *ax, *ay;
    double zero = 0.;
    char *Astr=NULL, *xstr=NULL, *ystr=NULL;
    
    ghost_densemat_traits vtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    
    ghost_sparsemat_traits mtraits = GHOST_SPARSEMAT_TRAITS_INITIALIZER;
    mtraits.datatype = (ghost_datatype)(GHOST_DT_REAL|GHOST_DT_DOUBLE);
    mtraits.flags = GHOST_SPARSEMAT_SAVE_ORIG_COLS; // needed for printing if more than one process

    GHOST_CALL_RETURN(ghost_init(argc,argv));

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    ScamacErrorCode err;
    ScamacGenerator * my_gen;
    if (argc>1) {
      SCAMAC_TRY_MPI(scamac_parse_argstr(argv[1],&my_gen,NULL));
    } else {
      SCAMAC_TRY_MPI(scamac_generator_obtain("Hubbard", &my_gen));
      SCAMAC_TRY_MPI(scamac_generator_set_int(my_gen,"n_sites",12));
    }
    SCAMAC_TRY_MPI(scamac_generator_finalize(my_gen));

    // create matrix source
    ghost_sparsemat_src_rowfunc matsrc = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    matsrc.func = my_func;
    matsrc.funcinit = my_funcinit;
    matsrc.maxrowlen = scamac_generator_query_maxnzrow(my_gen);
    matsrc.gnrows = scamac_generator_query_nrow(my_gen);
    matsrc.arg = my_gen;

    // create sparse matrix A from row-wise source function    
    GHOST_CALL_RETURN(ghost_sparsemat_create(&A, NULL, &mtraits, 1));
    GHOST_CALL_RETURN(ghost_sparsemat_init_rowfunc(A,&matsrc,MPI_COMM_WORLD,0.));

    // create and initialize input vector x and output vector y
    GHOST_CALL_RETURN(ghost_densemat_create(&x, ghost_context_max_map(A->context), vtraits));
    GHOST_CALL_RETURN(ghost_densemat_create(&y, ghost_context_max_map(A->context), vtraits));
    GHOST_CALL_RETURN(ghost_densemat_create(&ax, ghost_context_max_map(A->context), vtraits));
    GHOST_CALL_RETURN(ghost_densemat_create(&ay, ghost_context_max_map(A->context), vtraits));
    GHOST_CALL_RETURN(ghost_densemat_init_rand(x));      // x = random
    GHOST_CALL_RETURN(ghost_densemat_init_rand(y));      // y = random
    GHOST_CALL_RETURN(ghost_densemat_init_val(ax,&zero)); // ax = 0
    GHOST_CALL_RETURN(ghost_densemat_init_val(ay,&zero)); // ay = 0

    // compute ax = A*x
    GHOST_CALL_RETURN(ghost_spmv(ax,A,x,GHOST_SPMV_OPTS_INITIALIZER));
    // compute ay = A*y
    GHOST_CALL_RETURN(ghost_spmv(ay,A,y,GHOST_SPMV_OPTS_INITIALIZER));

    double sp1, sp2;
    GHOST_CALL_RETURN(ghost_dot(&sp1, ax, y));
    GHOST_CALL_RETURN(ghost_dot(&sp2, x, ay));

    if (mpi_rank == 0) {
      printf("for a matrix with %"PRGIDX" rows:\n",matsrc.gnrows);
      printf("scalar products <Ax, y>=%g =? <x,Ay>=%g [should be equal for symmetric A]\n",sp1,sp2);
    }

    // clean up 
    free(Astr); 
    free(xstr);
    free(ystr);
    ghost_sparsemat_destroy(A);
    ghost_densemat_destroy(x);
    ghost_densemat_destroy(y);

    SCAMAC_TRY_MPI(scamac_generator_destroy(my_gen));

    ghost_finalize();

    return 0;
}
