#include <ghost.h>
#include <stdio.h>
#include <math.h>

#define N 4

static int matfunc(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg)
{
    if (row == 0) {
        *rowlen = 3;
        col[1] = 1;
        col[2] = N-2;
        ((double *)val)[1] = 2.2;
        ((double *)val)[2] = (N-1)+0.1*(N-1);
        if (!arg) {
            *rowlen = 4;
            col[3] = N-1;
            ((double *)val)[3] = N+0.1*N;
        }
    } else {
        *rowlen =1;
    }

    col[0] = row;
    ((double *)val)[0] = (double)(row+1);
    
    return 0;
}


int main(int argc, char* argv[]) 
{
    ghost_sparsemat *mat, *submat;
    ghost_densemat *x, *y;
    ghost_context *ctx;
    int rank,nranks,i;
    ghost_lidx assigned_rows = 0;
    char *str;
    ghost_lidx *rows_per_rank;
    double zero = 0., one = 1.;
    
    ghost_init(argc,argv);
    ghost_rank(&rank,MPI_COMM_WORLD);
    ghost_nrank(&nranks,MPI_COMM_WORLD);

    /*if (nranks != 2) {
        printf("This test requires two MPI ranks!\n");
        return EXIT_FAILURE;
    }*/

    ghost_sparsemat_traits mtraits = GHOST_SPARSEMAT_TRAITS_INITIALIZER;
    mtraits.datatype = (ghost_datatype)(GHOST_DT_REAL|GHOST_DT_DOUBLE);
    mtraits.flags = GHOST_SPARSEMAT_SAVE_ORIG_COLS;
    
    ghost_sparsemat_src_rowfunc matsrc = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    matsrc.func = matfunc;
    matsrc.maxrowlen = 4;
    matsrc.gnrows = N;
    
    if (!rank) 
        printf("### Creating sparse matrix without anything prescribed\n"); 
    GHOST_CALL_RETURN(ghost_sparsemat_create(&mat, NULL, &mtraits, 1));
    GHOST_CALL_RETURN(ghost_sparsemat_init_rowfunc(mat,&matsrc,MPI_COMM_WORLD,1.));

    GHOST_CALL_RETURN(ghost_sparsemat_string(&str,mat,1));
    printf("[%d] mat with auto context\n%s\n\n",rank,str);
    free(str);
    printf("[%d] nhalo = %d\n",rank,mat->context->col_map->nhalo);
  
    ghost_sparsemat_destroy(mat); 
    
    if (!rank) 
        printf("\n\n### Creating sparse matrix with prescribed row map (only a single row at rank 0)\n"); 
    
    GHOST_CALL_RETURN(ghost_context_create(&ctx,N,N,GHOST_CONTEXT_DEFAULT,MPI_COMM_WORLD,1.));
    rows_per_rank = (ghost_lidx *)malloc(nranks*sizeof(ghost_lidx));
    rows_per_rank[0] = 1;
    assigned_rows = 1;
    for (i=1; i<nranks-1; i++) {
        rows_per_rank[i] = (N-1)/nranks;
        assigned_rows += rows_per_rank[i];
    }
    rows_per_rank[nranks-1] = N-assigned_rows;

    GHOST_CALL_RETURN(ghost_map_create_distribution(ctx->row_map,&matsrc,1.,GHOST_MAP_DIST_NROWS,rows_per_rank));
    GHOST_CALL_RETURN(ghost_sparsemat_create(&mat, ctx, &mtraits, 1));
    GHOST_CALL_RETURN(ghost_sparsemat_init_rowfunc(mat,&matsrc,MPI_COMM_WORLD,1.));

    ghost_sparsemat_string(&str,mat,1);
    printf("[%d] mat with prescribed row distribution (single row at rank 0)\n%s\n\n",rank,str);
    free(str);
    printf("[%d] nhalo = %d\n",rank,mat->context->col_map->nhalo);
    
    ghost_sparsemat_destroy(mat); 
    
    if (!rank) { 
        printf("\n\n### Creating sparse matrix with prescribed row and valid col map (only a single row and col at rank 0)\n"); 
    }
    
    GHOST_CALL_RETURN(ghost_context_create(&ctx,N,N,GHOST_CONTEXT_DEFAULT,MPI_COMM_WORLD,1.));
    GHOST_CALL_RETURN(ghost_map_create_distribution(ctx->row_map,&matsrc,1.,GHOST_MAP_DIST_NROWS,rows_per_rank));
    GHOST_CALL_RETURN(ghost_map_create_distribution(ctx->col_map,&matsrc,1.,GHOST_MAP_DIST_NROWS,rows_per_rank));
    GHOST_CALL_RETURN(ghost_sparsemat_create(&mat, ctx, &mtraits, 1));
    GHOST_CALL_RETURN(ghost_sparsemat_init_rowfunc(mat,&matsrc,MPI_COMM_WORLD,1.));
    
    GHOST_CALL_RETURN(ghost_sparsemat_string(&str,mat,1));
    printf("[%d] mat with prescribed row distribution (single row at rank 0)\n%s\n\n",rank,str);
    free(str);
    printf("[%d] nhalo = %d\n",rank,mat->context->col_map->nhalo);
  
    ghost_sparsemat_destroy(mat); 
    
    if (!rank) {
        printf("\n\n### Creating two sparse matrices with the first being a 'submatrix' of the second (should fail on rank 0!)\n"); 
    }

    ghost_sparsemat_create(&submat, NULL, &mtraits, 1);
    matsrc.arg = (void *)0x1; // set to anything non-NULL to create the submatrix
    ghost_sparsemat_init_rowfunc(submat,&matsrc,MPI_COMM_WORLD,1.);
    
    GHOST_CALL_RETURN(ghost_context_create(&ctx,N,N,GHOST_CONTEXT_DEFAULT,MPI_COMM_WORLD,1.));
    GHOST_CALL_RETURN(ghost_context_set_map(ctx,GHOST_MAP_ROW,submat->context->row_map));
    GHOST_CALL_RETURN(ghost_context_set_map(ctx,GHOST_MAP_COL,submat->context->col_map));
    GHOST_CALL_RETURN(ghost_sparsemat_create(&mat, ctx, &mtraits, 1));
    matsrc.arg = NULL; 
    ghost_error ret = ghost_sparsemat_init_rowfunc(mat,&matsrc,MPI_COMM_WORLD,1.);
    if (nranks > 1 && rank == 0 && ret == GHOST_SUCCESS) {
        printf("[%d] did not fail!\n",rank);
        return EXIT_FAILURE;
    } else {
        printf("[%d] failed successfully with \"%s\"!\n",rank,ghost_error_string(ret));
    }

    ghost_sparsemat_destroy(mat); 
    ghost_sparsemat_destroy(submat); 
    
    if (!rank) {
        printf("\n\n### Creating two sparse matrices with the second being a 'submatrix' of the first\n"); 
    }
    GHOST_CALL_RETURN(ghost_sparsemat_create(&mat, NULL, &mtraits, 1));
    GHOST_CALL_RETURN(ghost_sparsemat_init_rowfunc(mat,&matsrc,MPI_COMM_WORLD,1.));
    
    GHOST_CALL_RETURN(ghost_context_create(&ctx,N,N,GHOST_CONTEXT_DEFAULT,MPI_COMM_WORLD,1.));
    GHOST_CALL_RETURN(ghost_context_set_map(ctx,GHOST_MAP_ROW,mat->context->row_map));
    GHOST_CALL_RETURN(ghost_context_set_map(ctx,GHOST_MAP_COL,mat->context->col_map));
    GHOST_CALL_RETURN(ghost_sparsemat_create(&submat, ctx, &mtraits, 1));
    matsrc.arg = (void *)0x1; // set to anything non-NULL to create the submatrix
    GHOST_CALL_RETURN(ghost_sparsemat_init_rowfunc(submat,&matsrc,MPI_COMM_WORLD,1.));

    GHOST_CALL_RETURN(ghost_sparsemat_string(&str,mat,1));
    printf("[%d] mat with auto context\n%s\n\n",rank,str);
    free(str);
    GHOST_CALL_RETURN(ghost_sparsemat_string(&str,submat,1));
    printf("[%d] submat with auto context\n%s\n\n",rank,str);
    free(str);
    printf("[%d] nhalo = %d\n",rank,mat->context->col_map->nhalo);

    GHOST_CALL_RETURN(ghost_densemat_create(&x, mat->context->col_map, GHOST_DENSEMAT_TRAITS_INITIALIZER));
    GHOST_CALL_RETURN(ghost_densemat_create(&y, mat->context->row_map, GHOST_DENSEMAT_TRAITS_INITIALIZER));
    GHOST_CALL_RETURN(ghost_densemat_init_val(x,&one));      // x = random
    GHOST_CALL_RETURN(ghost_densemat_init_val(y,&zero)); // y = 0
    
    GHOST_CALL_RETURN(ghost_spmv(y,mat,x,GHOST_SPMV_OPTS_INITIALIZER));
    GHOST_CALL_RETURN(ghost_densemat_string(&str,y));
    printf("[%d] y = mat*1\n%s\n\n",rank,str);
    free(str);
    
    GHOST_CALL_RETURN(ghost_spmv(y,submat,x,GHOST_SPMV_OPTS_INITIALIZER));
    GHOST_CALL_RETURN(ghost_densemat_string(&str,y));
    printf("[%d] y = submat*1\n%s\n\n",rank,str);
    free(str);
  
    ghost_sparsemat_destroy(mat); 
    ghost_sparsemat_destroy(submat); 

    ghost_finalize();

    return EXIT_SUCCESS;
}
