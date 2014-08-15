#include <ghost.h>
#include <vector>
#include <map>
#include <complex>
#include <iostream>
#include "ghost_test.h"

#define N 4

using namespace std;

typedef void (*ref_func_t)(void *, ghost_gidx_t, void *); 

static int diag_d(ghost_gidx_t row, ghost_lidx_t *rowlen, ghost_gidx_t *col, void *val)
{
    *rowlen = 1;
    col[0] = row;
    ((double *)val)[0] = (double)(row+1);
    
    return 0;
}

static void diag_d_ref(void *ref, ghost_gidx_t row, void *x)
{
    *(double *)ref = (row+1)*(*(double *)x);
}

static int diag_s(ghost_gidx_t row, ghost_lidx_t *rowlen, ghost_gidx_t *col, void *val)
{
    *rowlen = 1;
    col[0] = row;
    ((float *)val)[0] = (float)(row+1);
    
    return 0;
}

static void diag_s_ref(void *ref, ghost_gidx_t row, void *x)
{
    *(float *)ref = (row+1)*(*(float *)x);
}

static int diag_z(ghost_gidx_t row, ghost_lidx_t *rowlen, ghost_gidx_t *col, void *val)
{
    *rowlen = 1;
    col[0] = row;
    ((complex<double> *)val)[0] = complex<double>((double)(row+1),(double)(row+1));
    
    return 0;
}

static void diag_z_ref(void *ref, ghost_gidx_t row, void *x)
{
    *(complex<double> *)ref = (complex<double>)(row+1)*(*(complex<double> *)x);
}

static int diag_c(ghost_gidx_t row, ghost_lidx_t *rowlen, ghost_gidx_t *col, void *val)
{
    *rowlen = 1;
    col[0] = row;
    ((complex<float> *)val)[0] = complex<float>((float)(row+1),(float)(row+1));
    
    return 0;
}

static void diag_c_ref(void *ref, ghost_gidx_t row, void *x)
{
    *(complex<float> *)ref = (complex<float>)(row+1)*(*(complex<float> *)x);
}


int main(int argc, char **argv) {
    ghost_context_t *ctx;
    ghost_sparsemat_t *A;
    ghost_densemat_t *y, *x;
    complex<double> zero = 0.;

    vector<ghost_sparsemat_traits_t> mtraits_vec;
    vector<ghost_densemat_traits_t> vtraits_vec;
    vector<ghost_sparsemat_format_t> formats_vec;
    vector<ghost_densemat_storage_t> densemat_storages_vec;
    vector<ghost_datatype_t> datatypes_vec;

    datatypes_vec.push_back((ghost_datatype_t)(GHOST_DT_REAL|GHOST_DT_DOUBLE));
    datatypes_vec.push_back((ghost_datatype_t)(GHOST_DT_COMPLEX|GHOST_DT_DOUBLE));
    datatypes_vec.push_back((ghost_datatype_t)(GHOST_DT_REAL|GHOST_DT_FLOAT));
    datatypes_vec.push_back((ghost_datatype_t)(GHOST_DT_COMPLEX|GHOST_DT_FLOAT));

    formats_vec.push_back(GHOST_SPARSEMAT_CRS);
    formats_vec.push_back(GHOST_SPARSEMAT_SELL);

    densemat_storages_vec.push_back(GHOST_DENSEMAT_ROWMAJOR);
    densemat_storages_vec.push_back(GHOST_DENSEMAT_COLMAJOR);
    
    ghost_sparsemat_traits_t mtraits = GHOST_SPARSEMAT_TRAITS_INITIALIZER;
    for (vector<ghost_sparsemat_format_t>::iterator formats_it = formats_vec.begin(); formats_it != formats_vec.end(); ++formats_it) {
        for (vector<ghost_datatype_t>::iterator datatypes_it = datatypes_vec.begin(); datatypes_it != datatypes_vec.end(); ++datatypes_it) {
            mtraits.format = *formats_it;
            mtraits.datatype = *datatypes_it;
            mtraits_vec.push_back(mtraits);
        }
    }
    
    ghost_densemat_traits_t vtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    for (vector<ghost_densemat_storage_t>::iterator densemat_storages_it = densemat_storages_vec.begin(); densemat_storages_it != densemat_storages_vec.end(); ++densemat_storages_it) {
        for (vector<ghost_datatype_t>::iterator datatypes_it = datatypes_vec.begin(); datatypes_it != datatypes_vec.end(); ++datatypes_it) {
            vtraits.storage = *densemat_storages_it;
            vtraits.datatype = *datatypes_it;
            vtraits_vec.push_back(vtraits);
        }
    }
            

    map<ghost_datatype_t,ghost_sparsemat_src_rowfunc_t> mat_funcs_diag;
    ghost_sparsemat_src_rowfunc_t matsrc = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    matsrc.maxrowlen = N;
    matsrc.func = diag_d;
    mat_funcs_diag[(ghost_datatype_t)(GHOST_DT_REAL|GHOST_DT_DOUBLE)] = matsrc;
    matsrc.func = diag_s;
    mat_funcs_diag[(ghost_datatype_t)(GHOST_DT_REAL|GHOST_DT_FLOAT)] = matsrc;
    matsrc.func = diag_z;
    mat_funcs_diag[(ghost_datatype_t)(GHOST_DT_COMPLEX|GHOST_DT_DOUBLE)] = matsrc;
    matsrc.func = diag_c;
    mat_funcs_diag[(ghost_datatype_t)(GHOST_DT_COMPLEX|GHOST_DT_FLOAT)] = matsrc;
    
    map<ghost_datatype_t,ref_func_t> ref_funcs_diag;
    ref_funcs_diag[(ghost_datatype_t)(GHOST_DT_REAL|GHOST_DT_DOUBLE)] = diag_d_ref;
    ref_funcs_diag[(ghost_datatype_t)(GHOST_DT_REAL|GHOST_DT_FLOAT)] = diag_s_ref;
    ref_funcs_diag[(ghost_datatype_t)(GHOST_DT_COMPLEX|GHOST_DT_DOUBLE)] = diag_z_ref;
    ref_funcs_diag[(ghost_datatype_t)(GHOST_DT_COMPLEX|GHOST_DT_FLOAT)] = diag_c_ref;


    ghost_spmv_flags_t spmvflags = GHOST_SPMV_DEFAULT;
    
    GHOST_TEST_CALL(ghost_init(argc,argv));
    
    GHOST_TEST_CALL(ghost_context_create(&ctx,N,N,GHOST_CONTEXT_DEFAULT,&matsrc,GHOST_SPARSEMAT_SRC_FUNC,MPI_COMM_WORLD,1.));
   
    for (vector<ghost_sparsemat_traits_t>::iterator mtraits_it = mtraits_vec.begin(); mtraits_it != mtraits_vec.end(); ++mtraits_it) {
        
        // create sparsemat with traits and set according source function
        GHOST_TEST_CALL(ghost_sparsemat_create(&A, ctx, &(*mtraits_it), 1));
        GHOST_TEST_CALL(A->fromRowFunc(A,&mat_funcs_diag[mtraits_it->datatype]));

        for (vector<ghost_densemat_traits_t>::iterator vtraits_it = vtraits_vec.begin(); vtraits_it != vtraits_vec.end(); ++vtraits_it) {
            GHOST_TEST_CALL(ghost_densemat_create(&x, ctx, *vtraits_it));
            GHOST_TEST_CALL(ghost_densemat_create(&y, ctx, *vtraits_it));
            GHOST_TEST_CALL(x->fromRand(x));
            GHOST_TEST_CALL(y->fromScalar(y,&zero));
          
            printf("Test SpMV with %s matrix (%s) and %s vectors (%s)\n",ghost_datatype_string(A->traits->datatype),A->formatName(A),ghost_datatype_string(x->traits.datatype),ghost_densemat_storage_string(x));
            GHOST_TEST_CALL(ghost_spmv(y,A,x,&spmvflags));

            size_t vecdtsize;
            ghost_datatype_size(&vecdtsize,vtraits_it->datatype);
            char yent[16], yent_ref[16], xent[16];

            ghost_lidx_t i;

            for (i=0; i<y->traits.nrows; i++) {
                GHOST_TEST_CALL(y->entry(y,&yent,i,0));
                GHOST_TEST_CALL(x->entry(x,&xent,i,0));
                ref_funcs_diag[vtraits_it->datatype](yent_ref,i,xent);
                RETURN_IF_DIFFER((void *)yent,(void *)yent_ref,1,vtraits_it->datatype);
            }

            x->destroy(x);
            y->destroy(y);
        }
            
        A->destroy(A);

    }


        
    
    GHOST_TEST_CALL(ghost_finalize());
   
    return EXIT_SUCCESS;
}
