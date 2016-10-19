#include <ghost.h>
#include <vector>
#include <map>
#include <iostream>
#include "ghost_test.h"

#define N 4

using namespace std;

template<typename m_t>
static int diag(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg)
{
    (void)(arg);
    *rowlen = 1;
    col[0] = row;
    ((m_t *)val)[0] = (m_t)(row+1);
    
    return 0;
}

template<typename v_t, typename m_t>
static void diag_ref(void *ref, ghost_gidx row, void *x)
{
    ghost_lidx dummyrowlen;
    ghost_gidx dummycol;
    m_t diagent;
    diag<m_t>(row,&dummyrowlen,&dummycol,&diagent,NULL);
    *((v_t *)ref) = (v_t)diagent*(*(v_t *)x);
}


typedef void (*ref_func_t)(void *, ghost_gidx, void *); 

GHOST_REGISTER_DT_D(dt_d)
GHOST_REGISTER_DT_S(dt_s)
GHOST_REGISTER_DT_Z(dt_z)
GHOST_REGISTER_DT_C(dt_c)

int main(int argc, char **argv) {
    ghost_sparsemat *A;
    ghost_densemat *y, *x;
    std::complex<double> zero = 0.;

    vector<ghost_sparsemat_traits> mtraits_vec;
    vector<ghost_densemat_traits> vtraits_vec;
    vector<ghost_densemat_storage> densemat_storages_vec;
    vector<ghost_datatype> datatypes_vec;

    datatypes_vec.push_back((ghost_datatype)(GHOST_DT_REAL|GHOST_DT_DOUBLE));
    datatypes_vec.push_back((ghost_datatype)(GHOST_DT_COMPLEX|GHOST_DT_DOUBLE));
    datatypes_vec.push_back((ghost_datatype)(GHOST_DT_REAL|GHOST_DT_FLOAT));
    datatypes_vec.push_back((ghost_datatype)(GHOST_DT_COMPLEX|GHOST_DT_FLOAT));


    densemat_storages_vec.push_back(GHOST_DENSEMAT_ROWMAJOR);
    densemat_storages_vec.push_back(GHOST_DENSEMAT_COLMAJOR);
    
    ghost_sparsemat_traits mtraits = GHOST_SPARSEMAT_TRAITS_INITIALIZER;
    for (vector<ghost_datatype>::iterator datatypes_it = datatypes_vec.begin(); datatypes_it != datatypes_vec.end(); ++datatypes_it) {
        mtraits.datatype = *datatypes_it;
        mtraits_vec.push_back(mtraits);
    }
    
    ghost_densemat_traits vtraits = GHOST_DENSEMAT_TRAITS_INITIALIZER;
    for (vector<ghost_densemat_storage>::iterator densemat_storages_it = densemat_storages_vec.begin(); densemat_storages_it != densemat_storages_vec.end(); ++densemat_storages_it) {
        for (vector<ghost_datatype>::iterator datatypes_it = datatypes_vec.begin(); datatypes_it != datatypes_vec.end(); ++datatypes_it) {
            vtraits.storage = *densemat_storages_it;
            vtraits.datatype = *datatypes_it;
            vtraits_vec.push_back(vtraits);
        }
    }
            

    map<ghost_datatype,ghost_sparsemat_src_rowfunc> mat_funcs_diag;
    ghost_sparsemat_src_rowfunc matsrc = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    matsrc.maxrowlen = 1;
    matsrc.gnrows = N;
    matsrc.gncols = N;
    matsrc.func = diag<double>;
    mat_funcs_diag[(ghost_datatype)(GHOST_DT_REAL|GHOST_DT_DOUBLE)] = matsrc;
    matsrc.func = diag<float>;
    mat_funcs_diag[(ghost_datatype)(GHOST_DT_REAL|GHOST_DT_FLOAT)] = matsrc;
    matsrc.func = diag<std::complex<double>>;
    mat_funcs_diag[(ghost_datatype)(GHOST_DT_COMPLEX|GHOST_DT_DOUBLE)] = matsrc;
    matsrc.func = diag<std::complex<float>>;
    mat_funcs_diag[(ghost_datatype)(GHOST_DT_COMPLEX|GHOST_DT_FLOAT)] = matsrc;
    
    map<pair<ghost_datatype,ghost_datatype>,ref_func_t> ref_funcs_diag;
    ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_d,dt_d)] = diag_ref<double,double>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_d,dt_s)] = diag_ref<double,float>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_d,dt_z)] = diag_ref<double,std::complex<double>>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_d,dt_c)] = diag_ref<double,std::complex<float>>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_s,dt_d)] = diag_ref<float,double>;
    ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_s,dt_s)] = diag_ref<float,float>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_s,dt_z)] = diag_ref<float,std::complex<double>>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_s,dt_c)] = diag_ref<float,std::complex<float>>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_z,dt_d)] = diag_ref<std::complex<double>,double>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_z,dt_s)] = diag_ref<std::complex<double>,float>;
    ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_z,dt_z)] = diag_ref<std::complex<double>,std::complex<double>>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_z,dt_c)] = diag_ref<std::complex<double>,std::complex<float>>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_c,dt_d)] = diag_ref<std::complex<float>,double>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_c,dt_s)] = diag_ref<std::complex<float>,float>;
    //ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_c,dt_z)] = diag_ref<std::complex<float>,std::complex<double>>;
    ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(dt_c,dt_c)] = diag_ref<std::complex<float>,std::complex<float>>;

    GHOST_TEST_CALL(ghost_init(argc,argv));
    
   
    for (vector<ghost_sparsemat_traits>::iterator mtraits_it = mtraits_vec.begin(); mtraits_it != mtraits_vec.end(); ++mtraits_it) {
        
        // create sparsemat with traits and set according source function
        GHOST_TEST_CALL(ghost_sparsemat_create(&A, NULL, &(*mtraits_it), 1));
        GHOST_TEST_CALL(ghost_sparsemat_init_rowfunc(A,&mat_funcs_diag[mtraits_it->datatype],MPI_COMM_WORLD,1.));

        for (vector<ghost_densemat_traits>::iterator vtraits_it = vtraits_vec.begin(); vtraits_it != vtraits_vec.end(); ++vtraits_it) {
            if (!ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(vtraits_it->datatype,mtraits_it->datatype)]) continue;

            GHOST_TEST_CALL(ghost_densemat_create(&x, A->context->col_map, *vtraits_it));
            GHOST_TEST_CALL(ghost_densemat_create(&y, A->context->row_map, *vtraits_it));
            GHOST_TEST_CALL(ghost_densemat_init_rand(x));
            GHOST_TEST_CALL(ghost_densemat_init_val(y,&zero));
          
            printf("Test SpMV with %s matrix (SELL-%d-%d) and %s vectors (%s)\n",ghost_datatype_string(A->traits.datatype),A->traits.C,A->traits.sortScope,ghost_datatype_string(x->traits.datatype),ghost_densemat_storage_string(x->traits.storage));
            GHOST_TEST_CALL(ghost_spmv(y,A,x,GHOST_SPMV_OPTS_INITIALIZER));

            size_t vecdtsize;
            ghost_datatype_size(&vecdtsize,vtraits_it->datatype);
            char yent[16], yent_ref[16], xent[16];

#ifdef GHOST_HAVE_CUDA
            ghost_type ghost_type;
            GHOST_TEST_CALL(ghost_type_get(&ghost_type));
            if (ghost_type == GHOST_TYPE_CUDA)
            {
               GHOST_TEST_CALL(ghost_densemat_download(x));
               GHOST_TEST_CALL(ghost_densemat_download(y));
            }
#endif


            ghost_lidx i;

            for (i=0; i<DM_NROWS(y); i++) {
                GHOST_TEST_CALL(ghost_densemat_entry(&yent,y,i,0));
                GHOST_TEST_CALL(ghost_densemat_entry(&xent,x,i,0));
                ref_funcs_diag[pair<ghost_datatype,ghost_datatype>(vtraits_it->datatype,mtraits_it->datatype)](yent_ref,i,xent);
                RETURN_IF_DIFFER((void *)yent,(void *)yent_ref,1,vtraits_it->datatype);
            }

            ghost_densemat_destroy(x);
            ghost_densemat_destroy(y);
        }
            
        ghost_sparsemat_destroy(A);

    }


        
    
    GHOST_TEST_CALL(ghost_finalize());
   
    return EXIT_SUCCESS;
}
