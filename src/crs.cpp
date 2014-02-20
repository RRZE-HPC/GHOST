#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/omp.h"

#ifdef GHOST_HAVE_MPI
#include <mpi.h> //mpi.h has to be included before stdio.h
#endif

#include "ghost/locality.h"
#include "ghost/complex.h"
#include "ghost/math.h"
#include "ghost/util.h"
#include "ghost/crs.h"
#include "ghost/constants.h"
#include "ghost/io.h"

#include <iostream>

using namespace std;
// TODO shift, scale als templateparameter

template<typename m_t, typename v_t> static ghost_error_t CRS_kernel_plain_tmpl(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{
    ghost_crs_t *cr = CR(mat);
    v_t *rhsv = NULL;
    v_t *lhsv = NULL;
    v_t *local_dot_product = NULL, *partsums = NULL;
    m_t *mval = (m_t *)(cr->val);
    ghost_idx_t i, j;
    ghost_idx_t v;
    int nthreads = 1;

// TODO false sharing avoidance w/ hwloc

    v_t hlp1 = 0.;
    v_t shift, scale, beta;
    if (options & GHOST_SPMV_APPLY_SHIFT)
        shift = *((v_t *)(mat->traits->shift));
    if (options & GHOST_SPMV_APPLY_SCALE)
        scale = *((v_t *)(mat->traits->scale));
    if (options & GHOST_SPMV_AXPBY)
        beta = *((v_t *)(mat->traits->beta));
    if (options & GHOST_SPMV_COMPUTE_LOCAL_DOTPRODUCT) {
        local_dot_product = ((v_t *)(lhs->traits->localdot));

#pragma omp parallel 
        {
#pragma omp single
        nthreads = ghost_ompGetNumThreads();
        }

        // 3 -> 16: avoid false sharing
        GHOST_CALL_RETURN(ghost_malloc((void **)&partsums,16*lhs->traits->ncols*nthreads*sizeof(v_t))); 
        for (i=0; i<16*lhs->traits->ncols*nthreads; i++) {
            partsums[i] = 0.;
        }
    }

#pragma omp parallel private (i,hlp1, j, rhsv, lhsv,v) shared (partsums)
    {
        int tid = ghost_ompGetThreadNum();
#pragma omp for schedule(runtime) 
        for (i=0; i<mat->nrows; i++){
            for (v=0; v<MIN(lhs->traits->ncols,rhs->traits->ncols); v++)
            {
                rhsv = (v_t *)rhs->val[v];
                lhsv = (v_t *)lhs->val[v];
                hlp1 = (v_t)0.0;
                for (j=cr->rpt[i]; j<cr->rpt[i+1]; j++){
                    hlp1 += ((v_t)(mval[j])) * rhsv[cr->col[j]];
                }

                if (options & GHOST_SPMV_APPLY_SHIFT) {
                    hlp1 = hlp1-shift*rhsv[i];
                }
                if (options & GHOST_SPMV_APPLY_SCALE) {
                    hlp1 = hlp1*scale;
                }
                if (options & GHOST_SPMV_AXPY) {
                    lhsv[i] += (hlp1);
                } else if (options & GHOST_SPMV_AXPBY) {
                    lhsv[i] = beta*lhsv[i] + hlp1;
                } else {
                    lhsv[i] = (hlp1);
                }

                if (options & GHOST_SPMV_COMPUTE_LOCAL_DOTPRODUCT) {
                    partsums[(v+tid*lhs->traits->ncols)*16 + 0] += conjugate(&lhsv[i])*lhsv[i];
                    partsums[(v+tid*lhs->traits->ncols)*16 + 1] += conjugate(&lhsv[i])*rhsv[i];
                    partsums[(v+tid*lhs->traits->ncols)*16 + 2] += conjugate(&rhsv[i])*rhsv[i];
                }
            }
        }
    }
    if (options & GHOST_SPMV_COMPUTE_LOCAL_DOTPRODUCT) {
        for (v=0; v<MIN(lhs->traits->ncols,rhs->traits->ncols); v++) {
            for (i=0; i<nthreads; i++) {
                local_dot_product[v                       ] += partsums[(v+i*lhs->traits->ncols)*16 + 0];
                local_dot_product[v +   lhs->traits->ncols] += partsums[(v+i*lhs->traits->ncols)*16 + 1];
                local_dot_product[v + 2*lhs->traits->ncols] += partsums[(v+i*lhs->traits->ncols)*16 + 2];
            }
        }
        free(partsums);
    }
    return GHOST_SUCCESS;
}

template <typename m_t> static const char * CRS_stringify(ghost_sparsemat_t *mat, int dense)
{
    ghost_idx_t i,j,col;
    m_t *val = (m_t *)CR(mat)->val;

    stringstream buffer;

    for (i=0; i<mat->nrows; i++) {
        if (dense) {
            for (col=0, j=CR(mat)->rpt[i]; col<mat->ncols; col++) {
                if (j<CR(mat)->rpt[i+1] && (CR(mat)->col[j] == col)) { // there is an entry at col
                    buffer << val[j] << "\t";
                    j++;
                } else {
                    buffer << ".\t";
                }

            }
        } else {
            for (j=CR(mat)->rpt[i]; j<CR(mat)->rpt[i+1]; j++) {
                if (mat->traits->flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                    buffer << val[j] << " (" << CR(mat)->col[j] << ")" << "\t";
                } else {
                    if (CR(mat)->col[j] < mat->nrows) {
                        buffer << val[j] << " (o " << mat->invRowPerm[CR(mat)->col[j]] << "|p " << CR(mat)->col[j] << ")" << "\t";
                    } else {
                        buffer << val[j] << " (p " << CR(mat)->col[j] << "|p " << CR(mat)->col[j] << ")" << "\t";
                    }
                }

            }
        }
        if (i<mat->nrows-1) {
            buffer << endl;
        }
    }

    return buffer.str().c_str();
}

extern "C" ghost_error_t dd_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< double,double >(mat,lhs,rhs,options); }

extern "C" ghost_error_t ds_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< double,float >(mat,lhs,rhs,options); }

extern "C" ghost_error_t dc_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< double,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" ghost_error_t dz_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< double,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" ghost_error_t sd_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< float,double >(mat,lhs,rhs,options); }

extern "C" ghost_error_t ss_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< float,float >(mat,lhs,rhs,options); }

extern "C" ghost_error_t sc_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< float,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" ghost_error_t sz_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< float,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" ghost_error_t cd_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< ghost_complex<float>,double >(mat,lhs,rhs,options); }

extern "C" ghost_error_t cs_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< ghost_complex<float>,float >(mat,lhs,rhs,options); }

extern "C" ghost_error_t cc_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< ghost_complex<float>,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" ghost_error_t cz_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< ghost_complex<float>,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" ghost_error_t zd_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< ghost_complex<double>,double >(mat,lhs,rhs,options); }

extern "C" ghost_error_t zs_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< ghost_complex<double>,float >(mat,lhs,rhs,options); }

extern "C" ghost_error_t zc_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< ghost_complex<double>,ghost_complex<float> >(mat,lhs,rhs,options); }

extern "C" ghost_error_t zz_CRS_kernel_plain(ghost_sparsemat_t *mat, ghost_densemat_t *lhs, ghost_densemat_t *rhs, ghost_spmv_flags_t options)
{ return CRS_kernel_plain_tmpl< ghost_complex<double>,ghost_complex<double> >(mat,lhs,rhs,options); }

extern "C" const char * d_CRS_stringify(ghost_sparsemat_t *mat, int dense)
{ return CRS_stringify< double >(mat, dense); }

extern "C" const char * s_CRS_stringify(ghost_sparsemat_t *mat, int dense)
{ return CRS_stringify< float >(mat, dense); }

extern "C" const char * z_CRS_stringify(ghost_sparsemat_t *mat, int dense)
{ return CRS_stringify< ghost_complex<double> >(mat, dense); }

extern "C" const char * c_CRS_stringify(ghost_sparsemat_t *mat, int dense)
{ return CRS_stringify< ghost_complex<float> >(mat, dense); }
