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
#include "ghost/machine.h"

#include <sstream>
#include <iostream>
#include <cstdarg>

using namespace std;

template <typename m_t> 
static ghost_error_t CRS_stringify_tmpl(ghost_sparsemat_t *mat, char ** str, int dense)
{
    ghost_lidx_t i,j,col;
    m_t *val = (m_t *)CR(mat)->val;

    stringstream buffer;

    for (i=0; i<mat->nrows; i++) {
        if (dense) {
            for (col=0, j=CR(mat)->rpt[i]; col<mat->ncols; col++) {
                if ((j<CR(mat)->rpt[i+1]) && ((mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)?(mat->col_orig[j] == col):(CR(mat)->col[j] == col))) { // there is an entry at col
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
                        buffer << val[j] << " (o " << mat->context->permutation->invPerm[CR(mat)->col[j]] << "|p " << CR(mat)->col[j] << ")" << "\t";
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

    *str = strdup(buffer.str().c_str());

    return GHOST_SUCCESS;
}

extern "C" ghost_error_t ghost_crs_stringify_selector(ghost_sparsemat_t *mat, char **str, int dense)
{
    ghost_error_t ret;

    SELECT_TMPL_1DATATYPE(mat->traits->datatype,ghost_complex,ret,CRS_stringify_tmpl,mat,str,dense);

    return ret;
}

