
#include <mpi.h>
#include <map>

#include <cmath>


#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/sparsemat.h"

using namespace std;
static map<ghost_sparsemat_t *,map<ghost_idx_t, ghost_idx_t> > rowlengths;

ghost_error_t ghost_sparsemat_registerrow(ghost_sparsemat_t *mat, ghost_idx_t row, ghost_idx_t *cols, ghost_idx_t rowlen, ghost_idx_t stride)
{
    ghost_idx_t c, col;

    for (c=0; c<rowlen; c++) {
        col = cols[c*stride];
        if (col < row) {
            mat->lowerBandwidth = MAX(mat->lowerBandwidth, row-col);
#ifdef GHOST_GATHER_GLOBAL_INFO
            mat->nzDist[mat->context->gnrows-1-(row-col)]++;
#endif
        } else if (col > row) {
            mat->upperBandwidth = MAX(mat->upperBandwidth, col-row);
#ifdef GHOST_GATHER_GLOBAL_INFO
            mat->nzDist[mat->context->gnrows-1+col-row]++;
#endif
        } else {
#ifdef GHOST_GATHER_GLOBAL_INFO
            mat->nzDist[mat->context->gnrows-1]++;
#endif
        }
    }

    mat->maxRowLen = MAX(mat->maxRowLen,rowlen);
    rowlengths[mat][rowlen]++;
    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_registerrow_finalize(ghost_sparsemat_t *mat)
{
    double avgRowlen = mat->nnz*1.0/(double)mat->nrows;

#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->lowerBandwidth,1,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->upperBandwidth,1,ghost_mpi_dt_idx,MPI_MAX,mat->context->mpicomm));
#ifdef GHOST_GATHER_GLOBAL_INFO
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,mat->nzDist,2*mat->context->gnrows-1,ghost_mpi_dt_idx,MPI_SUM,mat->context->mpicomm));
#endif
#endif
    mat->bandwidth = mat->lowerBandwidth+mat->upperBandwidth+1;

    rowlengths[mat].erase(0); // erase padded rows
    
    mat->variance = 0.;
    mat->deviation = 0.;
    for (map<ghost_idx_t,ghost_idx_t>::const_iterator it = rowlengths[mat].begin(); it != rowlengths[mat].end(); it++) {
        mat->variance += (it->first-avgRowlen)*(it->first-avgRowlen)*it->second;
    }
    mat->variance /= mat->nrows;
    mat->deviation = sqrt(mat->variance);
    mat->cv = mat->deviation*1./(mat->nnz*1.0/(double)mat->nrows);

    mat->nMaxRows = rowlengths[mat][mat->maxRowLen];
    
    return GHOST_SUCCESS;
}

