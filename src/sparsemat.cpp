#include "ghost/config.h"
#ifdef GHOST_HAVE_MPI
#include <mpi.h>
#endif

#include <map>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/sparsemat.h"

using namespace std;
static map<ghost_sparsemat_t *,map<ghost_lidx_t, ghost_gidx_t> > rowlengths;
static vector<ghost_gidx_t>lowerPerc90Dists;
static vector<ghost_gidx_t>upperPerc90Dists;

ghost_error_t ghost_sparsemat_registerrow(ghost_sparsemat_t *mat, ghost_gidx_t row, ghost_gidx_t *cols, ghost_lidx_t rowlen, ghost_lidx_t stride)
{
    ghost_lidx_t c;
    ghost_gidx_t col;
    ghost_gidx_t firstcol = mat->ncols-1, lastcol = 0;
    vector<ghost_gidx_t>lowerDists;
    vector<ghost_gidx_t>upperDists;
    ghost_gidx_t lowerDistsAcc = 0, upperDistsAcc = 0;
    ghost_gidx_t lowerEnts = 0, upperEnts = 0;


    for (c=0; c<rowlen; c++) {
        col = cols[c*stride];
        if (col < row) {
            mat->lowerBandwidth = MAX(mat->lowerBandwidth, row-col);
            lowerDists.push_back(row-col);
            lowerDistsAcc += row-col;
            lowerEnts++;
#ifdef GHOST_GATHER_SPARSEMAT_GLOBAL_STATISTICS
            mat->nzDist[mat->context->gnrows-1-(row-col)]++;
#endif
        } else if (col > row) {
            mat->upperBandwidth = MAX(mat->upperBandwidth, col-row);
            upperDists.push_back(col-row);
            upperDistsAcc += col-row;
            upperEnts++;
#ifdef GHOST_GATHER_SPARSEMAT_GLOBAL_STATISTICS
            mat->nzDist[mat->context->gnrows-1+col-row]++;
#endif
        } else {
            lowerDists.push_back(0);
            upperDists.push_back(0);
#ifdef GHOST_GATHER_SPARSEMAT_GLOBAL_STATISTICS
            mat->nzDist[mat->context->gnrows-1]++;
#endif
        }
        firstcol = MIN(col,firstcol);
        lastcol = MAX(col,lastcol);
        
    }

    rowlengths[mat][rowlen]++;
    mat->avgRowBand += lastcol-firstcol+1;
  
    nth_element(upperDists.begin(),upperDists.begin()+int(upperDists.size()*.9),upperDists.end());
    nth_element(lowerDists.begin(),lowerDists.begin()+int(lowerDists.size()*.9),lowerDists.end());
   
    if (upperDists.size()) {
        upperPerc90Dists.push_back(*(upperDists.begin()+int(upperDists.size()*.9)));
    }

    if (lowerDists.size()) {
        lowerPerc90Dists.push_back(*(lowerDists.begin()+int(lowerDists.size()*.9)));
    }
    
    if (lowerEnts) {
        mat->avgAvgRowBand += (double)lowerDistsAcc/lowerEnts;
    }
    if (upperEnts) {
        mat->avgAvgRowBand += (double)upperDistsAcc/upperEnts;
    }
    mat->avgAvgRowBand += 1;

    return GHOST_SUCCESS;
}

ghost_error_t ghost_sparsemat_registerrow_finalize(ghost_sparsemat_t *mat)
{
    ghost_gidx_t gnrows;
    double avgRowlen = mat->nnz*1.0/(double)mat->nrows;

#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->lowerBandwidth,1,ghost_mpi_dt_gidx,MPI_MAX,mat->context->mpicomm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->upperBandwidth,1,ghost_mpi_dt_gidx,MPI_MAX,mat->context->mpicomm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->avgRowBand,1,MPI_DOUBLE,MPI_SUM,mat->context->mpicomm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->avgAvgRowBand,1,MPI_DOUBLE,MPI_SUM,mat->context->mpicomm));
#ifdef GHOST_GATHER_SPARSEMAT_GLOBAL_STATISTICS
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,mat->nzDist,2*mat->context->gnrows-1,ghost_mpi_dt_idx,MPI_SUM,mat->context->mpicomm));
#endif
#endif
    mat->bandwidth = mat->lowerBandwidth+mat->upperBandwidth+1;

    nth_element(upperPerc90Dists.begin(),upperPerc90Dists.begin()+int(upperPerc90Dists.size()*.5),upperPerc90Dists.end());
    nth_element(lowerPerc90Dists.begin(),lowerPerc90Dists.begin()+int(lowerPerc90Dists.size()*.5),lowerPerc90Dists.end());
    
    if (upperPerc90Dists.size() && lowerPerc90Dists.size()) {
        mat->smartRowBand = (double) *(upperPerc90Dists.begin()+int(upperPerc90Dists.size()*.5)) + *(lowerPerc90Dists.begin()+int(lowerPerc90Dists.size()*.5)) + 1;
    }

    ghost_sparsemat_nrows(&gnrows,mat);
    mat->avgRowBand /= (double)(gnrows);
    mat->avgAvgRowBand /= (double)(gnrows);

    rowlengths[mat].erase(0); // erase padded rows
    
    mat->variance = 0.;
    mat->deviation = 0.;
    for (map<ghost_lidx_t,ghost_gidx_t>::const_iterator it = rowlengths[mat].begin(); it != rowlengths[mat].end(); it++) {
        mat->variance += (it->first-avgRowlen)*(it->first-avgRowlen)*it->second;
    }
    mat->variance /= mat->nrows;
    mat->deviation = sqrt(mat->variance);
    mat->cv = mat->deviation*1./(mat->nnz*1.0/(double)mat->nrows);

    mat->nMaxRows = rowlengths[mat][mat->maxRowLen];
    
    return GHOST_SUCCESS;
}

