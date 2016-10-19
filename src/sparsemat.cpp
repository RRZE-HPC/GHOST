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
#include <sstream>
#include <iomanip>
#include <complex>
#include <vector>

#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/sparsemat.h"

using namespace std;
static map<ghost_sparsemat *,map<ghost_lidx, ghost_gidx> > rowlengths;
static vector<ghost_gidx>lowerPerc90Dists;
static vector<ghost_gidx>upperPerc90Dists;

ghost_error ghost_sparsemat_registerrow(ghost_sparsemat *mat, ghost_gidx row, ghost_gidx *cols, ghost_lidx rowlen, ghost_lidx stride)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL); 
    
    ghost_lidx c;
    ghost_gidx col;
    ghost_gidx firstcol = SPM_NCOLS(mat)-1, lastcol = 0;
    vector<ghost_gidx>lowerDists;
    vector<ghost_gidx>upperDists;
    ghost_gidx lowerDistsAcc = 0, upperDistsAcc = 0;
    ghost_gidx lowerEnts = 0, upperEnts = 0;


    for (c=0; c<rowlen; c++) {
        col = cols[c*stride];
        if (col < row) {
            mat->context->lowerBandwidth = MAX(mat->context->lowerBandwidth, row-col);
            lowerDists.push_back(row-col);
            lowerDistsAcc += row-col;
            lowerEnts++;
#ifdef GHOST_SPARSEMAT_GLOBALSTATS
            mat->nzDist[mat->context->row_map->gdim-1-(row-col)]++;
#endif
        } else if (col > row) {
            mat->context->upperBandwidth = MAX(mat->context->upperBandwidth, col-row);
            upperDists.push_back(col-row);
            upperDistsAcc += col-row;
            upperEnts++;
#ifdef GHOST_SPARSEMAT_GLOBALSTATS
            mat->nzDist[mat->context->row_map->gdim-1+col-row]++;
#endif
        } else {
            lowerDists.push_back(0);
            upperDists.push_back(0);
#ifdef GHOST_SPARSEMAT_GLOBALSTATS
            mat->nzDist[mat->context->row_map->gdim-1]++;
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

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL); 
    return GHOST_SUCCESS;
}

ghost_error ghost_sparsemat_registerrow_finalize(ghost_sparsemat *mat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL); 

    ghost_gidx gnrows;
    double avgRowlen = SPM_NNZ(mat)*1.0/(double)SPM_NROWS(mat);

#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->context->lowerBandwidth,1,ghost_mpi_dt_gidx,MPI_MAX,mat->context->mpicomm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->context->upperBandwidth,1,ghost_mpi_dt_gidx,MPI_MAX,mat->context->mpicomm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->avgRowBand,1,MPI_DOUBLE,MPI_SUM,mat->context->mpicomm));
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,&mat->avgAvgRowBand,1,MPI_DOUBLE,MPI_SUM,mat->context->mpicomm));
#ifdef GHOST_SPARSEMAT_GLOBALSTATS
    MPI_CALL_RETURN(MPI_Allreduce(MPI_IN_PLACE,mat->nzDist,2*mat->context->row_map->gdim-1,ghost_mpi_dt_idx,MPI_SUM,mat->context->mpicomm));
#endif
#endif
    mat->context->bandwidth = mat->context->lowerBandwidth+mat->context->upperBandwidth+1;

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
    for (map<ghost_lidx,ghost_gidx>::const_iterator it = rowlengths[mat].begin(); it != rowlengths[mat].end(); it++) {
        mat->variance += (it->first-avgRowlen)*(it->first-avgRowlen)*it->second;
    }
    mat->variance /= SPM_NROWS(mat);
    mat->deviation = sqrt(mat->variance);
    mat->cv = mat->deviation*1./(SPM_NNZ(mat)*1.0/(double)SPM_NROWS(mat));

    mat->nMaxRows = rowlengths[mat][mat->maxRowLen];
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL); 
    return GHOST_SUCCESS;
}

template <typename m_t> 
static ghost_error ghost_sparsemat_string_tmpl(ghost_sparsemat *mat, char **str, int dense)
{
    if (!dense) {
        WARNING_LOG("Sparse printing currently not available.");
        dense = 1;
    }


    int nranks;
    ghost_nrank(&nranks,mat->context->mpicomm);

    if (!(mat->traits.flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS) && nranks > 1) {
        ERROR_LOG("Cannot print compress sparse matrix without original column indices!");
        return GHOST_ERR_INVALID_ARG;
    }

    if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
	    INFO_LOG("Original matrix without permutation is printed, since GHOST_PERM_NO_DISTINCTION is on");
    }

    ghost_lidx chunk,i,j,row=0;
    m_t *val = (m_t *)mat->val;

    stringstream buffer;
    buffer << std::setprecision(2)
           << std::right
           << std::scientific;

    for (chunk = 0; chunk < SPM_NCHUNKS(mat); chunk++) {
        for (i=0; i<mat->traits.C && row<SPM_NROWS(mat); i++, row++) {
            ghost_lidx rowOffs = mat->chunkStart[chunk]+i;
            std::vector<std::string> rowStrings(SPM_GNCOLS(mat));

            for (j=0; j<SPM_GNCOLS(mat); j++) {
                rowStrings[j] = ".         ";
            }
            
            for (j=0; j<mat->rowLen[row]; j++) {
                std::ostringstream strs;
                strs << std::setprecision(2)
                     << std::right
                     << std::scientific;
                strs << val[rowOffs+j*mat->traits.C] << "  ";
                if (mat->traits.flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS) {
                    rowStrings[mat->col_orig[rowOffs+j*mat->traits.C]] = strs.str();
                } else {
                    rowStrings[mat->col[rowOffs+j*mat->traits.C]] = strs.str();
                }
            }
           
            if (nranks > 1) { 
                buffer << " | ";
                
                for (j=0; j<mat->context->row_map->offs; j++) {
                    buffer << rowStrings[j];
                }
                
                buffer << " | ";
                
                for (; j<mat->context->row_map->offs+SPM_NROWS(mat); j++) {
                    buffer << rowStrings[j];
                }
                
                buffer << " | ";
                
                for (; j<SPM_GNCOLS(mat); j++) {
                    buffer << rowStrings[j];
                }
                
                buffer << " | ";
            } else {
                for (j=0; j<SPM_GNCOLS(mat); j++) {
                    buffer << rowStrings[j];
                }
            }

            if (i<SPM_NROWS(mat)-1) {
                buffer << endl;
            }
        }
    }

    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    return GHOST_SUCCESS;
}

extern "C" ghost_error ghost_sparsemat_string(char **str, ghost_sparsemat *mat, int dense)
{
    ghost_error ret;

    SELECT_TMPL_1DATATYPE(mat->traits.datatype,std::complex,ret,ghost_sparsemat_string_tmpl,mat,str,dense);

    return ret;
}
