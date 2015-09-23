#include "ghost/types.h"
#include "ghost/omp.h"

#include "ghost/complex.h"
#include "ghost/util.h"
#include "ghost/sell.h"
#include "ghost/timing.h"

#include <sstream>

using namespace std;

template <typename m_t> 
static ghost_error_t SELL_stringify_tmpl(ghost_sparsemat_t *mat, char **str, int dense)
{
    ghost_lidx_t chunk,i,j,row=0,col;
    m_t *val = (m_t *)SELL(mat)->val;

    stringstream buffer;

    for (chunk = 0; chunk < mat->nrowsPadded/SELL(mat)->chunkHeight; chunk++) {
        for (i=0; i<SELL(mat)->chunkHeight && row<mat->nrows; i++, row++) {
            ghost_lidx_t rowOffs = SELL(mat)->chunkStart[chunk]+i;
            if (dense) {
                for (col=0, j=0; col<mat->ncols; col++) {
                    if ((j < SELL(mat)->rowLen[row]) && ((mat->traits->flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)?(mat->col_orig[rowOffs+j*SELL(mat)->chunkHeight] == col):(SELL(mat)->col[rowOffs+j*SELL(mat)->chunkHeight] == col))) { // there is an entry at col
                         buffer << val[rowOffs+j*SELL(mat)->chunkHeight] << "\t";
                        j++;
                    } else {
                        buffer << ".\t";
                    }
                }
            } else {
                for (j=0; j<(dense?mat->ncols:SELL(mat)->chunkLen[chunk]); j++) {
                    ghost_lidx_t idx = SELL(mat)->chunkStart[chunk]+j*SELL(mat)->chunkHeight+i;
                    buffer << val[idx] << " (" << SELL(mat)->col[idx] << ")" << "\t";
                }
            }
            if (i<mat->nrows-1) {
                buffer << endl;
            }
        }
    }

    GHOST_CALL_RETURN(ghost_malloc((void **)str,buffer.str().length()+1));
    strcpy(*str,buffer.str().c_str());

    return GHOST_SUCCESS;
}
extern "C" ghost_error_t ghost_sell_stringify_selector(ghost_sparsemat_t *mat, char **str, int dense)
{
    ghost_error_t ret;

    SELECT_TMPL_1DATATYPE(mat->traits->datatype,ghost_complex,ret,SELL_stringify_tmpl,mat,str,dense);

    return ret;
}

