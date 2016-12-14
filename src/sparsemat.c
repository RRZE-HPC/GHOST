#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/sparsemat.h"
#include "ghost/context.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/log.h"
#include "ghost/machine.h"
#include "ghost/bincrs.h"
#include "ghost/matrixmarket.h"
#include "ghost/instr.h"
#include "ghost/constants.h"
#include "ghost/kacz_hybrid_split.h"
#include "ghost/kacz_split_analytical.h"
#include "ghost/rcm_dissection.h"
#include <libgen.h>
#include <math.h>
#include <limits.h>

const ghost_sparsemat_src_rowfunc GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER = {
    .func = NULL,
    .maxrowlen = 0,
    .base = 0,
    .flags = GHOST_SPARSEMAT_ROWFUNC_DEFAULT,
    .arg = NULL
};
    
const ghost_sparsemat_traits GHOST_SPARSEMAT_TRAITS_INITIALIZER = {
    .flags = GHOST_SPARSEMAT_DEFAULT,
    .symmetry = GHOST_SPARSEMAT_SYMM_GENERAL,
    .T = 1,
    .C = 32,
    .scotchStrat = (char*)GHOST_SCOTCH_STRAT_DEFAULT,
    .sortScope = 1,
    .datatype = GHOST_DT_NONE,
    .opt_blockvec_width = 0
};

static const char * SELL_formatName(ghost_sparsemat *mat);
static size_t SELL_byteSize (ghost_sparsemat *mat);
static ghost_error SELL_split(ghost_sparsemat *mat);
static ghost_error SELL_upload(ghost_sparsemat *mat);
static ghost_error SELL_toBinCRS(ghost_sparsemat *mat, char *matrixPath);
static ghost_error SELL_fromRowFunc(ghost_sparsemat *mat, ghost_sparsemat_src_rowfunc *src);

const ghost_spmv_opts GHOST_SPMV_OPTS_INITIALIZER = {
    .flags = GHOST_SPMV_DEFAULT,
    .alpha = NULL,
    .beta = NULL,
    .gamma = NULL,
    .delta = NULL,
    .eta = NULL,
    .dot = NULL,
    .z = NULL
};

ghost_error ghost_sparsemat_create(ghost_sparsemat ** mat, ghost_context *context, ghost_sparsemat_traits *traits, int nTraits)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    UNUSED(nTraits);
    ghost_error ret = GHOST_SUCCESS;

    int me;
    GHOST_CALL_GOTO(ghost_rank(&me, context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)mat,sizeof(ghost_sparsemat)),err,ret);

    (*mat)->traits = traits[0];
    if (nTraits == 3) {
        (*mat)->splittraits[0] = traits[1];
        (*mat)->splittraits[1] = traits[2];
    } else {
        (*mat)->splittraits[0] = traits[0];
        (*mat)->splittraits[1] = traits[0];
    }

    (*mat)->context = context;
    (*mat)->localPart = NULL;
    (*mat)->remotePart = NULL;
    (*mat)->name = "Sparse matrix";
    (*mat)->col_orig = NULL;
    (*mat)->sell = NULL;
    (*mat)->nzDist = NULL;
    (*mat)->fromFile = &ghost_sparsemat_from_bincrs;
    (*mat)->fromMM = &ghost_sparsemat_from_mm;
    (*mat)->fromCRS = &ghost_sparsemat_from_crs;
    (*mat)->formatName = NULL;
    (*mat)->upload = NULL;
    (*mat)->bandwidth = 0;
    (*mat)->lowerBandwidth = 0;
    (*mat)->upperBandwidth = 0;
    (*mat)->avgRowBand = 0.;
    (*mat)->avgAvgRowBand = 0.;
    (*mat)->smartRowBand = 0.;
    (*mat)->maxRowLen = 0;
    (*mat)->nMaxRows = 0;
    (*mat)->variance = 0.;
    (*mat)->deviation = 0.;
    (*mat)->cv = 0.;
    (*mat)->nrows = context->lnrows[me];
    (*mat)->nrowsPadded = (*mat)->nrows;
    (*mat)->ncols = context->gncols;
    (*mat)->nEnts = 0;
    (*mat)->nnz = 0;
    (*mat)->ncolors = 0;
    (*mat)->color_ptr = NULL;
    (*mat)->nzones = 0;
    (*mat)->zone_ptr = NULL;
    (*mat)->kacz_setting.kacz_method = MC;//fallback
    (*mat)->kacz_setting.active_threads = 0;

    if ((*mat)->traits.sortScope == GHOST_SPARSEMAT_SORT_GLOBAL) {
        (*mat)->traits.sortScope = (*mat)->context->gnrows;
    } else if ((*mat)->traits.sortScope == GHOST_SPARSEMAT_SORT_LOCAL) {
        (*mat)->traits.sortScope = (*mat)->nrows;
    }

#ifdef GHOST_SPARSEMAT_GLOBALSTATS
    GHOST_CALL_GOTO(ghost_malloc((void **)&((*mat)->nzDist),sizeof(ghost_gidx)*(2*context->gnrows-1)),err,ret);
#endif

    // Note: Datatpye check and elSize computation moved to creation
    // functions ghost_sparsemat_from_* and SELL_fromRowFunc.
    (*mat)->elSize = 0;

    GHOST_CALL_GOTO(ghost_malloc((void **)&(*mat)->sell,sizeof(ghost_sell)),err,ret);
    DEBUG_LOG(1,"Setting functions for SELL matrix");
    if (!((*mat)->traits.flags & (GHOST_SPARSEMAT_HOST | GHOST_SPARSEMAT_DEVICE)))
    { // no placement specified
        DEBUG_LOG(2,"Setting matrix placement");
        ghost_type ghost_type;
        GHOST_CALL_GOTO(ghost_type_get(&ghost_type),err,ret);
        if (ghost_type == GHOST_TYPE_CUDA) {
            (*mat)->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_DEVICE;
        } else {
            (*mat)->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_HOST;
        }
    }
    ghost_type ghost_type;
    GHOST_CALL_RETURN(ghost_type_get(&ghost_type));

    (*mat)->upload = &SELL_upload;
    (*mat)->toFile = &SELL_toBinCRS;
    (*mat)->fromRowFunc = &SELL_fromRowFunc;
    (*mat)->formatName = &SELL_formatName;
    (*mat)->byteSize   = &SELL_byteSize;
    (*mat)->spmv     = &ghost_sell_spmv_selector;
    (*mat)->kacz     = &ghost_sell_kacz_selector;
    (*mat)->kacz_shift   = &ghost_sell_kacz_shift_selector;
    (*mat)->string    = &ghost_sell_stringify_selector;
    (*mat)->split = &SELL_split;
#ifdef GHOST_HAVE_CUDA
    if ((ghost_type == GHOST_TYPE_CUDA) && ((*mat)->traits.flags & GHOST_SPARSEMAT_DEVICE)) {
        (*mat)->spmv   = &ghost_cu_sell_spmv_selector;
    }
#endif

    (*mat)->sell->val = NULL;
    (*mat)->sell->col = NULL;
    (*mat)->sell->chunkMin = NULL;
    (*mat)->sell->chunkLen = NULL;
    (*mat)->sell->chunkLenPadded = NULL;
    (*mat)->sell->rowLen = NULL;
    (*mat)->sell->rowLen2 = NULL;
    (*mat)->sell->rowLen4 = NULL;
    (*mat)->sell->rowLenPadded = NULL;
    (*mat)->sell->chunkStart = NULL;
    (*mat)->sell->cumat = NULL;

    if ((*mat)->traits.C == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
        (*mat)->traits.C = PAD((*mat)->nrows,GHOST_PAD_MAX);
    } else if ((*mat)->traits.C == GHOST_SELL_CHUNKHEIGHT_AUTO){
        (*mat)->traits.C = 32; // TODO
    }
    (*mat)->nrowsPadded = PAD((*mat)->nrows,(*mat)->traits.C);

    goto out;
err:
    ERROR_LOG("Error. Free'ing resources");
    free(*mat); *mat = NULL;
    free((*mat)->sell); (*mat)->sell = NULL;

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;    
}

ghost_error ghost_sparsemat_sortrow(ghost_gidx *col, char *val, size_t valSize, ghost_lidx rowlen, ghost_lidx stride)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    ghost_lidx n;
    ghost_lidx c;
    ghost_lidx swpcol;
    char swpval[valSize];
    for (n=rowlen; n>1; n--) {
        for (c=0; c<n-1; c++) {
            if (col[c*stride] > col[(c+1)*stride]) {
                swpcol = col[c*stride];
                col[c*stride] = col[(c+1)*stride];
                col[(c+1)*stride] = swpcol; 

                memcpy(&swpval,&val[c*stride*valSize],valSize);
                memcpy(&val[c*stride*valSize],&val[(c+1)*stride*valSize],valSize);
                memcpy(&val[(c+1)*stride*valSize],&swpval,valSize);
            }
        }
    }

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return GHOST_SUCCESS;
}


//calculates bandwidth of the matrix
ghost_error calculate_bw(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType) {
     GHOST_INSTR_START("calculate badwidth");
     ghost_error ret = GHOST_SUCCESS;
     int me;     
     GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);
 
     if (srcType == GHOST_SPARSEMAT_SRC_FUNC || srcType == GHOST_SPARSEMAT_SRC_FILE) {
       ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
       ghost_gidx * tmpcol = NULL;
       char * tmpval = NULL;     
       ghost_lidx rowlen;
       ghost_gidx lower_bw = 0, upper_bw = 0, max_col=0;
  
#pragma omp parallel private(tmpval,tmpcol,rowlen) 
  {
       ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));
       ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize); 
  #pragma omp for reduction(max:lower_bw) reduction(max:upper_bw) reduction(max:max_col)
       for (int i=0; i<mat->context->lnrows[me]; i++) {
            if (mat->context->perm_global && mat->context->perm_local) {
                    src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[i]],&rowlen,tmpcol,tmpval,src->arg);
                } else if (mat->context->perm_global) {
                    src->func(mat->context->perm_global->invPerm[i],&rowlen,tmpcol,tmpval,src->arg);
                } else if (mat->context->perm_local) {
                    src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[i],&rowlen,tmpcol,tmpval,src->arg);
                } else {
                    src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,src->arg);
                }

                ghost_gidx start_col = INT_MAX;
                ghost_gidx end_col   = 0;

                if(mat->context->perm_local){
            if(mat->context->perm_local->colPerm == NULL) {
                            for(int j=0; j<rowlen; ++j) {
                                    start_col = MIN(start_col, mat->context->perm_local->perm[tmpcol[j]]);
                                    end_col   = MAX(end_col, mat->context->perm_local->perm[tmpcol[j]]);
                            }
                    } else {
                            for(int j=0; j<rowlen; ++j) {
                                    start_col = MIN(start_col, mat->context->perm_local->colPerm[tmpcol[j]]);
                                    end_col   = MAX(end_col, mat->context->perm_local->colPerm[tmpcol[j]]);
                }
                    }
        } else {
                        for(int j=0; j<rowlen; ++j) {
                                    start_col = MIN(start_col, tmpcol[j]);
                                    end_col   = MAX(end_col, tmpcol[j]);
                            }
            }
                lower_bw = MAX(lower_bw, i-start_col);
                upper_bw = MAX(upper_bw, end_col - i);
                max_col    = MAX(max_col, end_col);
        }
        free(tmpcol);
    free(tmpval);
    }
    mat->lowerBandwidth = lower_bw;
    mat->upperBandwidth = upper_bw;
    mat->bandwidth      = lower_bw + upper_bw;
    mat->maxColRange    = max_col;
 
    mat->bandwidth = mat->lowerBandwidth + mat->upperBandwidth;
    INFO_LOG("RANK<%d>:  LOWER BANDWIDTH =%"PRGIDX", UPPER BANDWIDTH =%"PRGIDX", TOTAL BANDWIDTH =%"PRGIDX,me,mat->lowerBandwidth,mat->upperBandwidth,mat->bandwidth);
    GHOST_INSTR_STOP("calculate bandwidth");
    goto out;
  } else {
     goto err;
  }

err: 
   ERROR_LOG("ERROR in Bandwidth Calculation");
   return ret;
out:
   return ret;
}

ghost_error set_kacz_ratio(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType) 
{
  int *nthread = (int*) malloc(sizeof(int));  

#ifdef GHOST_HAVE_OPENMP
#pragma omp parallel
      {
       #pragma omp master
            nthread[0] = ghost_omp_nthread();
      }
#else
        nthread[0] = 1;
#endif
   
   mat->kacz_setting.active_threads = nthread[0];
   calculate_bw(mat,matrixSource,srcType);
   mat->kaczRatio = ((double)mat->nrows)/mat->bandwidth;
   free(nthread);
   return GHOST_SUCCESS;
}

ghost_error ghost_sparsemat_fromfunc_common_dummy(ghost_lidx *rl, ghost_lidx *rlp, ghost_lidx *cl, ghost_lidx *clp, ghost_lidx **chunkptr, char **val, ghost_gidx **col, ghost_sparsemat_src_rowfunc *src, ghost_sparsemat *mat, ghost_lidx C, ghost_lidx P)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    ghost_error ret = GHOST_SUCCESS;
    int funcerrs = 0;
    char *tmpval = NULL;
    ghost_gidx *tmpcol = NULL;
    ghost_lidx nchunks = (ghost_lidx)(ceil((double)mat->nrows/(double)C));
    ghost_lidx i,row,chunk,colidx;
    ghost_gidx gnents = 0, gnnz = 0;
    ghost_lidx maxRowLenInChunk = 0, maxRowLen = 0, privateMaxRowLen = 0;
    int me,nprocs;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
 
   
    mat->ncols = mat->context->gncols;
    mat->nrows = mat->context->lnrows[me];

#ifdef GHOST_SPARSEMAT_GLOBALSTATS
    memset(mat->nzDist,0,sizeof(ghost_gidx)*(2*mat->context->gnrows-1));
#endif
    mat->lowerBandwidth = 0;
    mat->upperBandwidth = 0;
    
    if (mat->traits.flags & GHOST_SPARSEMAT_SCOTCHIFY) {
        mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_PERMUTE;
    }

    if (0) {
        if (mat->traits.flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_sparsemat_perm_scotch(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } 
        if (mat->traits.flags & GHOST_SPARSEMAT_ZOLTAN) {
            ghost_sparsemat_perm_zoltan(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } 
        if (mat->traits.flags & GHOST_SPARSEMAT_RCM) { 
            ghost_sparsemat_perm_spmp(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } 
        if (mat->traits.flags & GHOST_SPARSEMAT_COLOR) {
            ghost_sparsemat_perm_color(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        }
        if (mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) {
            ghost_sparsemat_blockColor(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        }
        if (mat->traits.sortScope > 1) {
            ghost_sparsemat_perm_sort(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC,mat->traits.sortScope);
        }
        if ( mat->context->perm_local && mat->context->perm_local->colPerm == NULL) {
            mat->context->perm_local->colPerm = mat->context->perm_local->perm;
            mat->context->perm_local->colInvPerm = mat->context->perm_local->invPerm;
        }
        if (mat->context->perm_global && mat->context->perm_global->colPerm == NULL) {
            mat->context->perm_global->colPerm = mat->context->perm_global->perm;
            mat->context->perm_global->colInvPerm = mat->context->perm_global->invPerm;
        }
        if (mat->traits.flags & GHOST_SPARSEMAT_NOT_SORT_COLS) {
            PERFWARNING_LOG("Unsorted columns inside a row may yield to bad performance! However, matrix construnction will be faster.");
        }
    } else {
        if (mat->traits.sortScope > 1) {
            WARNING_LOG("Ignoring sorting scope");
        }
//        mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_NOT_PERMUTE_COLS;
//        mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_NOT_SORT_COLS;
    }

    ghost_lidx *tmpclp = NULL;
    if (!clp) {
        ghost_malloc((void **)&tmpclp,nchunks*sizeof(ghost_lidx));
        clp = tmpclp;
    }
    ghost_lidx *tmprl = NULL;
    if (!rl) {
        ghost_malloc((void **)&tmprl,nchunks*sizeof(ghost_lidx));
        rl = tmprl;
    }


    if (!(*chunkptr)) {
        GHOST_INSTR_START("rowlens");
        GHOST_CALL_GOTO(ghost_malloc_align((void **)chunkptr,(nchunks+1)*sizeof(ghost_lidx),GHOST_DATA_ALIGNMENT),err,ret);


#pragma omp parallel private(i,tmpval,tmpcol,row,maxRowLenInChunk) reduction (+:gnents,gnnz,funcerrs) reduction (max:privateMaxRowLen) 
        {
            ghost_lidx rowlen;
            maxRowLenInChunk = 0; 
            GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx)),ret);

            /*if (!(mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) && src->func == ghost_sparsemat_rowfunc_crs) {
#pragma omp single
                INFO_LOG("Fast matrix construction for CRS source and no permutation") 
#pragma omp for schedule(runtime)
                for( chunk = 0; chunk < nchunks; chunk++ ) {
                    chunkptr[chunk] = 0; // NUMA init
                    for (i=0, row = chunk*C; i < C && row < mat->nrows; i++, row++) {

                        rowlen=((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt[mat->context->lfRow[me]+row+1]-((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt[mat->context->lfRow[me]+row];

                        // rl _must_ not be NULL because we need it for the statistics
                        rl[row] = rowlen;
                        
                        if (rlp) {
                            rlp[row] = PAD(rowlen,P);
                        }

                        gnnz += rowlen;
                        maxRowLenInChunk = MAX(maxRowLenInChunk,rowlen);
                    }
                    if (cl) {
                        cl[chunk] = maxRowLenInChunk;
                    }

                    // clp _must_ not be NULL because we need it for the chunkptr computation
                    clp[chunk] = PAD(maxRowLenInChunk,P);

                    gnents += clp[chunk]*C;

                    privateMaxRowLen = MAX(privateMaxRowLen,maxRowLenInChunk);
                    maxRowLenInChunk = 0;
                }
            } else {*/
#pragma omp for schedule(runtime)
                for( chunk = 0; chunk < nchunks; chunk++ ) {
                    (*chunkptr)[chunk] = 0; // NUMA init
                    for (i=0, row = chunk*C; (i < C) && (row < mat->nrows); i++, row++) {

                    if (0) {//mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                      if (mat->context->perm_global && mat->context->perm_local) {
                        INFO_LOG("Global _and_ local permutation");
                            funcerrs += src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[row]],&rowlen,tmpcol,tmpval,src->arg);
                      } else if (mat->context->perm_global) {
                            funcerrs += src->func(mat->context->perm_global->invPerm[row],&rowlen,tmpcol,tmpval,src->arg);
                      } else if (mat->context->perm_local) {
                            funcerrs += src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[row],&rowlen,tmpcol,tmpval,src->arg);
                      }
                    } else {
                          funcerrs += src->func(mat->context->lfRow[me]+row,&rowlen,tmpcol,tmpval,src->arg);
                    }


                    // rl _must_ not be NULL because we need it for the statistics
                    rl[row] = rowlen;
                    
                    if (rlp) {
                        rlp[row] = PAD(rowlen,P);
                    }

                    gnnz += rowlen;
                    maxRowLenInChunk = MAX(maxRowLenInChunk,rowlen);
                    }
                    if (cl) {
                        cl[chunk] = maxRowLenInChunk;
                    }

                    // clp _must_ not be NULL because we need it for the chunkptr computation
                    clp[chunk] = PAD(maxRowLenInChunk,P);

                    gnents += clp[chunk]*C;

                    privateMaxRowLen = MAX(privateMaxRowLen,maxRowLenInChunk);
                    maxRowLenInChunk = 0;
                }
            //}


            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        GHOST_INSTR_STOP("rowlens");
        maxRowLen = privateMaxRowLen;
        mat->maxRowLen = maxRowLen;

        if (funcerrs) {
            ERROR_LOG("Matrix construction function returned error");
            ret = GHOST_ERR_UNKNOWN;
            goto err;
        }
        if (gnents > (ghost_gidx)GHOST_LIDX_MAX) {
            ERROR_LOG("The local number of entries is too large: %"PRGIDX,gnents);
            return GHOST_ERR_DATATYPE;
        }
        if (gnnz > (ghost_gidx)GHOST_LIDX_MAX) {
            ERROR_LOG("The local number of entries is too large: %"PRGIDX,gnents);
            return GHOST_ERR_DATATYPE;
        }

        mat->nnz = (ghost_lidx)gnnz;
        mat->nEnts = (ghost_lidx)gnents;

        GHOST_INSTR_START("chunkptr_init");
        for(chunk = 0; chunk < nchunks; chunk++ ) {
            (*chunkptr)[chunk+1] = (*chunkptr)[chunk] + clp[chunk]*C;
        }
        GHOST_INSTR_STOP("chunkptr_init");
        

#ifdef GHOST_HAVE_MPI
        ghost_gidx fent = 0;
        for (i=0; i<nprocs; i++) {
            if (i>0 && me==i) {
                MPI_CALL_GOTO(MPI_Recv(&fent,1,ghost_mpi_dt_gidx,me-1,me-1,mat->context->mpicomm,MPI_STATUS_IGNORE),err,ret);
            }
            if (me==i && i<nprocs-1) {
                ghost_gidx send = fent+mat->nEnts;
                MPI_CALL_GOTO(MPI_Send(&send,1,ghost_mpi_dt_gidx,me+1,me,mat->context->mpicomm),err,ret);
            }
        }
        
        MPI_CALL_GOTO(MPI_Allgather(&mat->nEnts,1,ghost_mpi_dt_lidx,mat->context->lnEnts,1,ghost_mpi_dt_lidx,mat->context->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Allgather(&fent,1,ghost_mpi_dt_gidx,mat->context->lfEnt,1,ghost_mpi_dt_gidx,mat->context->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Allreduce(&gnnz,&mat->context->gnnz,1,ghost_mpi_dt_gidx,MPI_SUM,mat->context->mpicomm),err,ret);
#endif
    }
    if (src->maxrowlen != mat->maxRowLen) {
        DEBUG_LOG(1,"The maximum row length was not correct. Setting it from %"PRLIDX" to %"PRGIDX,src->maxrowlen,mat->maxRowLen); 
        src->maxrowlen = mat->maxRowLen;
    }

   
    bool readcols = 0; // we only need to read the columns the first time the matrix is created
    if (!(*val)) {
        GHOST_CALL_GOTO(ghost_malloc_align((void **)val,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
    }
    
    if (!(*col)) {
        GHOST_CALL_GOTO(ghost_malloc_align((void **)col,sizeof(ghost_gidx)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
        readcols = 1;
    }

        
    if (src->func == ghost_sparsemat_rowfunc_crs && mat->context->perm_global) {
        ERROR_LOG("Global permutation does not work with local CRS source");
    }

    GHOST_INSTR_START("cols_and_vals");
#pragma omp parallel private(i,colidx,row,tmpval,tmpcol)
    {
        int funcret = 0;
        GHOST_CALL(ghost_malloc((void **)&tmpval,C*src->maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,C*src->maxrowlen*sizeof(ghost_gidx)),ret);
        
        if (src->func == ghost_sparsemat_rowfunc_crs) {
            ghost_gidx *crscol;
            char *crsval = (char *)(((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->val);
            ghost_lidx *crsrpt = ((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt;
#pragma omp single
            INFO_LOG("Fast matrix construction for CRS source and no permutation");

#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nchunks; chunk++ ) {
                //memset(tmpval,0,mat->elSize*src->maxrowlen*C);

                for (i=0, row = chunk*C; (i<C) && (chunk*C+i < mat->nrows); i++, row++) {
                    ghost_gidx actualrow;
                    if (0) {//mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                        actualrow = mat->context->perm_local->invPerm[row];
                    } else {
                        actualrow = row;
                    }
                    
                    crsval = &((char *)(((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->val))[crsrpt[actualrow]*mat->elSize];

#pragma vector nontemporal
                    for (colidx = 0; colidx<rl[row]; colidx++) {
                        // assignment is much faster than memcpy with non-constant size, so we need those branches...
                        if (mat->traits.datatype & GHOST_DT_REAL) {
                            if (mat->traits.datatype & GHOST_DT_DOUBLE) {
                                ((double *)(*val))[(*chunkptr)[chunk]+colidx*C+i] = ((double *)(crsval))[colidx];
                            } else {
                                ((float *)(*val))[(*chunkptr)[chunk]+colidx*C+i] = ((float *)(crsval))[colidx];
                            }
                        } else {
                            if (mat->traits.datatype & GHOST_DT_DOUBLE) {
                                ((complex double *)(*val))[(*chunkptr)[chunk]+colidx*C+i] = ((complex double *)(crsval))[colidx];
                            } else {
                                ((complex float *)(*val))[(*chunkptr)[chunk]+colidx*C+i] = ((complex float *)(crsval))[colidx];
                            }
                        }
                        if (readcols) {
                            crscol = &((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->col[crsrpt[actualrow]];
                            if (0){//mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                                // local permutation: distinction between global and local entriess, if GHOST_PERM_NO_DISTINCTION is not set 
                                if ((mat->context->flags & GHOST_PERM_NO_DISTINCTION) || ( (crscol[colidx] >= mat->context->lfRow[me]) && (crscol[colidx] < (mat->context->lfRow[me]+mat->ncols)) )) { // local entry: copy with permutation
                                    if (mat->traits.flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = crscol[colidx];
                                    } else if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[crscol[colidx]];
                                    } else {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[crscol[colidx]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                                    }

                                } else { // remote entry: copy without permutation
                                    (*col)[(*chunkptr)[chunk]+colidx*C+i] = crscol[colidx];
                                }
                            } else {
                                (*col)[(*chunkptr)[chunk]+colidx*C+i] = crscol[colidx];
                            }
                        }
                    }
                     for (; colidx < clp[chunk]; colidx++) {
                        memset(&(*val)[((*chunkptr)[chunk]+colidx*C+i)*mat->elSize],0,mat->elSize);
                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->lfRow[me];
                    }

                }
            }
        } else {
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nchunks; chunk++ ) {
                memset(tmpval,0,mat->elSize*src->maxrowlen*C);
                for (i=0; i<src->maxrowlen*C; i++) {
                    tmpcol[i] = mat->context->lfRow[me];
                }

                for (i=0, row = chunk*C; (i<C) && (chunk*C+i < mat->nrows); i++, row++) {

                    if (0) {//mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                        if (mat->context->perm_global && mat->context->perm_local) {
                            funcret = src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[row]],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize],src->arg);
                        } else if (mat->context->perm_global) {
                            funcret = src->func(mat->context->perm_global->invPerm[row],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize],src->arg);
                        } else if (mat->context->perm_local) {
                            funcret = src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[row],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize],src->arg);
                        }
                    } else {
                        funcret = src->func(mat->context->lfRow[me]+row,&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize],src->arg);
                    }
                    if (funcret) {
                        ERROR_LOG("Matrix construction function returned error");
                        ret = GHOST_ERR_UNKNOWN;
                    }

                    
                    for (colidx = 0; colidx<clp[chunk]; colidx++) {
                        memcpy(*val+mat->elSize*((*chunkptr)[chunk]+colidx*C+i),&tmpval[mat->elSize*(i*src->maxrowlen+colidx)],mat->elSize);
                        if (mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                            if (0){//mat->context->perm_global && !mat->context->perm_local) {
                                // no distinction between global and local entries
                                // global permutation will be done after all rows are read
                                (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                            } else { 
                                // local permutation: distinction between global and local entries, if GHOST_PERM_NO_DISTINCTION is not set 
                                if (0){//(mat->context->perm_local->flags & GHOST_PERM_NO_DISTINCTION) ||(tmpcol[i*src->maxrowlen+colidx] >= mat->context->lfRow[me]) && (tmpcol[i*src->maxrowlen+colidx] < (mat->context->lfRow[me]+mat->ncols))) { // local entry: copy with permutation
                                    if (mat->traits.flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                                    }else if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[tmpcol[i*src->maxrowlen+colidx]];
                                    } else {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[tmpcol[i*src->maxrowlen+colidx]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                                    }
                                } else { // remote entry: copy without permutation
                                    (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                                }
                            }
                        } else {
                            (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                        }
                    }
                }
            }
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
    if (mat->nrows % C) {
        for (i=mat->nrows%C; i < C; i++) {
            for (colidx = 0; colidx<clp[nchunks-1]; colidx++) {
                (*col)[(*chunkptr)[nchunks-1]+colidx*C+i] = mat->context->lfRow[me];
                memset(*val+mat->elSize*((*chunkptr)[nchunks-1]+colidx*C+i),0,mat->elSize);
            }
        }
    }

    GHOST_INSTR_STOP("cols_and_vals");
    
    if (mat->context->perm_global) {
        ghost_sparsemat_perm_global_cols(*col,mat->nEnts,mat->context);
    }
    
    GHOST_INSTR_START("sort_and_register");


    if (!(mat->traits.flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
        for( chunk = 0; chunk < nchunks; chunk++ ) {
            for (i=0; (i<C) && (chunk*C+i < mat->nrows); i++) {
                row = chunk*C+i;
                ghost_sparsemat_sortrow(&((*col)[(*chunkptr)[chunk]+i]),&(*val)[((*chunkptr)[chunk]+i)*mat->elSize],mat->elSize,rl[row],C);
#ifdef GHOST_SPARSEMAT_STATS
                ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&(*col)[(*chunkptr)[chunk]+i],rl[row],C);
#endif
            }
        }
    } else {
#ifdef GHOST_SPARSEMAT_STATS
        for( chunk = 0; chunk < nchunks; chunk++ ) {
            for (i=0; (i<C) && (chunk*C+i < mat->nrows); i++) {
                row = chunk*C+i;
                ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&(*col)[(*chunkptr)[chunk]+i],rl[row],C);
            }
        }
#endif
    }

#ifdef GHOST_SPARSEMAT_STATS
    ghost_sparsemat_registerrow_finalize(mat);
#endif
    GHOST_INSTR_STOP("sort_and_register");
    
    mat->context->lnEnts[me] = mat->nEnts;

    for (i=0; i<nprocs; i++) {
        mat->context->lfEnt[i] = 0;
    } 

    for (i=1; i<nprocs; i++) {
        mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
    } 

    free(tmpclp);
    free(tmprl);

    goto out;
err:

out:

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;
}


ghost_error ghost_sparsemat_fromfunc_common(ghost_lidx *rl, ghost_lidx *rlp, ghost_lidx *cl, ghost_lidx *clp, ghost_lidx **chunkptr, char **val, ghost_gidx **col, ghost_sparsemat_src_rowfunc *src, ghost_sparsemat *mat, ghost_lidx C, ghost_lidx P)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    ghost_error ret = GHOST_SUCCESS;
    int funcerrs = 0;
    char *tmpval = NULL;
    ghost_gidx *tmpcol = NULL;
    ghost_lidx nchunks = (ghost_lidx)(ceil((double)mat->nrows/(double)C));
    ghost_lidx i,row,chunk,colidx;
    ghost_gidx gnents = 0, gnnz = 0;
    ghost_lidx maxRowLenInChunk = 0, maxRowLen = 0, privateMaxRowLen = 0;
    int me,nprocs;
    
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
 
  
    if(mat->traits.flags & GHOST_PERM_NO_DISTINCTION) 
        mat->ncols = mat->context->nrowspadded; 
    else 
        mat->ncols = mat->context->gncols;

   mat->nrows = mat->context->lnrows[me];

#ifdef GHOST_SPARSEMAT_GLOBALSTATS
    memset(mat->nzDist,0,sizeof(ghost_gidx)*(2*mat->context->gnrows-1));
#endif
    mat->lowerBandwidth = 0;
    mat->upperBandwidth = 0;
    
    if (mat->traits.flags & GHOST_SPARSEMAT_SCOTCHIFY) {
        mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_PERMUTE;
    }
  //check whether BLOCKCOLOR is necessary, it is avoided if user explicitly request Multicoloring method
    if( (mat->traits.flags && (mat->traits.flags & GHOST_SOLVER_KACZ)) && !(mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) ) {
       set_kacz_ratio(mat, (void *)src, GHOST_SPARSEMAT_SRC_FUNC);
       if(mat->kaczRatio < mat->kacz_setting.active_threads) {
            mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_PERMUTE;
            //mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_BLOCKCOLOR; //This would be done by inner test
           }
    }
   if (mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
        if (mat->traits.flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_sparsemat_perm_scotch(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        } 
        if (mat->traits.flags & GHOST_SPARSEMAT_ZOLTAN) {
            ghost_sparsemat_perm_zoltan(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        }
      if (mat->traits.flags & GHOST_SPARSEMAT_RCM) { 
            ghost_sparsemat_perm_spmp(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        }

      if (mat->traits.flags & GHOST_SPARSEMAT_COLOR) {
            ghost_sparsemat_perm_color(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        }
    //blockcoloring needs to know bandwidth //TODO avoid 2 times calculating  bandwidth, if no RCM or no bandwidth disturbing permutations are done 
    if( (mat->traits.flags & GHOST_SOLVER_KACZ) && !(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
           set_kacz_ratio(mat, (void *)src, GHOST_SPARSEMAT_SRC_FUNC);
           if(mat->kaczRatio < mat->kacz_setting.active_threads) {
            mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_BLOCKCOLOR; 
          }
   }
   //take this branch only if the matrix cannot be bandwidth bound, 
    //else normal splitting with just RCM permutation would do the work
        if (mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) {
           ghost_sparsemat_blockColor(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC);
        }

        if (mat->traits.sortScope > 1) {
            ghost_sparsemat_perm_sort(mat,(void *)src,GHOST_SPARSEMAT_SRC_FUNC,mat->traits.sortScope);
        }
        if ( mat->context->perm_local && mat->context->perm_local->colPerm == NULL) {
            mat->context->perm_local->colPerm = mat->context->perm_local->perm;
            mat->context->perm_local->colInvPerm = mat->context->perm_local->invPerm;
        }
        if (mat->context->perm_global && mat->context->perm_global->colPerm == NULL) {
            mat->context->perm_global->colPerm = mat->context->perm_global->perm;
            mat->context->perm_global->colInvPerm = mat->context->perm_global->invPerm;
        }
        if (mat->traits.flags & GHOST_SPARSEMAT_NOT_SORT_COLS) {
            PERFWARNING_LOG("Unsorted columns inside a row may yield to bad performance! However, matrix construnction will be faster.");
        }
   } else {

       if (mat->traits.sortScope > 1) {
            WARNING_LOG("Ignoring sorting scope");
        }
        mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_NOT_PERMUTE_COLS;
        mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_NOT_SORT_COLS;
    }

    ghost_lidx *tmpclp = NULL;
    if (!clp) {
        ghost_malloc((void **)&tmpclp,nchunks*sizeof(ghost_lidx));
        clp = tmpclp;
    }
    ghost_lidx *tmprl = NULL;
    if (!rl) {
        ghost_malloc((void **)&tmprl,nchunks*sizeof(ghost_lidx));
        rl = tmprl;
    }


    if (!(*chunkptr)) {
        GHOST_INSTR_START("rowlens");
        GHOST_CALL_GOTO(ghost_malloc_align((void **)chunkptr,(nchunks+1)*sizeof(ghost_lidx),GHOST_DATA_ALIGNMENT),err,ret);
    
  } 
#pragma omp parallel private(i,tmpval,tmpcol,row,maxRowLenInChunk) reduction (+:gnents,gnnz,funcerrs) reduction (max:privateMaxRowLen) 
        {
            ghost_lidx rowlen;
            maxRowLenInChunk = 0; 
            GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
            GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx)),ret);

            /*if (!(mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) && src->func == ghost_sparsemat_rowfunc_crs) {
#pragma omp single
                INFO_LOG("Fast matrix construction for CRS source and no permutation") 
#pragma omp for schedule(runtime)
                for( chunk = 0; chunk < nchunks; chunk++ ) {
                    chunkptr[chunk] = 0; // NUMA init
                    for (i=0, row = chunk*C; i < C && row < mat->nrows; i++, row++) {

                        rowlen=((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt[mat->context->lfRow[me]+row+1]-((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt[mat->context->lfRow[me]+row];

                        // rl _must_ not be NULL because we need it for the statistics
                        rl[row] = rowlen;
                        
                        if (rlp) {
                            rlp[row] = PAD(rowlen,P);
                        }

                        gnnz += rowlen;
                        maxRowLenInChunk = MAX(maxRowLenInChunk,rowlen);
                    }
                    if (cl) {
                        cl[chunk] = maxRowLenInChunk;
                    }

                    // clp _must_ not be NULL because we need it for the chunkptr computation
                    clp[chunk] = PAD(maxRowLenInChunk,P);

                    gnents += clp[chunk]*C;

                    privateMaxRowLen = MAX(privateMaxRowLen,maxRowLenInChunk);
                    maxRowLenInChunk = 0;
                }
            } else {*/
#pragma omp for schedule(runtime)
                for( chunk = 0; chunk < nchunks; chunk++ ) {
                    (*chunkptr)[chunk] = 0; // NUMA init
                    for (i=0, row = chunk*C; (i < C) && (row < mat->nrows); i++, row++) {

                    if (mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                      if (mat->context->perm_global && mat->context->perm_local) {
                        INFO_LOG("Global _and_ local permutation");
                            funcerrs += src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[row]],&rowlen,tmpcol,tmpval,src->arg);
                      } else if (mat->context->perm_global) {
                            funcerrs += src->func(mat->context->perm_global->invPerm[row],&rowlen,tmpcol,tmpval,src->arg);
                      } else if (mat->context->perm_local) {
                           funcerrs += src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[row],&rowlen,tmpcol,tmpval,src->arg);
                      }
                    } else {
                          funcerrs += src->func(mat->context->lfRow[me]+row,&rowlen,tmpcol,tmpval,src->arg);
                    }


                    // rl _must_ not be NULL because we need it for the statistics
                    rl[row] = rowlen;
                    
                    if (rlp) {
                        rlp[row] = PAD(rowlen,P);
                    }

                    gnnz += rowlen;
                    maxRowLenInChunk = MAX(maxRowLenInChunk,rowlen);
                    }
                    if (cl) {
                        cl[chunk] = maxRowLenInChunk;
                    }

                    // clp _must_ not be NULL because we need it for the chunkptr computation
                    clp[chunk] = PAD(maxRowLenInChunk,P);

                    gnents += clp[chunk]*C;

                    privateMaxRowLen = MAX(privateMaxRowLen,maxRowLenInChunk);
                    maxRowLenInChunk = 0;
                }
            //}


            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;
        }
        GHOST_INSTR_STOP("rowlens");
        maxRowLen = privateMaxRowLen;
        mat->maxRowLen = maxRowLen;

        if (funcerrs) {
            ERROR_LOG("Matrix construction function returned error");
            ret = GHOST_ERR_UNKNOWN;
            goto err;
        }
        if (gnents > (ghost_gidx)GHOST_LIDX_MAX) {
            ERROR_LOG("The local number of entries is too large: %"PRGIDX,gnents);
            return GHOST_ERR_DATATYPE;
        }
        if (gnnz > (ghost_gidx)GHOST_LIDX_MAX) {
            ERROR_LOG("The local number of entries is too large: %"PRGIDX,gnents);
            return GHOST_ERR_DATATYPE;
        }

        mat->nnz = (ghost_lidx)gnnz;
        mat->nEnts = (ghost_lidx)gnents;

        GHOST_INSTR_START("chunkptr_init");
        for(chunk = 0; chunk < nchunks; chunk++ ) {
            (*chunkptr)[chunk+1] = (*chunkptr)[chunk] + clp[chunk]*C;
        }
        GHOST_INSTR_STOP("chunkptr_init");
        

#ifdef GHOST_HAVE_MPI
        ghost_gidx fent = 0;
        for (i=0; i<nprocs; i++) {
            if (i>0 && me==i) {
                MPI_CALL_GOTO(MPI_Recv(&fent,1,ghost_mpi_dt_gidx,me-1,me-1,mat->context->mpicomm,MPI_STATUS_IGNORE),err,ret);
            }
            if (me==i && i<nprocs-1) {
                ghost_gidx send = fent+mat->nEnts;
                MPI_CALL_GOTO(MPI_Send(&send,1,ghost_mpi_dt_gidx,me+1,me,mat->context->mpicomm),err,ret);
            }
        }
        
        MPI_CALL_GOTO(MPI_Allgather(&mat->nEnts,1,ghost_mpi_dt_lidx,mat->context->lnEnts,1,ghost_mpi_dt_lidx,mat->context->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Allgather(&fent,1,ghost_mpi_dt_gidx,mat->context->lfEnt,1,ghost_mpi_dt_gidx,mat->context->mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Allreduce(&gnnz,&mat->context->gnnz,1,ghost_mpi_dt_gidx,MPI_SUM,mat->context->mpicomm),err,ret);
#endif
   
 
    if (src->maxrowlen != mat->maxRowLen) {
        DEBUG_LOG(1,"The maximum row length was not correct. Setting it from %"PRLIDX" to %"PRGIDX,src->maxrowlen,mat->maxRowLen); 
        src->maxrowlen = mat->maxRowLen;
    }

   
    bool readcols = 0; // we only need to read the columns the first time the matrix is created
    if (!(*val)) {
        GHOST_CALL_GOTO(ghost_malloc_align((void **)val,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
    }
    
    if (!(*col)) {
        GHOST_CALL_GOTO(ghost_malloc_align((void **)col,sizeof(ghost_gidx)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
        readcols = 1;
    }

        
    if (src->func == ghost_sparsemat_rowfunc_crs && mat->context->perm_global) {
        ERROR_LOG("Global permutation does not work with local CRS source");
    }

    GHOST_INSTR_START("cols_and_vals");
#pragma omp parallel private(i,colidx,row,tmpval,tmpcol)
    {
        int funcret = 0;
        GHOST_CALL(ghost_malloc((void **)&tmpval,C*src->maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,C*src->maxrowlen*sizeof(ghost_gidx)),ret);
        
        if (src->func == ghost_sparsemat_rowfunc_crs) {
            ghost_gidx *crscol;
            char *crsval = (char *)(((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->val);
            ghost_lidx *crsrpt = ((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt;
#pragma omp single
            INFO_LOG("Fast matrix construction for CRS source and no permutation");

#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nchunks; chunk++ ) {
                //memset(tmpval,0,mat->elSize*src->maxrowlen*C);

                for (i=0, row = chunk*C; (i<C) && (chunk*C+i < mat->nrows); i++, row++) {
                    ghost_gidx actualrow;
                    if (mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                        actualrow = mat->context->perm_local->invPerm[row];
                    } else {
                        actualrow = row;
                    }
                    
                    crsval = &((char *)(((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->val))[crsrpt[actualrow]*mat->elSize];

#pragma vector nontemporal
                    for(colidx = 0; colidx<rl[row]; colidx++) {
                        // assignment is much faster than memcpy with non-constant size, so we need those branches...
                        if (mat->traits.datatype & GHOST_DT_REAL) {
                            if (mat->traits.datatype & GHOST_DT_DOUBLE) {
                                ((double *)(*val))[(*chunkptr)[chunk]+colidx*C+i] = ((double *)(crsval))[colidx];
                            } else {
                                ((float *)(*val))[(*chunkptr)[chunk]+colidx*C+i] = ((float *)(crsval))[colidx];
                            }
                        } else {
                            if (mat->traits.datatype & GHOST_DT_DOUBLE) {
                                ((complex double *)(*val))[(*chunkptr)[chunk]+colidx*C+i] = ((complex double *)(crsval))[colidx];
                            } else {
                                ((complex float *)(*val))[(*chunkptr)[chunk]+colidx*C+i] = ((complex float *)(crsval))[colidx];
                            }
                        }
                        if (readcols) {
                            crscol = &((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->col[crsrpt[actualrow]];
                            if (mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                                // local permutation: distinction between global and local entriess, if GHOST_PERM_NO_DISTINCTION is not set 
                                if ((mat->context->flags & GHOST_PERM_NO_DISTINCTION) || ( (crscol[colidx] >= mat->context->lfRow[me]) && (crscol[colidx] < (mat->context->lfRow[me]+mat->ncols)) )) { // local entry: copy with permutation
                                    if (mat->traits.flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = crscol[colidx];
                                    } else if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[crscol[colidx]];
                                    } else {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[crscol[colidx]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                                    }

                                } else { // remote entry: copy without permutation
                                    (*col)[(*chunkptr)[chunk]+colidx*C+i] = crscol[colidx];
                                }
                            } else {
                                (*col)[(*chunkptr)[chunk]+colidx*C+i] = crscol[colidx];
                            }
                        }
                    }
                     for (; colidx < clp[chunk]; colidx++) {
                        memset(&(*val)[((*chunkptr)[chunk]+colidx*C+i)*mat->elSize],0,mat->elSize);
                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->lfRow[me];
                    }

                }
            }
        } else {
#pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nchunks; chunk++ ) {
                memset(tmpval,0,mat->elSize*src->maxrowlen*C);  
                if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                    for (i=0; i<src->maxrowlen*C; i++) {    
                        tmpcol[i] = 0;
                    }
                } else {
                    for (i=0; i<src->maxrowlen*C; i++) {    
                        tmpcol[i] = mat->context->lfRow[me];
                    }
                }
                for (i=0, row = chunk*C; (i<C) && (chunk*C+i < mat->nrows); i++, row++) {
                    if (mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                        if (mat->context->perm_global && mat->context->perm_local) {
                            funcret = src->func(mat->context->perm_global->invPerm[mat->context->perm_local->invPerm[row]],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize],src->arg);
                        } else if (mat->context->perm_global) {
                            funcret = src->func(mat->context->perm_global->invPerm[row],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize],src->arg);
                        } else if (mat->context->perm_local) {
                            funcret = src->func(mat->context->lfRow[me]+mat->context->perm_local->invPerm[row],&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize],src->arg);
                        }

                    } else {
                        funcret = src->func(mat->context->lfRow[me]+row,&rl[row],&tmpcol[src->maxrowlen*i],&tmpval[src->maxrowlen*i*mat->elSize],src->arg);
                    }
                    if (funcret) {
                        ERROR_LOG("Matrix construction function returned error");
                        ret = GHOST_ERR_UNKNOWN;
                    }
                    for (colidx = 0; colidx<clp[chunk]; colidx++) {
                        memcpy(*val+mat->elSize*((*chunkptr)[chunk]+colidx*C+i),&tmpval[mat->elSize*(i*src->maxrowlen+colidx)],mat->elSize);
                        if (mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
                            if (mat->context->perm_global) {
                                // no distinction between global and local entries
                                // global permutation will be done after all rows are read
                                (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                            } else { 
                                // local permutation: distinction between global and local entries, if GHOST_PERM_NO_DISTINCTION is not set 
                                if ((mat->context->flags & GHOST_PERM_NO_DISTINCTION) || ((tmpcol[i*src->maxrowlen+colidx] >= mat->context->lfRow[me]) && (tmpcol[i*src->maxrowlen+colidx] < (mat->context->lfRow[me]+mat->nrows)))) { 
                                    // local entry: copy with permutation
                                    if (mat->traits.flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                                    } else if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                                        // (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[tmpcol[i*src->maxrowlen+colidx]]   
                                        // do not permute remote and do not allow local to go to remote
                                        if(tmpcol[i*src->maxrowlen+colidx] < mat->context->nrowspadded ) {
                                            if( mat->context->perm_local->colPerm[tmpcol[i*src->maxrowlen+colidx]]>=mat->context->nrowspadded ) {       
                                                ERROR_LOG("Ensure you have halo number of paddings, since GHOST_PERM_NO_DISTINCTION is switched on");       
                                            }
                                            (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[tmpcol[i*src->maxrowlen+colidx]];
                                        } else {
                                            (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                                        }
                                    } else {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[tmpcol[i*src->maxrowlen+colidx]-mat->context->lfRow[me]]+mat->context->lfRow[me];
//                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->perm_local->colPerm[tmpcol[i*src->maxrowlen+colidx]-mat->context->lfRow[me]]+mat->context->lfRow[me];
                                    }
                                } else { 
                                    // remote entry: copy without permutation
                                    (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                                }
                            }
                        } else {
                            (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*src->maxrowlen+colidx];
                        }
                   }
                }
            }
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }

    if (mat->nrows % C) {
        for (i=mat->nrows%C; i < C; i++) {
            for (colidx = 0; colidx<clp[nchunks-1]; colidx++) {
                (*col)[(*chunkptr)[nchunks-1]+colidx*C+i] = mat->context->lfRow[me];
                memset(*val+mat->elSize*((*chunkptr)[nchunks-1]+colidx*C+i),0,mat->elSize);
            }
        }
    }

    GHOST_INSTR_STOP("cols_and_vals");
    
    if (mat->context->perm_global) {
        ghost_sparsemat_perm_global_cols(*col,mat->nEnts,mat->context);
    }
    
    GHOST_INSTR_START("sort_and_register");
    
    if (!(mat->traits.flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
        for( chunk = 0; chunk < nchunks; chunk++ ) {
            for (i=0; (i<C) && (chunk*C+i < mat->nrows); i++) {
                row = chunk*C+i;
                ghost_sparsemat_sortrow(&((*col)[(*chunkptr)[chunk]+i]),&(*val)[((*chunkptr)[chunk]+i)*mat->elSize],mat->elSize,rl[row],C);
#ifdef GHOST_SPARSEMAT_STATS
                ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&(*col)[(*chunkptr)[chunk]+i],rl[row],C);
#endif
            }
        }
    } else {
#ifdef GHOST_SPARSEMAT_STATS
        for( chunk = 0; chunk < nchunks; chunk++ ) {
            for (i=0; (i<C) && (chunk*C+i < mat->nrows); i++) {
                row = chunk*C+i;
                ghost_sparsemat_registerrow(mat,mat->context->lfRow[me]+row,&(*col)[(*chunkptr)[chunk]+i],rl[row],C);
            }
        }
#endif
    }

#ifdef GHOST_SPARSEMAT_STATS
    ghost_sparsemat_registerrow_finalize(mat);
#endif
    GHOST_INSTR_STOP("sort_and_register");
    
    mat->context->lnEnts[me] = mat->nEnts;

    for (i=0; i<nprocs; i++) {
        mat->context->lfEnt[i] = 0;
    } 

    for (i=1; i<nprocs; i++) {
        mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
    } 

    free(tmpclp);
    free(tmprl);

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;
}

ghost_error ghost_sparsemat_perm_global_cols(ghost_gidx *col, ghost_lidx ncols, ghost_context *context) 
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_COMMUNICATION);
    int me, nprocs,i;
    ghost_rank(&me,context->mpicomm);
    ghost_nrank(&nprocs,context->mpicomm);

    for (i=0; i<nprocs; i++) {
        ghost_lidx nels = 0;
        if (i==me) {
            nels = ncols;
        }
        MPI_Bcast(&nels,1,ghost_mpi_dt_lidx,i,context->mpicomm);

        ghost_gidx *colsfromi;
        ghost_malloc((void **)&colsfromi,nels*sizeof(ghost_gidx));
    
        if (i==me) {
            memcpy(colsfromi,col,nels*sizeof(ghost_gidx));
        }
        MPI_Bcast(colsfromi,nels,ghost_mpi_dt_gidx,i,context->mpicomm);

        ghost_lidx el;
        for (el=0; el<nels; el++) {
            if ((colsfromi[el] >= context->lfRow[me]) && (colsfromi[el] < (context->lfRow[me]+context->lnrows[me]))) {
                colsfromi[el] = context->perm_global->perm[colsfromi[el]-context->lfRow[me]];
            } else {
                colsfromi[el] = 0;
            }
        }

        if (i==me) {
            MPI_Reduce(MPI_IN_PLACE,colsfromi,nels,ghost_mpi_dt_gidx,MPI_MAX,i,context->mpicomm);
        } else {
            MPI_Reduce(colsfromi,NULL,nels,ghost_mpi_dt_gidx,MPI_MAX,i,context->mpicomm);
        }

        if (i==me) {
            if (context->perm_local) {
                for (el=0; el<nels; el++) {
                    if ((colsfromi[el] >= context->lfRow[me]) && (colsfromi[el] < context->lfRow[me]+context->lnrows[me])) {
                        col[el] = context->lfRow[me] + context->perm_local->perm[colsfromi[el]-context->lfRow[me]];
                    } else {
                        col[el] = colsfromi[el];
                    }

                }
            } else {
                memcpy(col,colsfromi,nels*sizeof(ghost_gidx));
            }
        }

        free(colsfromi);
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_COMMUNICATION);
#else
    ERROR_LOG("This function should not have been called without MPI!");
    UNUSED(col);
    UNUSED(ncols);
    UNUSED(context);
#endif
    return GHOST_SUCCESS;
}

ghost_error ghost_sparsemat_nrows(ghost_gidx *nrows, ghost_sparsemat *mat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (!nrows) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }

    *nrows = mat->context->gnrows;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_sparsemat_nnz(ghost_gidx *nnz, ghost_sparsemat *mat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    if (!nnz) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
  /*  ghost_gidx lnnz = mat->nnz;

#ifdef GHOST_HAVE_MPI
    MPI_CALL_RETURN(MPI_Allreduce(&lnnz,nnz,1,ghost_mpi_dt_gidx,MPI_SUM,mat->context->mpicomm));
#else
    *nnz = lnnz;
#endif
*/
    *nnz = mat->context->gnnz;

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;
}

ghost_error ghost_sparsemat_info_string(char **str, ghost_sparsemat *mat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_CALL_RETURN(ghost_malloc((void **)str,1));
    memset(*str,'\0',1);

    int myrank;
    ghost_gidx nrows = 0;
    ghost_gidx nnz = 0;

    GHOST_CALL_RETURN(ghost_sparsemat_nrows(&nrows,mat));
    GHOST_CALL_RETURN(ghost_sparsemat_nnz(&nnz,mat));
    GHOST_CALL_RETURN(ghost_rank(&myrank, mat->context->mpicomm));


    char *matrixLocation;
    if (mat->traits.flags & GHOST_SPARSEMAT_DEVICE)
        matrixLocation = "Device";
    else if (mat->traits.flags & GHOST_SPARSEMAT_HOST)
        matrixLocation = "Host";
    else
        matrixLocation = "Default";


    ghost_header_string(str,"%s @ rank %d",mat->name,myrank);
    ghost_line_string(str,"Data type",NULL,"%s",ghost_datatype_string(mat->traits.datatype));
    ghost_line_string(str,"Matrix location",NULL,"%s",matrixLocation);
    ghost_line_string(str,"Total number of rows",NULL,"%"PRGIDX,nrows);
    ghost_line_string(str,"Total number of nonzeros",NULL,"%"PRGIDX,nnz);
    ghost_line_string(str,"Avg. nonzeros per row",NULL,"%.3f",(double)nnz/nrows);
    ghost_line_string(str,"Bandwidth",NULL,"%"PRGIDX,mat->bandwidth);
    ghost_line_string(str,"Avg. row band",NULL,"%.3f",mat->avgRowBand);
    ghost_line_string(str,"Avg. avg. row band",NULL,"%.3f",mat->avgAvgRowBand);
    ghost_line_string(str,"Smart row band",NULL,"%.3f",mat->smartRowBand);

    ghost_line_string(str,"Local number of rows",NULL,"%"PRLIDX,mat->nrows);
    ghost_line_string(str,"Local number of rows (padded)",NULL,"%"PRLIDX,mat->nrowsPadded);
    ghost_line_string(str,"Local number of nonzeros",NULL,"%"PRLIDX,mat->nnz);

    ghost_line_string(str,"Full   matrix format",NULL,"%s",mat->formatName(mat));
    if (mat->localPart) {
        ghost_line_string(str,"Local  matrix format",NULL,"%s",mat->localPart->formatName(mat->localPart));
        ghost_line_string(str,"Local  matrix symmetry",NULL,"%s",ghost_sparsemat_symmetry_string(mat->localPart->traits.symmetry));
        ghost_line_string(str,"Local  matrix size","MB","%u",mat->localPart->byteSize(mat->localPart)/(1024*1024));
    }
    if (mat->remotePart) {
        ghost_line_string(str,"Remote matrix format",NULL,"%s",mat->remotePart->formatName(mat->remotePart));
        ghost_line_string(str,"Remote matrix size","MB","%u",mat->remotePart->byteSize(mat->remotePart)/(1024*1024));
    }

    ghost_line_string(str,"Full   matrix size","MB","%u",mat->byteSize(mat)/(1024*1024));
    
    if (mat->traits.flags & GHOST_SPARSEMAT_PERMUTE) {
        ghost_line_string(str,"Permuted",NULL,"Yes");
        if (mat->context->perm_global) {
            if (mat->context->perm_local) {
                ghost_line_string(str,"Permutation scope",NULL,"Global+local");
            } else {
                ghost_line_string(str,"Permutation scope",NULL,"Global");
            }
            if (mat->traits.flags & GHOST_SPARSEMAT_SCOTCHIFY) {
                ghost_line_string(str,"Global permutation strategy",NULL,"SCOTCH");
                ghost_line_string(str,"SCOTCH ordering strategy",NULL,"%s",mat->traits.scotchStrat);
            }
            if (mat->traits.flags & GHOST_SPARSEMAT_ZOLTAN) {
                ghost_line_string(str,"Global permutation strategy",NULL,"ZOLTAN");
            }
        } else if (mat->context->perm_local) {
            ghost_line_string(str,"Permutation scope",NULL,"Local");
        }
        if (mat->context->perm_local) {
            if (mat->traits.sortScope > 1) {
                if (mat->traits.flags & GHOST_SPARSEMAT_RCM) {
                    ghost_line_string(str,"Local permutation strategy",NULL,"RCM+Sorting");
                } else {
                    ghost_line_string(str,"Local permutation strategy",NULL,"Sorting");
                }
            } else if (mat->traits.flags & GHOST_SPARSEMAT_RCM) {
                ghost_line_string(str,"Local permutation strategy",NULL,"RCM");
            }
        }
        ghost_line_string(str,"Permuted column indices",NULL,"%s",mat->traits.flags&GHOST_SPARSEMAT_NOT_PERMUTE_COLS?"No":"Yes");
    } else {
        ghost_line_string(str,"Permuted",NULL,"No");
    }
    
    ghost_line_string(str,"Row length sorting scope (sigma)",NULL,"%d",mat->traits.sortScope);
    ghost_line_string(str,"Ascending columns in row",NULL,"%s",mat->traits.flags&GHOST_SPARSEMAT_NOT_SORT_COLS?"Maybe":"Yes");
    ghost_line_string(str,"Max row length (# rows)",NULL,"%d (%d)",mat->maxRowLen,mat->nMaxRows);
    ghost_line_string(str,"Row length variance",NULL,"%f",mat->variance);
    ghost_line_string(str,"Row length standard deviation",NULL,"%f",mat->deviation);
    ghost_line_string(str,"Row length coefficient of variation",NULL,"%f",mat->cv);
    ghost_line_string(str,"Chunk height (C)",NULL,"%d",mat->traits.C);
    ghost_line_string(str,"Chunk occupancy (beta)",NULL,"%f",(double)(mat->nnz)/(double)(mat->nEnts));
    ghost_line_string(str,"Threads per row (T)",NULL,"%d",mat->traits.T);

    ghost_footer_string(str);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return GHOST_SUCCESS;

}

ghost_error ghost_sparsematofile_header(ghost_sparsemat *mat, char *path)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_IO);

    ghost_gidx mnrows,mncols,mnnz;
    GHOST_CALL_RETURN(ghost_sparsemat_nrows(&mnrows,mat));
    mncols = mnrows;
    GHOST_CALL_RETURN(ghost_sparsemat_nnz(&mnnz,mat));
    
    int32_t endianess = ghost_machine_bigendian();
    int32_t version = 1;
    int32_t base = 0;
    int32_t symmetry = GHOST_BINCRS_SYMM_GENERAL;
    int32_t datatype = mat->traits.datatype;
    int64_t nrows = (int64_t)mnrows;
    int64_t ncols = (int64_t)mncols;
    int64_t nnz = (int64_t)mnnz;

    size_t ret;
    FILE *filed;

    if ((filed = fopen64(path, "w")) == NULL){
        ERROR_LOG("Could not open binary CRS file %s: %s",path,strerror(errno));
        return GHOST_ERR_IO;
    }

    if ((ret = fwrite(&endianess,sizeof(endianess),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&version,sizeof(version),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&base,sizeof(base),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&symmetry,sizeof(symmetry),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&datatype,sizeof(datatype),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&nrows,sizeof(nrows),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&ncols,sizeof(ncols),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    if ((ret = fwrite(&nnz,sizeof(nnz),1,filed)) != 1) {
        ERROR_LOG("fwrite failed: %zu",ret);
        fclose(filed);
        return GHOST_ERR_IO;
    }
    fclose(filed);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_IO);
    return GHOST_SUCCESS;
}

bool ghost_sparsemat_symmetry_valid(ghost_sparsemat_symmetry symmetry)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);

    if ((symmetry & (ghost_sparsemat_symmetry)GHOST_SPARSEMAT_SYMM_GENERAL) &&
            (symmetry & ~(ghost_sparsemat_symmetry)GHOST_SPARSEMAT_SYMM_GENERAL))
        return 0;

    if ((symmetry & (ghost_sparsemat_symmetry)GHOST_SPARSEMAT_SYMM_SYMMETRIC) &&
            (symmetry & ~(ghost_sparsemat_symmetry)GHOST_SPARSEMAT_SYMM_SYMMETRIC))
        return 0;

    return 1;
}

const char * ghost_sparsemat_symmetry_string(ghost_sparsemat_symmetry symmetry)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    
    if (symmetry & GHOST_SPARSEMAT_SYMM_GENERAL)
        return "General";

    if (symmetry & GHOST_SPARSEMAT_SYMM_SYMMETRIC)
        return "Symmetric";

    if (symmetry & GHOST_SPARSEMAT_SYMM_SKEW_SYMMETRIC) {
        if (symmetry & GHOST_SPARSEMAT_SYMM_HERMITIAN)
            return "Skew-hermitian";
        else
            return "Skew-symmetric";
    } else {
        if (symmetry & GHOST_SPARSEMAT_SYMM_HERMITIAN)
            return "Hermitian";
    }

    return "Invalid";
}

void ghost_sparsemat_destroy(ghost_sparsemat *mat)
{
    if (!mat) {
        return;
    }

    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_TEARDOWN);
    if (mat->sell) {
#ifdef GHOST_HAVE_CUDA
        if (mat->traits.flags & GHOST_SPARSEMAT_DEVICE && SELL(mat)->cumat) {
            ghost_cu_free(SELL(mat)->cumat->rowLen);
            ghost_cu_free(SELL(mat)->cumat->rowLenPadded);
            ghost_cu_free(SELL(mat)->cumat->col);
            ghost_cu_free(SELL(mat)->cumat->val);
            ghost_cu_free(SELL(mat)->cumat->chunkStart);
            ghost_cu_free(SELL(mat)->cumat->chunkLen);
            free(SELL(mat)->cumat);
        }
#endif
        free(SELL(mat)->val); SELL(mat)->val = NULL;
        free(SELL(mat)->col); SELL(mat)->col = NULL;
        free(SELL(mat)->chunkStart); SELL(mat)->chunkStart = NULL;
        free(SELL(mat)->chunkMin); SELL(mat)->chunkMin = NULL;
        free(SELL(mat)->chunkLen); SELL(mat)->chunkLen = NULL;
        free(SELL(mat)->chunkLenPadded); SELL(mat)->chunkLenPadded = NULL;
        free(SELL(mat)->rowLen); SELL(mat)->rowLen = NULL;
        free(SELL(mat)->rowLen2); SELL(mat)->rowLen2 = NULL;
        free(SELL(mat)->rowLen4); SELL(mat)->rowLen4 = NULL;
        free(SELL(mat)->rowLenPadded); SELL(mat)->rowLenPadded = NULL;
    }

         
    if (mat->localPart) {
        ghost_sparsemat_destroy(mat->localPart);
    }

    if (mat->remotePart) {
        ghost_sparsemat_destroy(mat->remotePart);
    }
    
    if (mat->color_ptr)  {
        free(mat->color_ptr);
    }
    
     if (mat->zone_ptr)  {
        free(mat->zone_ptr);
    }

    free(mat->sell); mat->sell = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
    
    free(mat);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TEARDOWN);
}

ghost_error ghost_sparsemat_from_bincrs(ghost_sparsemat *mat, char *path)
{
    PERFWARNING_LOG("The current implementation of binCRS read-in is "
            "inefficient in terms of memory consumption!");
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    
    ghost_error ret = GHOST_SUCCESS;
    ghost_sparsemat_rowfunc_bincrs_initargs args;
    ghost_gidx dim[2];
    ghost_lidx bincrs_dt = 0; // or use args.dt directly...
    ghost_sparsemat_src_rowfunc src = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    
    src.func = &ghost_sparsemat_rowfunc_bincrs;
    src.arg = mat;
    args.filename = path;
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETDIM,&bincrs_dt,dim,&args,src.arg)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    
    // Apply file datatype only if still unspecified.
    if(mat->traits.datatype == GHOST_DT_NONE) mat->traits.datatype = (ghost_datatype)bincrs_dt;
    // Require valid datatype here.
    GHOST_CALL_GOTO(ghost_datatype_size(&mat->elSize,mat->traits.datatype),err,ret);   
    args.dt = mat->traits.datatype;

    if (src.func(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_INIT,NULL,NULL,&args,src.arg)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
   
    src.maxrowlen = dim[1];
    
    GHOST_CALL_GOTO(mat->fromRowFunc(mat,&src),err,ret);
    
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_FINALIZE,NULL,NULL,NULL,src.arg)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    return ret;

}

ghost_error ghost_sparsemat_from_mm(ghost_sparsemat *mat, char *path)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    
    ghost_error ret = GHOST_SUCCESS;
    ghost_sparsemat_rowfunc_mm_initargs args;
    ghost_gidx dim[2];
    ghost_lidx bincrs_dt = 0;
    ghost_sparsemat_src_rowfunc src = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
  
    int symmetric = 0;
    src.arg = &symmetric;

    if (mat->traits.flags & GHOST_SPARSEMAT_TRANSPOSE_MM) { 
        src.func = &ghost_sparsemat_rowfunc_mm_transpose;
    } else {
        src.func = &ghost_sparsemat_rowfunc_mm;
    }
    args.filename = path;
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETDIM,&bincrs_dt,dim,&args,src.arg)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    
    // Construct final datatype.
    if(mat->traits.datatype == GHOST_DT_NONE) mat->traits.datatype = GHOST_DT_DOUBLE;
    if((mat->traits.datatype == GHOST_DT_DOUBLE) || (mat->traits.datatype == GHOST_DT_FLOAT))
        mat->traits.datatype |= (ghost_datatype)bincrs_dt;
    GHOST_CALL_GOTO(ghost_datatype_size(&mat->elSize,mat->traits.datatype),err,ret);   
    args.dt = mat->traits.datatype;
    
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_INIT,NULL,NULL,&args,src.arg)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
    
    src.maxrowlen = dim[1];
    
    GHOST_CALL_GOTO(mat->fromRowFunc(mat,&src),err,ret);
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_FINALIZE,NULL,NULL,NULL,src.arg)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }

    if (*(int *)src.arg) {
        mat->traits.symmetry = GHOST_SPARSEMAT_SYMM_SYMMETRIC;
    }

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    return ret;

}

extern inline int ghost_sparsemat_rowfunc_crs(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg);

ghost_error ghost_sparsemat_from_crs(ghost_sparsemat *mat, ghost_gidx offs, ghost_lidx n, ghost_gidx *col, void *val, ghost_lidx *rpt)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    
    ghost_error ret = GHOST_SUCCESS;
    ghost_sparsemat_rowfunc_crs_arg args;

    // Require valid datatpye here.
    GHOST_CALL_GOTO(ghost_datatype_size(&mat->elSize,mat->traits.datatype),err,ret);

    args.dtsize = mat->elSize;
    args.col = col;
    args.val = val;
    args.rpt = rpt;
    args.offs = offs;

    ghost_sparsemat_src_rowfunc src = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    
    src.func = &ghost_sparsemat_rowfunc_crs;
    src.arg = &args;
    src.maxrowlen = n;
    
    GHOST_CALL_GOTO(mat->fromRowFunc(mat,&src),err,ret);

    goto out;
err:

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;

}

static const char * SELL_formatName(ghost_sparsemat *mat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    // TODO format SELL-C-sigma
    UNUSED(mat);
    return "SELL";
}

static size_t SELL_byteSize (ghost_sparsemat *mat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    if (mat->sell == NULL) {
        return 0;
    }
    return (size_t)((mat->nrowsPadded/mat->traits.C)*sizeof(ghost_lidx) + 
            mat->nEnts*(sizeof(ghost_lidx)+mat->elSize));
}


typedef struct 
{
    ghost_lidx *col;//only ghost_lidx is required, since compressed
    void *val;
    ghost_lidx *chunk_ptr;
    ghost_lidx *rowLen;
    size_t dtsize;
    int CHUNKHEIGHT;
    ghost_gidx offs;
} 
ghost_sparsemat_rowfunc_after_split_arg;


static inline int ghost_sparsemat_rowfunc_after_split_func(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *data)
{
    ghost_lidx *data_col = ((ghost_sparsemat_rowfunc_after_split_arg *)data)->col;
    ghost_lidx *rowLen = ((ghost_sparsemat_rowfunc_after_split_arg *)data)->rowLen;
    ghost_lidx *data_chunk_ptr = ((ghost_sparsemat_rowfunc_after_split_arg *)data)->chunk_ptr;
    char *data_val = (char *)((ghost_sparsemat_rowfunc_after_split_arg *)data)->val;
    size_t dtsize = ((ghost_sparsemat_rowfunc_after_split_arg *)data)->dtsize;   
    ghost_gidx offs = ((ghost_sparsemat_rowfunc_after_split_arg *)data)->offs;

    int C =  ((ghost_sparsemat_rowfunc_after_split_arg *)data)->CHUNKHEIGHT;

    *rowlen = rowLen[row-offs];
   for(int i =0; i<(*rowlen); ++i) {
        ghost_gidx curr_col = (ghost_gidx) data_col[data_chunk_ptr[(row-offs)/C] + (row-offs)%C + C*i];
    col[i] = curr_col; 
    for(int t=0; t<dtsize; ++t) {
        ((char*)val)[dtsize*i+t] = (char) data_val[(data_chunk_ptr[(row-offs)/C] + (row-offs)%C + C*i)*dtsize + t];
    }

   }
 

  //  memcpy(col,&data_col[data_rpt[row-offs]],*rowlen * sizeof(ghost_gidx));
  //  memcpy(val,&data_val[dtsize*data_rpt[row-offs]],*rowlen * dtsize);
  /*  for(int i =0; i<(*rowlen); ++i) {
    for(int t=0; t<dtsize; ++t) {
            ((char*)val)[dtsize*i+t] = (char) data_val[(data_chunk_ptr[(row-offs)/C] + (row-offs)%C + C*i)*dtsize + t];
    }
    }*/


    return 0;
}

ghost_error initHaloAvg(ghost_sparsemat *mat)
{
    ghost_error ret = GHOST_SUCCESS;
    int me,nprocs;
    GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
 
    bool *compression_flag;
    int *temp_nrankspresent;
    ghost_context *ctx = mat->context; 
    //calculate rankspresent here and store it, no need to do this each time averaging is done
    GHOST_CALL_GOTO(ghost_malloc((void **)&temp_nrankspresent, ctx->nrowspadded*sizeof(int)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&compression_flag, ctx->nrowspadded*sizeof(bool)),err,ret);
   
    #pragma omp parallel for schedule(runtime)
     for (int i=0; i<ctx->nrowspadded; i++) {	
	     if(ctx->perm_local) {
		     if(ctx->perm_local->colInvPerm[i]< ctx->lnrows[me] ) { //This check is important since entsInCol has only lnrows(NO_DISTINCTION
			     //might give seg fault else) the rest are halo anyway, not needed for local sums
			     temp_nrankspresent[i] = ctx->entsInCol[ctx->perm_local->colInvPerm[i]]?1:0; //this has also to be permuted since it was
         } else {
		       temp_nrankspresent[i] = 0;//ctx->entsInCol[i]?1:0;		
         }
      } else {
         if(i < ctx->lnrows[me]) {
			      temp_nrankspresent[i] = ctx->entsInCol[i]?1:0; //this has also to be permuted since it was
          } else {
            temp_nrankspresent[i] = 0;
          }
      } 

    compression_flag[i] = false;
    }

    ghost_lidx ndues = 0;
  	for (int i=0; i<nprocs; i++) {
      if(ctx->perm_local) {
      #pragma omp parallel for schedule(runtime) 
        for (int d=0 ;d < ctx->dues[i]; d++) {
		      temp_nrankspresent[ctx->perm_local->colPerm[ctx->duelist[i][d]]]++; 
          compression_flag[ctx->perm_local->colPerm[ctx->duelist[i][d]]] = true;
        }
      } else {
       #pragma omp parallel for schedule(runtime) 
        for (int d=0 ;d < ctx->dues[i]; d++) {
	        temp_nrankspresent[ctx->duelist[i][d]]++; 
          compression_flag[ctx->duelist[i][d]] = true;
        }
      }
    ndues += ctx->dues[i];
    }

    ghost_lidx *temp_avg_ptr;
    GHOST_CALL_GOTO(ghost_malloc((void **)&temp_avg_ptr, ctx->nrowspadded*sizeof(ghost_lidx)),err,ret);
    ghost_lidx ctr = 0;
    //count number of elements
    for(ghost_lidx i=0; i<ctx->nrowspadded; ++i) {
       if(ctr==0 && compression_flag[i]==true){
         temp_avg_ptr[ctr] = i; 
         ctr += 1; 
       }
       else if(ctr!=0 && (compression_flag[i-1]!=compression_flag[i])) {
         temp_avg_ptr[ctr] = i; 
         ctr += 1;
       } 
       else if(i==ctx->nrowspadded-1 && (compression_flag[i-1]==true)) {
         temp_avg_ptr[ctr] = i+1;
         ctr += 1;
      }
    }
  
    ghost_lidx totalElem = 0;
    for(ghost_lidx i=0; i<ctr/2; ++i) {
      totalElem += (temp_avg_ptr[2*i+1]-temp_avg_ptr[2*i]);
    }

   ctx->nChunkAvg = ctr/2;
   ctx->nElemAvg = totalElem;
   //printf("Nchunk(%d) = %d, Total Elem(%d) = %d\n",me,ctx->nChunkAvg,me,totalElem); 
   GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->avg_ptr, ctr*sizeof(ghost_lidx)),err,ret);
   //now have a compressed column pointer for averaging
 #pragma omp parallel for schedule(runtime)
    for(int i=0; i<ctr; ++i) {
      ctx->avg_ptr[i] = temp_avg_ptr[i];
      //printf("pointers[%d] = %d\n",i,temp_avg_ptr[i]);
    }

    ghost_lidx *map; //map from original column to compressed column
    GHOST_CALL_GOTO(ghost_malloc((void **)&map, ctx->nrowspadded*sizeof(ghost_lidx)),err,ret); 
  
    ghost_lidx col_ctr = 0;
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->nrankspresent, totalElem*sizeof(ghost_lidx)),err,ret);
    for(ghost_lidx i=0; i<ctx->nChunkAvg; ++i) {
      for(ghost_lidx j=ctx->avg_ptr[2*i]; j<ctx->avg_ptr[2*i+1]; ++j) {
        ctx->nrankspresent[col_ctr] = temp_nrankspresent[j];
        //printf("nrankspresent[%d] = %d\n",col_ctr,ctx->nrankspresent[col_ctr]);
        map[j] = col_ctr;
        ++col_ctr;
      }  
    }
    
    ctx->mapAvg = map;
    //now get a mapped duelist
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->mappedDuelist, ndues*sizeof(ghost_lidx)),err,ret); 
    ctr = 0;
   	for (int i=0; i<nprocs; i++) {
      if(ctx->perm_local) {
       for (int d=0 ;d < ctx->dues[i]; d++) {
		      ctx->mappedDuelist[ctr] = map[ ctx->perm_local->colPerm[ctx->duelist[i][d]] ]; 
          ++ctr;
        }
      } else {
        for (int d=0 ;d < ctx->dues[i]; d++) {
	        ctx->mappedDuelist[ctr] = map[ ctx->duelist[i][d] ]; 
          ++ctr;
        }
      }
   }

    /*printf("Mapped Due List(%d) = \n",me);
   	for (int i=0; i<nprocs; i++) {
     for (int d=0 ;d < ctx->dues[i]; d++) {
        if(ctx->perm_local)
        printf("%d -> %d\n",ctx->perm_local->colPerm[ctx->duelist[i][d]], map[ctx->perm_local->colPerm[ctx->duelist[i][d]]]);
        else
        printf("%d -> %d\n",ctx->duelist[i][d], map[ctx->duelist[i][d]]);
     }
    }
    */        
 
    free(temp_avg_ptr); 
    free(temp_nrankspresent);
    goto out;

    err:
        ERROR_LOG("ERROR in initHaloAvg");
        return  GHOST_ERR_MPI; 

    out:
        return ret;        
}
 
static ghost_error SELL_fromRowFunc(ghost_sparsemat *mat, ghost_sparsemat_src_rowfunc *src)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    ghost_error ret = GHOST_SUCCESS;

    ghost_lidx nChunks = mat->nrowsPadded/mat->traits.C;
   
    // Require valid datatpye here.
    GHOST_CALL_GOTO(ghost_datatype_size(&mat->elSize,mat->traits.datatype),err,ret);

    if (!SELL(mat)->chunkMin) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkMin, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!SELL(mat)->chunkLen) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLen, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!SELL(mat)->chunkLenPadded) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLenPadded, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!SELL(mat)->rowLen) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLen, (mat->nrowsPadded)*sizeof(ghost_lidx)),err,ret);
    if (!SELL(mat)->rowLenPadded) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_lidx)),err,ret); 
 
    int me,nprocs;
    GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
 
//set NO_DISTINCTION when block multicolor and RCM is on and more than 2 processors, TODO pure MC and MPI
//this has to be invoked even if no permutations are carried out and more than 2 processors, since we need to
//know amount of remote entries before (used in sparsemat_blockcolor); 
if(nprocs>1 && mat->traits.flags & GHOST_SOLVER_KACZ) {

     INFO_LOG("NO DISTINCTION is set");
     mat->context->flags |=   (ghost_context_flags_t) GHOST_PERM_NO_DISTINCTION; 
}


   if (mat->context->flags & GHOST_PERM_NO_DISTINCTION) { 
    //TODO avoid this dummy
        GHOST_CALL_GOTO(ghost_sparsemat_fromfunc_common_dummy(SELL(mat)->rowLen,SELL(mat)->rowLenPadded,SELL(mat)->chunkLen,SELL(mat)->chunkLenPadded,&(SELL(mat)->chunkStart),&(SELL(mat)->val),&(mat->col_orig),src,mat,mat->traits.C,mat->traits.T),err,ret);

    if (ret != GHOST_SUCCESS) {
      goto err;
    }
 
    GHOST_CALL_GOTO(mat->split(mat),err,ret);

    //copy all values since the values will be modified in next call
     ghost_lidx *sell_col;
     GHOST_CALL_GOTO(ghost_malloc((void **)&sell_col, mat->nEnts*sizeof(ghost_lidx)),err,ret);

    for(int i=0;i<mat->nEnts;++i) {
        sell_col[i] = SELL(mat)->col[i];
    }

    ghost_lidx *sell_chunkStart, *sell_rowLen;
    ghost_lidx nchunks = (ghost_lidx)(ceil((double)mat->nrows/(double)mat->traits.C));
    GHOST_CALL_GOTO(ghost_malloc((void **)&sell_chunkStart, (nchunks+1)*sizeof(ghost_lidx)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&sell_rowLen, (mat->nrows)*sizeof(ghost_lidx)),err,ret);
 
    for(int i=0;i<nchunks+1; ++i){
        sell_chunkStart[i] = SELL(mat)->chunkStart[i];
    }

    char *sell_val;
    GHOST_CALL_GOTO(ghost_malloc((void **)&sell_val, mat->nEnts*mat->elSize*sizeof(char)),err,ret);

        ghost_lidx i;
    for(i=0;i<(ghost_lidx)(mat->nEnts*mat->elSize);++i) {
            sell_val[i] = (char)SELL(mat)->val[i];
    }

    for(i=0;i<mat->nrows;++i) {
            sell_rowLen[i] = SELL(mat)->rowLen[i];
    }

    ghost_sparsemat_rowfunc_after_split_arg after_split_arg;
    after_split_arg.col = sell_col;
    after_split_arg.val = sell_val;
    after_split_arg.chunk_ptr = sell_chunkStart;
    after_split_arg.rowLen = sell_rowLen;
    after_split_arg.dtsize = mat->elSize;
    after_split_arg.offs = mat->context->lfRow[me];
    after_split_arg.CHUNKHEIGHT = mat->traits.C;
    //create new src function
    ghost_sparsemat_src_rowfunc after_split =  GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER; 
    after_split.func = ghost_sparsemat_rowfunc_after_split_func;
    after_split.maxrowlen = mat->maxRowLen;
    after_split.base = 0;
    after_split.flags= GHOST_SPARSEMAT_ROWFUNC_DEFAULT;
    after_split.arg = &after_split_arg; 
       
    free(SELL(mat)->chunkStart); SELL(mat)->chunkStart=NULL;
    free(SELL(mat)->val); SELL(mat)->val=NULL;
    //free(mat->col_orig); mat->col_orig=NULL;//don't destroy will be used for printing
    free(SELL(mat)->chunkMin); SELL(mat)->chunkMin=NULL;
    free(SELL(mat)->chunkLen); SELL(mat)->chunkLen=NULL;
    free(SELL(mat)->chunkLenPadded); SELL(mat)->chunkLenPadded=NULL;
    free(SELL(mat)->rowLen); SELL(mat)->rowLen=NULL;
    free(SELL(mat)->rowLenPadded); SELL(mat)->rowLenPadded=NULL;

    if (!SELL(mat)->chunkMin) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkMin, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!SELL(mat)->chunkLen) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLen, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!SELL(mat)->chunkLenPadded) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->chunkLenPadded, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!SELL(mat)->rowLen) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLen, (mat->nrowsPadded)*sizeof(ghost_lidx)),err,ret);
    if (!SELL(mat)->rowLenPadded) GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_lidx)),err,ret);

    ghost_gidx *new_col = NULL;
    //GHOST_CALL_GOTO(ghost_malloc((void **)&new_col, mat->nEnts*sizeof(ghost_gidx)),err,ret); //will be allocated in the call

    GHOST_CALL_GOTO(ghost_sparsemat_fromfunc_common(SELL(mat)->rowLen,SELL(mat)->rowLenPadded,SELL(mat)->chunkLen,SELL(mat)->chunkLenPadded,&(SELL(mat)->chunkStart),&(SELL(mat)->val),&new_col,&after_split,mat,mat->traits.C,mat->traits.T),err,ret);

 
    free(SELL(mat)->col); SELL(mat)->col=NULL;

    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->col, mat->nEnts*sizeof(ghost_lidx)),err,ret);

    for(i=0;i<mat->nEnts;++i) {
      SELL(mat)->col[i] = (ghost_lidx) new_col[i];
    }
 
    //mat->col_orig = new_col;
    free(new_col);
    free(sell_col);
    free(sell_val);
    free(sell_chunkStart); 
    free(sell_rowLen);
    
  } else {

    mat->context->nrowspadded = PAD(mat->context->lnrows[me],ghost_densemat_row_padding());
    GHOST_CALL_GOTO(ghost_sparsemat_fromfunc_common(SELL(mat)->rowLen,SELL(mat)->rowLenPadded,SELL(mat)->chunkLen,SELL(mat)->chunkLenPadded,&(SELL(mat)->chunkStart),&(SELL(mat)->val),&(mat->col_orig),src,mat,mat->traits.C,mat->traits.T),err,ret);
    if (ret != GHOST_SUCCESS) {
       goto err;
    }
    GHOST_CALL_GOTO(mat->split(mat),err,ret);
 }

if(mat->traits.flags & GHOST_SOLVER_KACZ) {

  initHaloAvg(mat);

  //split transition zones 
  if(mat->traits.flags & (ghost_sparsemat_flags)GHOST_SPARSEMAT_BLOCKCOLOR) {
    split_transition(mat);
  } 
  //split if no splitting was done before and MC is off
  else if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
    if( (mat->kaczRatio >= 2*mat->kacz_setting.active_threads) ) {
      ghost_rcm_dissect(mat);
    } else {
      split_analytical(mat);
    }
  }
}

#ifdef GHOST_HAVE_CUDA
    if (!(mat->traits.flags & GHOST_SPARSEMAT_HOST))
        mat->upload(mat);
#endif


    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLen2,mat->nrowsPadded/2*sizeof(ghost_lidx)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&SELL(mat)->rowLen4,mat->nrowsPadded/4*sizeof(ghost_lidx)),err,ret);
    ghost_lidx max4,max2,i;
    for (i=0; i<mat->nrowsPadded; i++) {
        if (!(i%2)) {
            max2 = 0;
        }
        if (!(i%4)) {
            max4 = 0;
        }
        if (SELL(mat)->rowLen[i] > max2) {
            max2 = SELL(mat)->rowLen[i];
        }
        if (SELL(mat)->rowLen[i] > max4) {
            max4 = SELL(mat)->rowLen[i];
        }
        if (!((i+1)%2)) {
            SELL(mat)->rowLen2[i/2] = max2;
        }
        if (!((i+1)%4)) {
            SELL(mat)->rowLen4[i/4] = max4;
        }
    }
        
        

    goto out;
err:
    free(SELL(mat)->val); SELL(mat)->val = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
    free(SELL(mat)->chunkMin); SELL(mat)->chunkMin = NULL;
    free(SELL(mat)->chunkLen); SELL(mat)->chunkLen = NULL;
    free(SELL(mat)->chunkLenPadded); SELL(mat)->chunkLenPadded = NULL;
    free(SELL(mat)->rowLen); SELL(mat)->rowLen = NULL;
    free(SELL(mat)->rowLenPadded); SELL(mat)->rowLenPadded = NULL;
    free(SELL(mat)->chunkStart); SELL(mat)->chunkStart = NULL;
    mat->nEnts = 0;
    mat->nnz = 0;

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;

}

static ghost_error SELL_split(ghost_sparsemat *mat)
{

    if (!mat) {
        ERROR_LOG("Matrix is NULL");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_error ret = GHOST_SUCCESS;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);


    ghost_sell *fullSELL = SELL(mat);
    ghost_sell *localSELL = NULL, *remoteSELL = NULL;
    DEBUG_LOG(1,"Splitting the SELL matrix into a local and remote part");
    ghost_gidx i,j;
    int me;
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));

    ghost_lidx lnEnts_l, lnEnts_r;
    ghost_lidx current_l, current_r;


    ghost_lidx chunk;
    ghost_lidx idx, row;

    GHOST_INSTR_START("init_compressed_cols");
#ifdef GHOST_IDX_UNIFORM
    if (!(mat->traits.flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
        DEBUG_LOG(1,"In-place column compression!");
        SELL(mat)->col = mat->col_orig;
    } else 
#endif
    {
        if (!SELL(mat)->col) {
            DEBUG_LOG(1,"Duplicate col array!");
            GHOST_CALL_GOTO(ghost_malloc_align((void **)&SELL(mat)->col,sizeof(ghost_lidx)*mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
#pragma omp parallel for private(j) schedule(runtime)
            for (i=0; i<mat->nrowsPadded/mat->traits.C; i++) {
                for (j=SELL(mat)->chunkStart[i]; j<SELL(mat)->chunkStart[i+1]; j++) {
                    SELL(mat)->col[j] = 0;
                }
            }
        }
    }
    GHOST_INSTR_STOP("init_compressed_cols");
   
    GHOST_CALL_GOTO(ghost_context_comm_init(mat->context,mat->col_orig,mat,fullSELL->col),err,ret);

#ifndef GHOST_IDX_UNIFORM
    if (!(mat->traits.flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
        DEBUG_LOG(1,"Free orig cols");
        free(mat->col_orig);
        mat->col_orig = NULL;
    }
#endif
    if (!(mat->traits.flags & GHOST_SPARSEMAT_NOT_STORE_SPLIT)) { // split computation
        GHOST_INSTR_START("split");

        ghost_sparsemat_create(&(mat->localPart),mat->context,&mat->splittraits[0],1);
        localSELL = mat->localPart->sell;
        mat->localPart->traits.symmetry = mat->traits.symmetry;

        ghost_sparsemat_create(&(mat->remotePart),mat->context,&mat->splittraits[1],1);
        remoteSELL = mat->remotePart->sell; 

        mat->localPart->traits.T = mat->traits.T;
        mat->remotePart->traits.T = mat->traits.T;

        ghost_lidx nChunks = mat->nrowsPadded/mat->traits.C;
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkStart, (nChunks+1)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkMin, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkLen, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->chunkLenPadded, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->rowLen, (mat->nrowsPadded)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_lidx)),err,ret);

        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkStart, (nChunks+1)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkMin, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkLen, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->chunkLenPadded, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->rowLen, (mat->nrowsPadded)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->rowLenPadded, (mat->nrowsPadded)*sizeof(ghost_lidx)),err,ret);

#pragma omp parallel for schedule(runtime)
        for (i=0; i<mat->nrowsPadded; i++) {
            localSELL->rowLen[i] = 0;
            remoteSELL->rowLen[i] = 0;
            localSELL->rowLenPadded[i] = 0;
            remoteSELL->rowLenPadded[i] = 0;
        }

#pragma omp parallel for schedule(runtime)
        for(chunk = 0; chunk < mat->nrowsPadded/mat->traits.C; chunk++) {
            localSELL->chunkLen[chunk] = 0;
            remoteSELL->chunkLen[chunk] = 0;
            localSELL->chunkLenPadded[chunk] = 0;
            remoteSELL->chunkLenPadded[chunk] = 0;
            localSELL->chunkMin[chunk] = 0;
            remoteSELL->chunkMin[chunk] = 0;
        }
        localSELL->chunkStart[0] = 0;
        remoteSELL->chunkStart[0] = 0;

        mat->localPart->nnz = 0;
        mat->remotePart->nnz = 0;

        lnEnts_l = 0;
        lnEnts_r = 0;

        for(chunk = 0; chunk < mat->nrowsPadded/mat->traits.C; chunk++) {

            for (i=0; i<fullSELL->chunkLen[chunk]; i++) {
                for (j=0; j<mat->traits.C; j++) {
                    row = chunk*mat->traits.C+j;
                    idx = fullSELL->chunkStart[chunk]+i*mat->traits.C+j;

                    if (i < fullSELL->rowLen[row]) {
                        if (fullSELL->col[idx] < mat->context->lnrows[me]) {
                            localSELL->rowLen[row]++;
                            mat->localPart->nnz++;
                        } else {
                            remoteSELL->rowLen[row]++;
                            mat->remotePart->nnz++;
                        }
                        localSELL->rowLenPadded[row] = PAD(localSELL->rowLen[row],mat->localPart->traits.T);
                        remoteSELL->rowLenPadded[row] = PAD(remoteSELL->rowLen[row],mat->remotePart->traits.T);
                    }
                }
            }

            for (j=0; j<mat->traits.C; j++) {
                row = chunk*mat->traits.C+j;
                localSELL->chunkLen[chunk] = MAX(localSELL->chunkLen[chunk],localSELL->rowLen[row]);
                remoteSELL->chunkLen[chunk] = MAX(remoteSELL->chunkLen[chunk],remoteSELL->rowLen[row]);
            }
            lnEnts_l += localSELL->chunkLen[chunk]*mat->traits.C;
            lnEnts_r += remoteSELL->chunkLen[chunk]*mat->traits.C;
            localSELL->chunkStart[chunk+1] = lnEnts_l;
            remoteSELL->chunkStart[chunk+1] = lnEnts_r;

            localSELL->chunkLenPadded[chunk] = PAD(localSELL->chunkLen[chunk],mat->localPart->traits.T);
            remoteSELL->chunkLenPadded[chunk] = PAD(remoteSELL->chunkLen[chunk],mat->remotePart->traits.T);

        }



        /*
           for (i=0; i<fullSELL->nEnts;i++) {
           if (fullSELL->col[i]<mat->context->lnrows[me]) lnEnts_l++;
           }
           lnEnts_r = mat->context->lnEnts[me]-lnEnts_l;*/


        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->val,lnEnts_l*mat->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&localSELL->col,lnEnts_l*sizeof(ghost_lidx)),err,ret); 

        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->val,lnEnts_r*mat->elSize),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteSELL->col,lnEnts_r*sizeof(ghost_lidx)),err,ret); 

        mat->localPart->nrows = mat->nrows;
        mat->localPart->nrowsPadded = mat->nrowsPadded;
        mat->localPart->nEnts = lnEnts_l;
        mat->localPart->traits.C = mat->traits.C;

        mat->remotePart->nrows = mat->nrows;
        mat->remotePart->nrowsPadded = mat->nrowsPadded;
        mat->remotePart->nEnts = lnEnts_r;
        mat->remotePart->traits.C = mat->traits.C;

#pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < mat->localPart->nrowsPadded/mat->localPart->traits.C; chunk++) {
            for (i=0; i<localSELL->chunkLenPadded[chunk]; i++) {
                for (j=0; j<mat->localPart->traits.C; j++) {
                    idx = localSELL->chunkStart[chunk]+i*mat->localPart->traits.C+j;
                    memset(&((char *)(localSELL->val))[idx*mat->elSize],0,mat->elSize);
                    localSELL->col[idx] = 0;
                }
            }
        }

#pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < mat->remotePart->nrowsPadded/mat->remotePart->traits.C; chunk++) {
            for (i=0; i<remoteSELL->chunkLenPadded[chunk]; i++) {
                for (j=0; j<mat->remotePart->traits.C; j++) {
                    idx = remoteSELL->chunkStart[chunk]+i*mat->remotePart->traits.C+j;
                    memset(&((char *)(remoteSELL->val))[idx*mat->elSize],0,mat->elSize);
                    remoteSELL->col[idx] = 0;
                }
            }
        }

        current_l = 0;
        current_r = 0;
        ghost_lidx *col_l, *col_r;
        ghost_malloc((void **)&col_l,sizeof(ghost_lidx)*mat->traits.C);
        ghost_malloc((void **)&col_r,sizeof(ghost_lidx)*mat->traits.C);

        for(chunk = 0; chunk < mat->nrowsPadded/mat->traits.C; chunk++) {

            for (j=0; j<mat->traits.C; j++) {
                col_l[j] = 0;
                col_r[j] = 0;
            }

            for (i=0; i<fullSELL->chunkLen[chunk]; i++) {
                for (j=0; j<mat->traits.C; j++) {
                    row = chunk*mat->traits.C+j;
                    idx = fullSELL->chunkStart[chunk]+i*mat->traits.C+j;

                    if (i<fullSELL->rowLen[row]) {
                        if (fullSELL->col[idx] < mat->context->lnrows[me]) {
                            if (col_l[j] < localSELL->rowLen[row]) {
                                ghost_lidx lidx = localSELL->chunkStart[chunk]+col_l[j]*mat->localPart->traits.C+j;
                                localSELL->col[lidx] = fullSELL->col[idx];
                                memcpy(&localSELL->val[lidx*mat->elSize],&fullSELL->val[idx*mat->elSize],mat->elSize);
                                current_l++;
                            }
                            col_l[j]++;
                        }
                        else{
                            if (col_r[j] < remoteSELL->rowLen[row]) {
                                ghost_lidx ridx = remoteSELL->chunkStart[chunk]+col_r[j]*mat->remotePart->traits.C+j;
                                remoteSELL->col[ridx] = fullSELL->col[idx];
                                memcpy(&remoteSELL->val[ridx*mat->elSize],&fullSELL->val[idx*mat->elSize],mat->elSize);
                                current_r++;
                            }
                            col_r[j]++;
                        }
                    }
                }
            }
        }

        free(col_l);
        free(col_r);

#ifdef GHOST_HAVE_CUDA
        if (!(mat->traits.flags & GHOST_SPARSEMAT_HOST)) {
            mat->localPart->upload(mat->localPart);
            mat->remotePart->upload(mat->remotePart);
        }
#endif
        GHOST_INSTR_STOP("split");
    }

    goto out;
err:
    ghost_sparsemat_destroy(mat->localPart); mat->localPart = NULL;
    ghost_sparsemat_destroy(mat->remotePart); mat->remotePart = NULL;

out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;
}

static ghost_error SELL_toBinCRS(ghost_sparsemat *mat, char *matrixPath)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_IO);
    UNUSED(mat);
    UNUSED(matrixPath);

    ERROR_LOG("SELL matrix to binary CRS file not implemented");
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_IO);
    return GHOST_ERR_NOT_IMPLEMENTED;
}

static ghost_error SELL_upload(ghost_sparsemat* mat) 
{
#ifdef GHOST_HAVE_CUDA
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    if (!(mat->traits.flags & GHOST_SPARSEMAT_HOST)) {
        DEBUG_LOG(1,"Creating matrix on CUDA device");
        GHOST_CALL_RETURN(ghost_malloc((void **)&SELL(mat)->cumat,sizeof(ghost_cu_sell)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->rowLen,(mat->nrows)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->rowLenPadded,(mat->nrows)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->col,(mat->nEnts)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->val,(mat->nEnts)*mat->elSize));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->chunkStart,(mat->nrowsPadded/mat->traits.C+1)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&SELL(mat)->cumat->chunkLen,(mat->nrowsPadded/mat->traits.C)*sizeof(ghost_lidx)));

        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->rowLen, SELL(mat)->rowLen, mat->nrows*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->rowLenPadded, SELL(mat)->rowLenPadded, mat->nrows*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->col, SELL(mat)->col, mat->nEnts*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->val, SELL(mat)->val, mat->nEnts*mat->elSize));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->chunkStart, SELL(mat)->chunkStart, (mat->nrowsPadded/mat->traits.C+1)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_upload(SELL(mat)->cumat->chunkLen, SELL(mat)->chunkLen, (mat->nrowsPadded/mat->traits.C)*sizeof(ghost_lidx)));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
#else
    if (mat->traits.flags & GHOST_SPARSEMAT_DEVICE) {
        ERROR_LOG("Device matrix cannot be created without CUDA");
        return GHOST_ERR_CUDA;
    }
#endif
    return GHOST_SUCCESS;
}

