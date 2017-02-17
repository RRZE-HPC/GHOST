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
//#include "ghost/kacz_split_analytical.h"
#include "ghost/rcm_dissection.h"
#include <libgen.h>
#include <math.h>
#include <limits.h>

const ghost_sparsemat_src_rowfunc GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER = {
    .func = NULL,
    .maxrowlen = 0,
    .base = 0,
    .flags = GHOST_SPARSEMAT_ROWFUNC_DEFAULT,
    .arg = NULL,
    .gnrows = 0,
    .gncols = 0
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

static const char * ghost_sparsemat_formatName(ghost_sparsemat *mat);
static ghost_error ghost_sparsemat_split(ghost_sparsemat *mat);
#ifdef GHOST_HAVE_CUDA
static ghost_error ghost_sparsemat_upload(ghost_sparsemat *mat);
#endif
static ghost_error ghost_set_kacz_ratio(ghost_context *ctx, ghost_sparsemat *mat); 

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
    if (context) {
        context->nmats++;
    }

    (*mat)->localPart = NULL;
    (*mat)->remotePart = NULL;
    (*mat)->name = "Sparse matrix";
    (*mat)->col_orig = NULL;
    (*mat)->nzDist = NULL;
    (*mat)->avgRowBand = 0.;
    (*mat)->avgAvgRowBand = 0.;
    (*mat)->smartRowBand = 0.;
    (*mat)->maxRowLen = 0;
    (*mat)->nMaxRows = 0;
    (*mat)->variance = 0.;
    (*mat)->deviation = 0.;
    (*mat)->cv = 0.;
//    (*mat)->ncols = context->col_map->dim;
    (*mat)->nEnts = 0;
    //(*mat)->nnz = 0;
   
    // TODO do this in the actual sorting function
    /* 
    if ((*mat)->traits.sortScope == GHOST_SPARSEMAT_SORT_GLOBAL) {
        (*mat)->traits.sortScope = (*mat)->context->row_map->gdim;
    } else if ((*mat)->traits.sortScope == GHOST_SPARSEMAT_SORT_LOCAL) {
        (*mat)->traits.sortScope = (*mat)->context->row_map->dim;
    }
    */
    
    // Note: Datatpye check and elSize computation moved to creation
    // functions ghost_sparsemat_init_*
    (*mat)->elSize = 0;
    
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
    
    (*mat)->val = NULL;
    (*mat)->col = NULL;
    (*mat)->chunkMin = NULL;
    (*mat)->chunkLen = NULL;
    (*mat)->chunkLenPadded = NULL;
    (*mat)->rowLen = NULL;
    (*mat)->rowLen2 = NULL;
    (*mat)->rowLen4 = NULL;
    (*mat)->rowLenPadded = NULL;
    (*mat)->chunkStart = NULL;
    
    
    goto out;
    err:
    ERROR_LOG("Error. Free'ing resources");
    free(*mat); *mat = NULL;
    
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


//calculates bandwidth of the matrix mat with possible permutations applied from ctx and stores the information in ctx
static ghost_error ghost_calculate_bw(ghost_context *ctx, ghost_sparsemat *mat) 
{
    GHOST_INSTR_START("calculate badwidth");
    ghost_error ret = GHOST_SUCCESS;
    int me;     
    GHOST_CALL_GOTO(ghost_rank(&me,ctx->mpicomm),err,ret);
    
    ghost_gidx lower_bw = 0, upper_bw = 0, max_col=0;
    
    #pragma omp parallel for reduction(max:lower_bw) reduction(max:upper_bw) reduction(max:max_col)
    for (ghost_lidx i=0; i<ctx->row_map->ldim[me]; i++) {
        
        ghost_lidx orig_row = i;
        if (ctx->row_map->loc_perm) {
            orig_row = ctx->row_map->loc_perm_inv[i];
        }
        ghost_lidx * col = &mat->col[mat->chunkStart[orig_row]];
        ghost_lidx orig_row_len = mat->chunkStart[orig_row+1]-mat->chunkStart[orig_row];

        ghost_gidx start_col = INT_MAX;
        ghost_gidx end_col   = 0;
        
        if(ctx->row_map->loc_perm){
            if(ctx->col_map->loc_perm == NULL) {
                for(int j=0; j<orig_row_len; ++j) {
                    start_col = MIN(start_col, ctx->row_map->loc_perm[col[j]]);
                    end_col   = MAX(end_col, ctx->row_map->loc_perm[col[j]]);
                }
            } else {
                for(int j=0; j<orig_row_len; ++j) {
                    start_col = MIN(start_col, ctx->col_map->loc_perm[col[j]]);
                    end_col   = MAX(end_col, ctx->col_map->loc_perm[col[j]]);
                }
            }
        } else {
            for(int j=0; j<orig_row_len; ++j) {
                start_col = MIN(start_col, col[j]);
                end_col   = MAX(end_col, col[j]);
            }
        }
        lower_bw = MAX(lower_bw, i-start_col);
        upper_bw = MAX(upper_bw, end_col - i);
        max_col    = MAX(max_col, end_col);
    }
    ctx->lowerBandwidth = lower_bw;
    ctx->upperBandwidth = upper_bw;
    ctx->bandwidth      = lower_bw + upper_bw;
    ctx->maxColRange    = max_col;
    
    ctx->bandwidth = ctx->lowerBandwidth + ctx->upperBandwidth;
    INFO_LOG("RANK<%d>:  LOWER BANDWIDTH =%"PRGIDX", UPPER BANDWIDTH =%"PRGIDX", TOTAL BANDWIDTH =%"PRGIDX,me,ctx->lowerBandwidth,ctx->upperBandwidth,ctx->bandwidth);
    GHOST_INSTR_STOP("calculate bandwidth");
    goto out;
    
    err: 
    ERROR_LOG("ERROR in Bandwidth Calculation");
    return ret;
    out:
    return ret;
}

static ghost_error ghost_set_kacz_ratio(ghost_context *ctx, ghost_sparsemat *mat) 
{
    int nthread;
    
    #ifdef GHOST_HAVE_OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        nthread = ghost_omp_nthread();
    }
    #else
    nthread = 1;
    #endif
    
    ctx->kacz_setting.active_threads = nthread;
    ghost_calculate_bw(ctx,mat);
    ctx->kaczRatio = ((double)SPM_NROWS(mat))/ctx->bandwidth;
    return GHOST_SUCCESS;
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
            if ((colsfromi[el] >= context->row_map->goffs[me]) && (colsfromi[el] < (context->row_map->goffs[me]+context->row_map->ldim[me]))) {
                colsfromi[el] = context->row_map->glb_perm[colsfromi[el]-context->row_map->goffs[me]];
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
            if (context->row_map->loc_perm) {
                for (el=0; el<nels; el++) {
                    if ((colsfromi[el] >= context->row_map->goffs[me]) && (colsfromi[el] < context->row_map->goffs[me]+context->row_map->ldim[me])) {
                        col[el] = context->row_map->goffs[me] + context->row_map->loc_perm[colsfromi[el]-context->row_map->goffs[me]];
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
    
    *nrows = mat->context->row_map->gdim;
    
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
    /*  ghost_gidx lnnz = SPM_NNZ(mat);
     * 
     * #ifdef GHOST_HAVE_MPI
     *    MPI_CALL_RETURN(MPI_Allreduce(&lnnz,nnz,1,ghost_mpi_dt_gidx,MPI_SUM,mat->context->mpicomm));
     * #else
     *nnz = lnnz;
     * #endif
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
    ghost_line_string(str,"Bandwidth",NULL,"%"PRGIDX,mat->context->bandwidth);
    ghost_line_string(str,"Avg. row band",NULL,"%.3f",mat->avgRowBand);
    ghost_line_string(str,"Avg. avg. row band",NULL,"%.3f",mat->avgAvgRowBand);
    ghost_line_string(str,"Smart row band",NULL,"%.3f",mat->smartRowBand);
    
    ghost_line_string(str,"Local number of rows",NULL,"%"PRLIDX,SPM_NROWS(mat));
    ghost_line_string(str,"Local number of rows (padded)",NULL,"%"PRLIDX,SPM_NROWSPAD(mat));
    ghost_line_string(str,"Local number of nonzeros",NULL,"%"PRLIDX,SPM_NNZ(mat));
    
    ghost_line_string(str,"Full   matrix format",NULL,"%s",ghost_sparsemat_formatName(mat));
    if (mat->localPart) {
        ghost_line_string(str,"Local  matrix format",NULL,"%s",ghost_sparsemat_formatName(mat->localPart));
        ghost_line_string(str,"Local  matrix symmetry",NULL,"%s",ghost_sparsemat_symmetry_string(mat->localPart->traits.symmetry));
        ghost_line_string(str,"Local  matrix size","MB","%u",ghost_sparsemat_bytesize(mat->localPart)/(1024*1024));
    }
    if (mat->remotePart) {
        ghost_line_string(str,"Remote matrix format",NULL,"%s",ghost_sparsemat_formatName(mat->remotePart));
        ghost_line_string(str,"Remote matrix size","MB","%u",ghost_sparsemat_bytesize(mat->remotePart)/(1024*1024));
    }
    
    ghost_line_string(str,"Full   matrix size","MB","%u",ghost_sparsemat_bytesize(mat)/(1024*1024));
    
    if (mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY) {
        ghost_line_string(str,"Permuted",NULL,"Yes");
        if (mat->context->row_map->glb_perm) {
            if (mat->context->row_map->loc_perm) {
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
        } else if (mat->context->row_map->loc_perm) {
            ghost_line_string(str,"Permutation scope",NULL,"Local");
        }
        if (mat->context->row_map->loc_perm) {
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
    ghost_line_string(str,"Chunk occupancy (beta)",NULL,"%f",(double)(SPM_NNZ(mat))/(double)(mat->nEnts));
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
    #ifdef GHOST_HAVE_CUDA
    if (mat->traits.flags & GHOST_SPARSEMAT_DEVICE) {
        ghost_cu_free(mat->cu_rowLen);
        ghost_cu_free(mat->cu_rowLenPadded);
        ghost_cu_free(mat->cu_col);
        ghost_cu_free(mat->cu_val);
        ghost_cu_free(mat->cu_chunkStart);
        ghost_cu_free(mat->cu_chunkLen);
    }
    #endif
    free(mat->val); mat->val = NULL;
    free(mat->col); mat->col = NULL;
    free(mat->chunkStart); mat->chunkStart = NULL;
    free(mat->chunkMin); mat->chunkMin = NULL;
    free(mat->chunkLen); mat->chunkLen = NULL;
    free(mat->chunkLenPadded); mat->chunkLenPadded = NULL;
    free(mat->rowLen); mat->rowLen = NULL;
    free(mat->rowLen2); mat->rowLen2 = NULL;
    free(mat->rowLen4); mat->rowLen4 = NULL;
    free(mat->rowLenPadded); mat->rowLenPadded = NULL;
   
    mat->context->nmats--;
    if (mat->context->nmats == 0) {
        ghost_context_destroy(mat->context);
    }
    
    if (mat->localPart) {
        ghost_sparsemat_destroy(mat->localPart);
    }
    
    if (mat->remotePart) {
        ghost_sparsemat_destroy(mat->remotePart);
    }
    
    
    free(mat->col_orig); mat->col_orig = NULL;
    
    free(mat);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_TEARDOWN);
}

ghost_error ghost_sparsemat_init_bin(ghost_sparsemat *mat, char *path, ghost_mpi_comm mpicomm, double weight)
{
    PERFWARNING_LOG("The current implementation of binCRS read-in is "
    "inefficient in terms of memory consumption!");
    
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    
    ghost_error ret = GHOST_SUCCESS;
    ghost_sparsemat_rowfunc_file_initargs args;
    ghost_gidx dim[2];
    ghost_lidx bincrs_dt = 0; // or use args.dt directly...
    ghost_sparsemat_src_rowfunc src = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    
    src.func = &ghost_sparsemat_rowfunc_bincrs;
    src.arg = &args; 
   
    args.mat = mat; 
    args.filename = path;
    
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_BINCRS_ROW_GETDIM,&bincrs_dt,dim,NULL,src.arg)) {
        ERROR_LOG("Error in matrix creation function");
        ret = GHOST_ERR_UNKNOWN;
        goto err;
    }
   
    // Apply file datatype only if still unspecified.
    if(mat->traits.datatype == GHOST_DT_NONE) mat->traits.datatype = (ghost_datatype)bincrs_dt;
    // Require valid datatype here.
    GHOST_CALL_GOTO(ghost_datatype_size(&mat->elSize,mat->traits.datatype),err,ret);   
    args.dt = mat->traits.datatype;
    
    
    src.gnrows = dim[0];
    src.gncols = dim[1];
    src.maxrowlen = dim[1];
    
    GHOST_CALL_GOTO(ghost_sparsemat_init_rowfunc(mat,&src,mpicomm,weight),err,ret);
    
    goto out;
    err:
    
    out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    return ret;
    
}

ghost_error ghost_sparsemat_init_mm(ghost_sparsemat *mat, char *path, ghost_mpi_comm mpicomm, double weight)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    
    ghost_error ret = GHOST_SUCCESS;
    ghost_sparsemat_rowfunc_file_initargs args;
    ghost_gidx dim[2];
    ghost_lidx bincrs_dt = 0;
    ghost_sparsemat_src_rowfunc src = GHOST_SPARSEMAT_SRC_ROWFUNC_INITIALIZER;
    
    if (mat->traits.flags & GHOST_SPARSEMAT_TRANSPOSE_MM) { 
        src.func = &ghost_sparsemat_rowfunc_mm_transpose;
    } else {
        src.func = &ghost_sparsemat_rowfunc_mm;
    }
    
    src.arg = &args; 
    args.filename = path;
    args.mat = mat;
    
    if (src.func(GHOST_SPARSEMAT_ROWFUNC_MM_ROW_GETDIM,&bincrs_dt,dim,NULL,src.arg)) {
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
    
    src.gnrows = dim[0];
    src.gncols = dim[1];
    src.maxrowlen = dim[1];

    
    GHOST_CALL_GOTO(ghost_sparsemat_init_rowfunc(mat,&src,mpicomm,weight),err,ret);
    
    goto out;
    err:
    
    out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION|GHOST_FUNCTYPE_IO);
    return ret;
    
}

extern inline int ghost_sparsemat_rowfunc_crs(ghost_gidx row, ghost_lidx *rowlen, ghost_gidx *col, void *val, void *arg);

ghost_error ghost_sparsemat_init_crs(ghost_sparsemat *mat, ghost_gidx offs, ghost_lidx n, ghost_gidx *col, void *val, ghost_lidx *rpt, ghost_mpi_comm mpicomm, double weight)
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
    
    GHOST_CALL_GOTO(ghost_sparsemat_init_rowfunc(mat,&src,mpicomm,weight),err,ret);
    
    goto out;
    err:
    
    out:
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);
    return ret;
    
}

static const char * ghost_sparsemat_formatName(ghost_sparsemat *mat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    // TODO format SELL-C-sigma
    UNUSED(mat);
    return "SELL";
}

size_t ghost_sparsemat_bytesize (ghost_sparsemat *mat)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_UTIL);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_UTIL);
    return (size_t)((SPM_NROWSPAD(mat)/mat->traits.C)*sizeof(ghost_lidx) + 
    mat->nEnts*(sizeof(ghost_lidx)+mat->elSize));
}

static ghost_error initHaloAvg(ghost_sparsemat *mat)
{
    ghost_error ret = GHOST_SUCCESS;
    int me,nprocs;
    GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
   
    ghost_context *ctx = mat->context; 
    ghost_lidx ctx_nrowspadded = ctx->col_map->dim; 
    bool *compression_flag;
    int *temp_nrankspresent;
    //calculate rankspresent here and store it, no need to do this each time averaging is done
    GHOST_CALL_GOTO(ghost_malloc((void **)&temp_nrankspresent, ctx_nrowspadded*sizeof(int)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&compression_flag, ctx_nrowspadded*sizeof(bool)),err,ret);
    
    #pragma omp parallel for schedule(runtime)
    for (int i=0; i<ctx_nrowspadded; i++) {	
        if(ctx->col_map->loc_perm) {
            if(ctx->col_map->loc_perm_inv[i]< ctx->row_map->dim ) { //This check is important since entsInCol has only lnrows(NO_DISTINCTION
                //might give seg fault else) the rest are halo anyway, not needed for local sums
                temp_nrankspresent[i] = ctx->entsInCol[ctx->col_map->loc_perm_inv[i]]?1:0; //this has also to be permuted since it was
            } else {
                //temp_nrankspresent[i] = 0;//ctx->entsInCol[i]?1:0;		
            }
        } else {
            if(i < ctx->row_map->ldim[me]) {
                temp_nrankspresent[i] = ctx->entsInCol[i]?1:0; //this has also to be permuted since it was
            } else {
                temp_nrankspresent[i] = 0;
            }
        } 
        
        compression_flag[i] = false;
    }
    
    ghost_lidx ndues = 0;
    for (int i=0; i<nprocs; i++) {
        if(ctx->row_map->loc_perm) {
            #pragma omp parallel for schedule(runtime) 
            for (int d=0 ;d < ctx->dues[i]; d++) {
                temp_nrankspresent[ctx->col_map->loc_perm[ctx->duelist[i][d]]]++; 
                compression_flag[ctx->col_map->loc_perm[ctx->duelist[i][d]]] = true;
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
    GHOST_CALL_GOTO(ghost_malloc((void **)&temp_avg_ptr, ctx_nrowspadded*sizeof(ghost_lidx)),err,ret);
    ghost_lidx ctr = 0;
    //count number of elements
    for(ghost_lidx i=0; i<ctx_nrowspadded; ++i) {
        if(ctr==0 && compression_flag[i]==true){
            temp_avg_ptr[ctr] = i; 
            ctr += 1; 
        }
        else if(ctr!=0 && (compression_flag[i-1]!=compression_flag[i])) {
            temp_avg_ptr[ctr] = i; 
            ctr += 1;
        } 
        else if(i==ctx_nrowspadded-1 && (compression_flag[i-1]==true)) {
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
    GHOST_CALL_GOTO(ghost_malloc((void **)&map, ctx_nrowspadded*sizeof(ghost_lidx)),err,ret); 
    
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
        if(ctx->row_map->loc_perm) {
            for (int d=0 ;d < ctx->dues[i]; d++) {
                ctx->mappedDuelist[ctr] = map[ ctx->col_map->loc_perm[ctx->duelist[i][d]] ]; 
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
     *   	for (int i=0; i<nprocs; i++) {
     *     for (int d=0 ;d < ctx->dues[i]; d++) {
     *        if(ctx->perm_local)
     *        printf("%d -> %d\n",ctx->perm_local->colPerm[ctx->duelist[i][d]], map[ctx->perm_local->colPerm[ctx->duelist[i][d]]]);
     *        else
     *        printf("%d -> %d\n",ctx->duelist[i][d], map[ctx->duelist[i][d]]);
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

ghost_error ghost_sparsemat_init_rowfunc(ghost_sparsemat *mat, ghost_sparsemat_src_rowfunc *src, ghost_mpi_comm mpicomm, double weight)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    ghost_error ret = GHOST_SUCCESS;

    int me,nprocs;
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mpicomm),err,ret);
    
    if (!(mat->context)) {
        ghost_context_create(&(mat->context),src->gnrows,src->gncols,GHOST_CONTEXT_DEFAULT,mpicomm,weight);
    } 
    if (!mat->context->row_map->dim) {
        ghost_map_create_distribution(mat->context->row_map,src,mat->context->weight,GHOST_MAP_DIST_NROWS,NULL);
    }
    if (!mat->context->col_map->dim) {
        ghost_map_create_distribution(mat->context->col_map,src,mat->context->weight,GHOST_MAP_DIST_NROWS,NULL);
    }
    if (mat->traits.flags & GHOST_SPARSEMAT_PERM_NO_DISTINCTION) {
        mat->context->col_map->flags = (ghost_map_flags)(mat->context->col_map->flags&GHOST_PERM_NO_DISTINCTION);
    }

    if (mat->traits.C == GHOST_SELL_CHUNKHEIGHT_ELLPACK) {
        mat->traits.C = PAD(SPM_NROWS(mat),GHOST_PAD_MAX);
    } else if (mat->traits.C == GHOST_SELL_CHUNKHEIGHT_AUTO){
        mat->traits.C = 32; // TODO
    }
    mat->nchunks = CEILDIV(SPM_NROWS(mat),mat->traits.C);
    //ERROR_LOG("set no_distinction");
    //mat->context->flags = mat->context->flags | GHOST_PERM_NO_DISTINCTION;
    if (mat->context->row_map->dimpad == mat->context->row_map->dim) {
        mat->context->row_map->dimpad = PAD(SPM_NROWS(mat),ghost_densemat_row_padding());
    }
    if (mat->context->col_map->dimpad == mat->context->col_map->dim) {
        mat->context->col_map->dimpad = PAD(mat->context->col_map->dim,ghost_densemat_row_padding());
    }
    
    ghost_lidx nChunks = CEILDIV(SPM_NROWS(mat),mat->traits.C);
    
    // Require valid datatpye here.
    GHOST_CALL_GOTO(ghost_datatype_size(&mat->elSize,mat->traits.datatype),err,ret);
   
    if (!mat->chunkMin) GHOST_CALL_GOTO(ghost_malloc((void **)&mat->chunkMin, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!mat->chunkLen) GHOST_CALL_GOTO(ghost_malloc((void **)&mat->chunkLen, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!mat->chunkLenPadded) GHOST_CALL_GOTO(ghost_malloc((void **)&mat->chunkLenPadded, (nChunks)*sizeof(ghost_lidx)),err,ret);
    if (!mat->rowLen) GHOST_CALL_GOTO(ghost_malloc((void **)&mat->rowLen, (SPM_NROWSPAD(mat))*sizeof(ghost_lidx)),err,ret);
    if (!mat->rowLenPadded) GHOST_CALL_GOTO(ghost_malloc((void **)&mat->rowLenPadded, (SPM_NROWSPAD(mat))*sizeof(ghost_lidx)),err,ret); 
    
    GHOST_CALL_GOTO(ghost_rank(&me,mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);


    //set NO_DISTINCTION when block multicolor and RCM is on and more than 2 processors, TODO pure MC and MPI
    //this has to be invoked even if no permutations are carried out and more than 2 processors, since we need to
    //know amount of remote entries before (used in sparsemat_blockcolor); 
    if(nprocs>1 && mat->traits.flags & GHOST_SOLVER_KACZ) {
        
        INFO_LOG("NO DISTINCTION is set");
        mat->context->flags |=   (ghost_context_flags_t) GHOST_PERM_NO_DISTINCTION; 
    }
    
    
    //mat->context->nrowspadded = PAD(mat->context->row_map->ldim[me],ghost_densemat_row_padding());
    
    ghost_lidx *rl = mat->rowLen;
    ghost_lidx *rlp = mat->rowLenPadded;
    ghost_lidx *cl = mat->chunkLen;
    ghost_lidx *clp = mat->chunkLenPadded;
    ghost_lidx ** chunkptr = &(mat->chunkStart);
    char **val = &(mat->val);
    ghost_gidx **col = &(mat->col_orig);
    ghost_lidx C = mat->traits.C;
    ghost_lidx P = mat->traits.T;

    int funcerrs = 0;
    char *tmpval = NULL;
    ghost_gidx *tmpcol = NULL;
    ghost_lidx i,row,chunk,colidx;
    ghost_gidx gnents = 0, gnnz = 0;
    ghost_lidx maxRowLenInChunk = 0, maxRowLen = 0, privateMaxRowLen = 0;
    
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    
    
    /*if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) 
     *        SPM_NCOLS(mat) = mat->context->nrowspadded; 
     *    else */
    
    #ifdef GHOST_SPARSEMAT_GLOBALSTATS
    GHOST_CALL_GOTO(ghost_malloc((void **)&(mat->nzDist),sizeof(ghost_gidx)*(2*mat->context->row_map->gdim-1)),err,ret);
    memset(mat->nzDist,0,sizeof(ghost_gidx)*(2*mat->context->row_map->gdim-1));
    #endif
    mat->context->lowerBandwidth = 0;
    mat->context->upperBandwidth = 0;
    
    if (mat->traits.sortScope > 1) {
        mat->traits.flags = (ghost_sparsemat_flags)(mat->traits.flags|GHOST_SPARSEMAT_SORT_ROWS);
    }
   
    // _Only_ global permutation:
    // Create dummymat without any permutation and create global permutation
    // based on this dummymat
    if ((mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY_GLOBAL) && 
            !(mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY_LOCAL)) {
        ghost_sparsemat *dummymat = NULL;
        ghost_sparsemat_traits mtraits = mat->traits;
        mtraits.flags = (ghost_sparsemat_flags)(mtraits.flags & ~(GHOST_SPARSEMAT_PERM_ANY_GLOBAL));
        mtraits.flags = (ghost_sparsemat_flags)(mtraits.flags | GHOST_SPARSEMAT_SAVE_ORIG_COLS);
        mtraits.C = 1;
        mtraits.sortScope = 1;
        ghost_sparsemat_create(&dummymat,NULL,&mtraits,1);
        ghost_sparsemat_init_rowfunc(dummymat,src,mat->context->mpicomm,mat->context->weight);

        if (mat->traits.flags & GHOST_SPARSEMAT_SCOTCHIFY) {
            ghost_sparsemat_perm_scotch(mat->context,dummymat);
        } 
        if (mat->traits.flags & GHOST_SPARSEMAT_ZOLTAN) {
            ghost_sparsemat_perm_zoltan(mat->context,dummymat);
        }
        ghost_sparsemat_destroy(dummymat);

    } 
    // Any combination of only local or global+local permutations:
    // Create dummymat with global permutations only and create local
    // permutations based on this dummymat
    else if (mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY) {
        ghost_sparsemat *dummymat = NULL;
        ghost_sparsemat_traits mtraits = mat->traits;
        mtraits.flags = (ghost_sparsemat_flags)(mtraits.flags & ~(GHOST_SPARSEMAT_PERM_ANY_LOCAL));
        mtraits.flags = (ghost_sparsemat_flags)(mtraits.flags | GHOST_SPARSEMAT_SAVE_ORIG_COLS);
        if (mat->traits.flags & GHOST_SOLVER_KACZ && nprocs > 1) {
            mtraits.flags = (ghost_sparsemat_flags)(mtraits.flags | GHOST_SPARSEMAT_PERM_NO_DISTINCTION);
        }
        mtraits.C = 1;
        mtraits.sortScope = 1;
        ghost_sparsemat_create(&dummymat,NULL,&mtraits,1);
        ghost_sparsemat_init_rowfunc(dummymat,src,mat->context->mpicomm,mat->context->weight);

        if (mat->traits.flags & GHOST_SPARSEMAT_RCM) { 
            ghost_sparsemat_perm_spmp(mat->context,dummymat);
        }
        if (mat->traits.flags & GHOST_SPARSEMAT_COLOR) {
            ghost_sparsemat_perm_color(mat->context,dummymat);
        }
        //blockcoloring needs to know bandwidth //TODO avoid 2 times calculating  bandwidth, if no RCM or no bandwidth disturbing permutations are done 
        //take this branch only if the matrix cannot be bandwidth bound, 
        //else normal splitting with just RCM permutation would do the work
        //check whether BLOCKCOLOR is necessary, it is avoided if user explicitly request Multicoloring method
        if(mat->traits.flags & GHOST_SOLVER_KACZ) {
            ghost_set_kacz_ratio(mat->context,dummymat);
            if(mat->context->kaczRatio < mat->context->kacz_setting.active_threads && !(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
                mat->traits.flags |= (ghost_sparsemat_flags)GHOST_SPARSEMAT_BLOCKCOLOR; 
            }
        }
        
        if (mat->traits.flags & GHOST_SPARSEMAT_BLOCKCOLOR) {
            ghost_sparsemat_blockColor(mat->context,dummymat);
        }
        
        if (mat->traits.sortScope > 1) {
            ghost_sparsemat_perm_sort(mat->context,dummymat,mat->traits.sortScope);
        }
        
        ghost_sparsemat_destroy(dummymat);

        if (mat->context->row_map->loc_perm && mat->context->col_map->loc_perm == NULL) {
            mat->context->col_map->cu_loc_perm = mat->context->row_map->cu_loc_perm;
            mat->context->col_map->loc_perm = mat->context->row_map->loc_perm;
            mat->context->col_map->loc_perm_inv = mat->context->row_map->loc_perm_inv;
        }
        if (mat->context->row_map->glb_perm && mat->context->col_map->glb_perm == NULL) {
            mat->context->col_map->glb_perm = mat->context->row_map->glb_perm;
            mat->context->col_map->glb_perm_inv = mat->context->row_map->glb_perm_inv;
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
    if (src->func == ghost_sparsemat_rowfunc_bincrs || src->func == ghost_sparsemat_rowfunc_mm) {
        if (src->func(GHOST_SPARSEMAT_ROWFUNC_INIT,NULL,NULL,NULL,src->arg)) {
            ERROR_LOG("Error in matrix creation function");
            ret = GHOST_ERR_UNKNOWN;
            goto err;
        }
    }
        
   
    
    ghost_lidx *tmpclp = NULL;
    if (!clp) {
        ghost_malloc((void **)&tmpclp,nChunks*sizeof(ghost_lidx));
        clp = tmpclp;
    }
    ghost_lidx *tmprl = NULL;
    if (!rl) {
        ghost_malloc((void **)&tmprl,nChunks*sizeof(ghost_lidx));
        rl = tmprl;
    }
    
    
    if (!(*chunkptr)) {
        GHOST_INSTR_START("rowlens");
        GHOST_CALL_GOTO(ghost_malloc_align((void **)chunkptr,(nChunks+1)*sizeof(ghost_lidx),GHOST_DATA_ALIGNMENT),err,ret);
        
    }
    #pragma omp parallel private(i,tmpval,tmpcol,row,maxRowLenInChunk) reduction (+:gnents,gnnz,funcerrs) reduction (max:privateMaxRowLen) 
    {
        ghost_lidx rowlen;
        maxRowLenInChunk = 0; 
        GHOST_CALL(ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx)),ret);
        
        /*if (!(mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY) && src->func == ghost_sparsemat_rowfunc_crs) {
         * #pragma omp single
         *                INFO_LOG("Fast matrix construction for CRS source and no permutation") 
         * #pragma omp for schedule(runtime)
         *                for( chunk = 0; chunk < nChunks; chunk++ ) {
         *                    chunkptr[chunk] = 0; // NUMA init
         *                    for (i=0, row = chunk*C; i < C && row < SPM_NROWS(mat); i++, row++) {
         * 
         *                        rowlen=((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt[mat->context->row_map->goffs[me]+row+1]-((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt[mat->context->row_map->goffs[me]+row];
         * 
         *                        // rl _must_ not be NULL because we need it for the statistics
         *                        rl[row] = rowlen;
         *                        
         *                        if (rlp) {
         *                            rlp[row] = PAD(rowlen,P);
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
        for( chunk = 0; chunk < nChunks; chunk++ ) {
            (*chunkptr)[chunk] = 0; // NUMA init
            for (i=0, row = chunk*C; (i < C) && (row < SPM_NROWS(mat)); i++, row++) {
               
                if (mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY) {
                    if (mat->context->row_map->glb_perm && mat->context->row_map->loc_perm) {
                        INFO_LOG("Global _and_ local permutation");
                        funcerrs += src->func(mat->context->row_map->glb_perm_inv[mat->context->row_map->loc_perm_inv[row]],&rowlen,tmpcol,tmpval,src->arg);
                    } else if (mat->context->row_map->glb_perm) {
                        funcerrs += src->func(mat->context->row_map->glb_perm_inv[row],&rowlen,tmpcol,tmpval,src->arg);
                    } else if (mat->context->row_map->loc_perm) {
                        funcerrs += src->func(mat->context->row_map->goffs[me]+mat->context->row_map->loc_perm_inv[row],&rowlen,tmpcol,tmpval,src->arg);
                    }
                } else {
                    funcerrs += src->func(mat->context->row_map->goffs[me]+row,&rowlen,tmpcol,tmpval,src->arg);
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
   
    SPM_NNZ(mat) = (ghost_lidx)gnnz;
    mat->nEnts = (ghost_lidx)gnents;
    
    GHOST_INSTR_START("chunkptr_init");
    for(chunk = 0; chunk < nChunks; chunk++ ) {
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
    
    //MPI_CALL_GOTO(MPI_Allgather(&mat->nEnts,1,ghost_mpi_dt_lidx,mat->context->lnEnts,1,ghost_mpi_dt_lidx,mat->context->mpicomm),err,ret);
    //MPI_CALL_GOTO(MPI_Allgather(&fent,1,ghost_mpi_dt_gidx,mat->context->lfEnt,1,ghost_mpi_dt_gidx,mat->context->mpicomm),err,ret);
    MPI_CALL_GOTO(MPI_Allreduce(&gnnz,&(mat->context->gnnz),1,ghost_mpi_dt_gidx,MPI_SUM,mat->context->mpicomm),err,ret);
    #endif
    
    /* 
    if (src->maxrowlen != mat->maxRowLen) {
        DEBUG_LOG(1,"The maximum row length was not correct. Setting it from %"PRLIDX" to %"PRGIDX,src->maxrowlen,mat->maxRowLen); 
        src->maxrowlen = mat->maxRowLen;
    }
    */
    
    
    bool readcols = 0; // we only need to read the columns the first time the matrix is created
    if (!(*val)) {
        GHOST_CALL_GOTO(ghost_malloc_align((void **)val,mat->elSize*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
    }
    
    if (!(*col)) {
        GHOST_CALL_GOTO(ghost_malloc_align((void **)col,sizeof(ghost_gidx)*(size_t)mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
        readcols = 1;
    }
    
    
    if (src->func == ghost_sparsemat_rowfunc_crs && mat->context->row_map->glb_perm) {
        ERROR_LOG("Global permutation does not work with local CRS source");
    }
    
    GHOST_INSTR_START("cols_and_vals");
    #pragma omp parallel private(i,colidx,row,tmpval,tmpcol)
    {
        int funcret = 0;
        GHOST_CALL(ghost_malloc((void **)&tmpval,C*mat->maxRowLen*mat->elSize),ret);
        GHOST_CALL(ghost_malloc((void **)&tmpcol,C*mat->maxRowLen*sizeof(ghost_gidx)),ret);
        
        if (src->func == ghost_sparsemat_rowfunc_crs) {
            ghost_gidx *crscol;
            char *crsval = (char *)(((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->val);
            ghost_lidx *crsrpt = ((ghost_sparsemat_rowfunc_crs_arg *)src->arg)->rpt;
            #pragma omp single
            INFO_LOG("Fast matrix construction for CRS source and no permutation");
            
            #pragma omp for schedule(runtime)
            for( chunk = 0; chunk < nChunks; chunk++ ) {
                //memset(tmpval,0,mat->elSize*src->maxrowlen*C);
                
                for (i=0, row = chunk*C; (i<C) && (chunk*C+i < SPM_NROWS(mat)); i++, row++) {
                    ghost_gidx actualrow;
                    if (mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY) {
                        actualrow = mat->context->row_map->loc_perm_inv[row];
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
                            if (mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY) {
                                // local permutation: distinction between global and local entriess, if GHOST_PERM_NO_DISTINCTION is not set 
                                if ((mat->context->flags & GHOST_PERM_NO_DISTINCTION) || ( (crscol[colidx] >= mat->context->row_map->goffs[me]) && (crscol[colidx] < (mat->context->row_map->goffs[me]+SPM_NROWS(mat))) )) { // local entry: copy with permutation
                                    if (mat->traits.flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = crscol[colidx];
                                    } else if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->col_map->loc_perm[crscol[colidx]];
                                    } else {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->col_map->loc_perm[crscol[colidx]-mat->context->row_map->goffs[me]]+mat->context->row_map->goffs[me];
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
                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->row_map->goffs[me];
                    }
                    
                }
            }
        } else {
            #pragma omp for schedule(runtime)
            for (chunk = 0; chunk < nChunks; chunk++) {
                if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                    memset(&(*col)[(*chunkptr)[chunk]],0,C*clp[chunk]*sizeof(ghost_gidx));
                } else {
                    for (i=0; i<C*clp[chunk]; i++) {    
                        (*col)[(*chunkptr)[chunk]] = mat->context->row_map->offs;
                    }
                }
                memset(&(*val)[(*chunkptr)[chunk]*mat->elSize],0,C*clp[chunk]*mat->elSize);
            }
            #pragma omp for schedule(runtime)
            for (chunk = 0; chunk < nChunks; chunk++) {
                if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                    memset(tmpcol,0,C*mat->maxRowLen*sizeof(ghost_gidx));
                } else {
                    for (i=0; i<C*mat->maxRowLen; i++) {    
                        tmpcol[i] = mat->context->row_map->offs;
                    }
                }
                memset(tmpval,0,C*mat->maxRowLen*mat->elSize);

                for (i=0, row = chunk*C; (i<C) && (chunk*C+i < SPM_NROWS(mat)); i++, row++) {
                    if (mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY) {
                        if (mat->context->row_map->glb_perm && mat->context->row_map->loc_perm) {
                            funcret = src->func(mat->context->row_map->glb_perm_inv[mat->context->row_map->loc_perm_inv[row]],&rl[row],&tmpcol[mat->maxRowLen*i],&tmpval[mat->maxRowLen*i*mat->elSize],src->arg);
                        } else if (mat->context->row_map->glb_perm) {
                            funcret = src->func(mat->context->row_map->glb_perm_inv[row],&rl[row],&tmpcol[mat->maxRowLen*i],&tmpval[mat->maxRowLen*i*mat->elSize],src->arg);
                        } else if (mat->context->row_map->loc_perm) {
                            funcret = src->func(mat->context->row_map->goffs[me]+mat->context->row_map->loc_perm_inv[row],&rl[row],&tmpcol[mat->maxRowLen*i],&tmpval[mat->maxRowLen*i*mat->elSize],src->arg);
                        }
                        
                    } else {
                        funcret = src->func(mat->context->row_map->goffs[me]+row,&rl[row],&tmpcol[mat->maxRowLen*i],&tmpval[mat->maxRowLen*i*mat->elSize],src->arg);
                    }
                    if (funcret) {
                        ERROR_LOG("Matrix construction function returned error");
                        ret = GHOST_ERR_UNKNOWN;
                    }
                    for (colidx = 0; colidx<clp[chunk]; colidx++) {
                        memcpy(*val+mat->elSize*((*chunkptr)[chunk]+colidx*C+i),&tmpval[mat->elSize*(i*mat->maxRowLen+colidx)],mat->elSize);
                        if (mat->traits.flags & GHOST_SPARSEMAT_PERM_ANY) {
                            if (mat->context->row_map->glb_perm) {
                                // no distinction between global and local entries
                                // global permutation will be done after all rows are read
                                (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*mat->maxRowLen+colidx];
                            } else { 
                                // local permutation: distinction between global and local entries, if GHOST_PERM_NO_DISTINCTION is not set 
                                if ((mat->context->flags & GHOST_PERM_NO_DISTINCTION) || ((tmpcol[i*mat->maxRowLen+colidx] >= mat->context->row_map->goffs[me]) && (tmpcol[i*mat->maxRowLen+colidx] < (mat->context->row_map->goffs[me]+SPM_NROWS(mat))))) { 
                                    // local entry: copy with permutation
                                    if (mat->traits.flags & GHOST_SPARSEMAT_NOT_PERMUTE_COLS) {
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*mat->maxRowLen+colidx];
                                    } else if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                                        // (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->row_map->loc_perm->colPerm[tmpcol[i*mat->maxRowLen+colidx]]   
                                        // do not permute remote and do not allow local to go to remote
                                        if(tmpcol[i*mat->maxRowLen+colidx] < mat->context->col_map->dimpad) {
                                            if( mat->context->col_map->loc_perm[tmpcol[i*mat->maxRowLen+colidx]]>=mat->context->col_map->dimpad ) {       
                                                ERROR_LOG("Ensure you have halo number of paddings, since GHOST_PERM_NO_DISTINCTION is switched on");       
                                            }
                                            (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->col_map->loc_perm[tmpcol[i*mat->maxRowLen+colidx]];
                                        } else {
                                            (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*mat->maxRowLen+colidx];
                                        }
                                    } else {
                                        
                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->col_map->loc_perm[tmpcol[i*mat->maxRowLen+colidx]-mat->context->row_map->goffs[me]]+mat->context->row_map->goffs[me];
                                        //                                        (*col)[(*chunkptr)[chunk]+colidx*C+i] = mat->context->row_map->loc_perm->colPerm[tmpcol[i*mat->maxRowLen+colidx]-mat->context->row_map->goffs[me]]+mat->context->row_map->goffs[me];
                                    }
                                } else { 
                                    // remote entry: copy without permutation
                                    (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*mat->maxRowLen+colidx];
                                }
                            }
                        } else {
                            (*col)[(*chunkptr)[chunk]+colidx*C+i] = tmpcol[i*mat->maxRowLen+colidx];
                        }
                    }
                }
            }
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }
    
    if (SPM_NROWS(mat) % C) {
        for (i=SPM_NROWS(mat)%C; i < C; i++) {
            for (colidx = 0; colidx<clp[nChunks-1]; colidx++) {
                (*col)[(*chunkptr)[nChunks-1]+colidx*C+i] = mat->context->row_map->goffs[me];
                memset(*val+mat->elSize*((*chunkptr)[nChunks-1]+colidx*C+i),0,mat->elSize);
            }
        }
    }
    
    GHOST_INSTR_STOP("cols_and_vals");
    
    if (mat->context->row_map->glb_perm) {
        ghost_sparsemat_perm_global_cols(*col,mat->nEnts,mat->context);
    }
    
    GHOST_INSTR_START("sort_and_register");
    
    if (!(mat->traits.flags & GHOST_SPARSEMAT_NOT_SORT_COLS)) {
        for( chunk = 0; chunk < nChunks; chunk++ ) {
            for (i=0; (i<C) && (chunk*C+i < SPM_NROWS(mat)); i++) {
                row = chunk*C+i;
                ghost_sparsemat_sortrow(&((*col)[(*chunkptr)[chunk]+i]),&(*val)[((*chunkptr)[chunk]+i)*mat->elSize],mat->elSize,rl[row],C);
                #ifdef GHOST_SPARSEMAT_STATS
                ghost_sparsemat_registerrow(mat,mat->context->row_map->goffs[me]+row,&(*col)[(*chunkptr)[chunk]+i],rl[row],C);
                #endif
            }
        }
    } else {
        #ifdef GHOST_SPARSEMAT_STATS
        for( chunk = 0; chunk < nChunks; chunk++ ) {
            for (i=0; (i<C) && (chunk*C+i < SPM_NROWS(mat)); i++) {
                row = chunk*C+i;
                ghost_sparsemat_registerrow(mat,mat->context->row_map->goffs[me]+row,&(*col)[(*chunkptr)[chunk]+i],rl[row],C);
            }
        }
        #endif
    }
    
    #ifdef GHOST_SPARSEMAT_STATS
    ghost_sparsemat_registerrow_finalize(mat);
    #endif
    GHOST_INSTR_STOP("sort_and_register");
/*    
    mat->context->lnEnts[me] = mat->nEnts;
    
    for (i=0; i<nprocs; i++) {
        mat->context->lfEnt[i] = 0;
    } 
    
    for (i=1; i<nprocs; i++) {
        mat->context->lfEnt[i] = mat->context->lfEnt[i-1]+mat->context->lnEnts[i-1];
    } */
    
    free(tmpclp);
    free(tmprl);
    
    
    if (ret != GHOST_SUCCESS) {
        goto err;
    }
    GHOST_CALL_GOTO(ghost_sparsemat_split(mat),err,ret);
    
    if(mat->traits.flags & GHOST_SOLVER_KACZ) {
        //split transition zones 
        if(mat->traits.flags & (ghost_sparsemat_flags)GHOST_SPARSEMAT_BLOCKCOLOR) {
            split_transition(mat);
        } 
        //split if no splitting was done before and MC is off
        else if(!(mat->traits.flags & GHOST_SPARSEMAT_COLOR)) {
            if( (mat->context->kaczRatio >= 2*mat->context->kacz_setting.active_threads) ) {
                ghost_rcm_dissect(mat);
            } else {
                split_analytical(mat);
            }
        }
    }
    
    #ifdef GHOST_HAVE_CUDA
    if (!(mat->traits.flags & GHOST_SPARSEMAT_HOST))
        ghost_sparsemat_upload(mat);
    #endif


    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->rowLen2,SPM_NROWSPAD(mat)/2*sizeof(ghost_lidx)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->rowLen4,SPM_NROWSPAD(mat)/4*sizeof(ghost_lidx)),err,ret);
    ghost_lidx max4 = 0 , max2 = 0;
    for (i=0; i<SPM_NROWSPAD(mat); i++) {
        if (!(i%2)) {
            max2 = 0;
        }
        if (!(i%4)) {
            max4 = 0;
        }
        if (mat->rowLen[i] > max2) {
            max2 = mat->rowLen[i];
        }
        if (mat->rowLen[i] > max4) {
            max4 = mat->rowLen[i];
        }
        if (!((i+1)%2)) {
            mat->rowLen2[i/2] = max2;
        }
        if (!((i+1)%4)) {
            mat->rowLen4[i/4] = max4;
        }
    }
        
        

    if (src->func == ghost_sparsemat_rowfunc_bincrs || src->func == ghost_sparsemat_rowfunc_mm) {
        if (src->func(GHOST_SPARSEMAT_ROWFUNC_FINALIZE,NULL,NULL,NULL,src->arg)) {
            ERROR_LOG("Error in matrix creation function");
            ret = GHOST_ERR_UNKNOWN;
            goto err;
        }
    }
    
    goto out;
    err:
    free(mat->val); mat->val = NULL;
    free(mat->col_orig); mat->col_orig = NULL;
    free(mat->chunkMin); mat->chunkMin = NULL;
    free(mat->chunkLen); mat->chunkLen = NULL;
    free(mat->chunkLenPadded); mat->chunkLenPadded = NULL;
    free(mat->rowLen); mat->rowLen = NULL;
    free(mat->rowLen2); mat->rowLen2 = NULL;
    free(mat->rowLen4); mat->rowLen4 = NULL;
    free(mat->rowLenPadded); mat->rowLenPadded = NULL;
    free(mat->chunkStart); mat->chunkStart = NULL;
    mat->nEnts = 0;
    
    out:
    

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_INITIALIZATION);

    return ret;
}

static ghost_error ghost_sparsemat_split(ghost_sparsemat *mat)
{
    
    if (!mat) {
        ERROR_LOG("Matrix is NULL");
        return GHOST_ERR_INVALID_ARG;
    }
    ghost_error ret = GHOST_SUCCESS;
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_INITIALIZATION);
    
    
    DEBUG_LOG(1,"Splitting the SELL matrix into a local and remote part");
    ghost_gidx i,j;
    int me,nproc;
    GHOST_CALL_RETURN(ghost_rank(&me, mat->context->mpicomm));
    GHOST_CALL_RETURN(ghost_nrank(&nproc, mat->context->mpicomm));
    
    ghost_lidx lnEnts_l, lnEnts_r;
    ghost_lidx current_l, current_r;
    
    
    ghost_lidx chunk;
    ghost_lidx idx, row;
    
    GHOST_INSTR_START("init_compressed_cols");
    #ifdef GHOST_IDX_UNIFORM
    if (!(mat->traits.flags & GHOST_SPARSEMAT_SAVE_ORIG_COLS)) {
        DEBUG_LOG(1,"In-place column compression!");
        mat->col = mat->col_orig;
    } else 
        #endif
    {
        if (!mat->col) {
            DEBUG_LOG(1,"Duplicate col array!");
            GHOST_CALL_GOTO(ghost_malloc_align((void **)&mat->col,sizeof(ghost_lidx)*mat->nEnts,GHOST_DATA_ALIGNMENT),err,ret);
            #pragma omp parallel for private(j) schedule(runtime)
            for (i=0; i<SPM_NCHUNKS(mat); i++) {
                for (j=mat->chunkStart[i]; j<mat->chunkStart[i+1]; j++) {
                    mat->col[j] = 0;
                }
            }
        }
    }
    GHOST_INSTR_STOP("init_compressed_cols");
  
    ghost_lidx nhalo; 
    GHOST_CALL_GOTO(ghost_context_comm_init(mat->context,mat->col_orig,mat,mat->col,&nhalo),err,ret);
    
    if (nproc > 1) { 
        if (mat->context->col_map->nhalo) {
            if (nhalo > mat->context->col_map->nhalo) {
                ERROR_LOG("The maps are not compatible!");
                ret = GHOST_ERR_INVALID_ARG;
                goto err;
            }
        } else {
            mat->context->col_map->nhalo = nhalo;
            if(mat->context->flags & GHOST_PERM_NO_DISTINCTION) {
                mat->context->col_map->dimhalo = mat->context->col_map->dimpad+2*mat->context->col_map->nhalo;
                mat->context->col_map->dimpad = PAD(mat->context->col_map->dimpad+2*mat->context->col_map->nhalo,ghost_densemat_row_padding());
                initHaloAvg(mat);
            } else {
                mat->context->col_map->dimhalo = mat->context->col_map->dimpad+mat->context->col_map->nhalo;
                mat->context->col_map->dimpad = PAD(mat->context->col_map->dimpad+mat->context->col_map->nhalo,ghost_densemat_row_padding());
            }
        }
    }

    
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
        ghost_sparsemat *localMat = mat->localPart;
        mat->localPart->traits.symmetry = mat->traits.symmetry;
        
        ghost_sparsemat_create(&(mat->remotePart),mat->context,&mat->splittraits[1],1);
        ghost_sparsemat *remoteMat = mat->remotePart; 
        
        mat->localPart->traits.T = mat->traits.T;
        mat->remotePart->traits.T = mat->traits.T;

        mat->localPart->elSize = mat->elSize;
        mat->remotePart->elSize = mat->elSize;
    
        mat->localPart->nchunks = CEILDIV(SPM_NROWS(mat->localPart),mat->localPart->traits.C);
        mat->remotePart->nchunks = CEILDIV(SPM_NROWS(mat->remotePart),mat->remotePart->traits.C);
        
        ghost_lidx nChunks = SPM_NCHUNKS(mat);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localMat->chunkStart, (nChunks+1)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localMat->chunkMin, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localMat->chunkLen, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localMat->chunkLenPadded, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localMat->rowLen, (SPM_NROWSPAD(mat))*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&localMat->rowLenPadded, (SPM_NROWSPAD(mat))*sizeof(ghost_lidx)),err,ret);
        
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteMat->chunkStart, (nChunks+1)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteMat->chunkMin, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteMat->chunkLen, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteMat->chunkLenPadded, (nChunks)*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteMat->rowLen, (SPM_NROWSPAD(mat))*sizeof(ghost_lidx)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&remoteMat->rowLenPadded, (SPM_NROWSPAD(mat))*sizeof(ghost_lidx)),err,ret);
        
        #pragma omp parallel for schedule(runtime)
        for (i=0; i<SPM_NROWSPAD(mat); i++) {
            localMat->rowLen[i] = 0;
            remoteMat->rowLen[i] = 0;
            localMat->rowLenPadded[i] = 0;
            remoteMat->rowLenPadded[i] = 0;
        }
        
        #pragma omp parallel for schedule(runtime)
        for(chunk = 0; chunk < SPM_NCHUNKS(mat); chunk++) {
            localMat->chunkLen[chunk] = 0;
            remoteMat->chunkLen[chunk] = 0;
            localMat->chunkLenPadded[chunk] = 0;
            remoteMat->chunkLenPadded[chunk] = 0;
            localMat->chunkMin[chunk] = 0;
            remoteMat->chunkMin[chunk] = 0;
        }
        localMat->chunkStart[0] = 0;
        remoteMat->chunkStart[0] = 0;
        
        lnEnts_l = 0;
        lnEnts_r = 0;
        
        for(chunk = 0; chunk < SPM_NCHUNKS(mat); chunk++) {
            
            for (i=0; i<mat->chunkLen[chunk]; i++) {
                for (j=0; j<mat->traits.C; j++) {
                    row = chunk*mat->traits.C+j;
                    idx = mat->chunkStart[chunk]+i*mat->traits.C+j;
                    
                    if (i < mat->rowLen[row]) {
                        if (mat->col[idx] < mat->context->row_map->ldim[me]) {
                            localMat->rowLen[row]++;
                        } else {
                            remoteMat->rowLen[row]++;
                        }
                        localMat->rowLenPadded[row] = PAD(localMat->rowLen[row],mat->localPart->traits.T);
                        remoteMat->rowLenPadded[row] = PAD(remoteMat->rowLen[row],mat->remotePart->traits.T);
                    }
                }
            }
            
            for (j=0; j<mat->traits.C; j++) {
                row = chunk*mat->traits.C+j;
                localMat->chunkLen[chunk] = MAX(localMat->chunkLen[chunk],localMat->rowLen[row]);
                remoteMat->chunkLen[chunk] = MAX(remoteMat->chunkLen[chunk],remoteMat->rowLen[row]);
            }
            lnEnts_l += localMat->chunkLen[chunk]*mat->traits.C;
            lnEnts_r += remoteMat->chunkLen[chunk]*mat->traits.C;
            localMat->chunkStart[chunk+1] = lnEnts_l;
            remoteMat->chunkStart[chunk+1] = lnEnts_r;
            
            localMat->chunkLenPadded[chunk] = PAD(localMat->chunkLen[chunk],mat->localPart->traits.T);
            remoteMat->chunkLenPadded[chunk] = PAD(remoteMat->chunkLen[chunk],mat->remotePart->traits.T);
            
        }
        
        
        
        /*
         *           for (i=0; i<mat->nEnts;i++) {
         *           if (mat->col[i]<mat->context->row_map->ldim[me]) lnEnts_l++;
    }
    lnEnts_r = mat->context->lnEnts[me]-lnEnts_l;*/
        
        
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&localMat->val,lnEnts_l*mat->elSize,GHOST_DATA_ALIGNMENT),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&localMat->col,lnEnts_l*sizeof(ghost_lidx),GHOST_DATA_ALIGNMENT),err,ret); 
        
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&remoteMat->val,lnEnts_r*mat->elSize,GHOST_DATA_ALIGNMENT),err,ret); 
        GHOST_CALL_GOTO(ghost_malloc_align((void **)&remoteMat->col,lnEnts_r*sizeof(ghost_lidx),GHOST_DATA_ALIGNMENT),err,ret); 
       
        mat->localPart->nEnts = lnEnts_l;
        mat->localPart->traits.C = mat->traits.C;
        
        mat->remotePart->nEnts = lnEnts_r;
        mat->remotePart->traits.C = mat->traits.C;
        
        #pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < SPM_NCHUNKS(mat->localPart); chunk++) {
            for (i=0; i<localMat->chunkLenPadded[chunk]; i++) {
                for (j=0; j<mat->localPart->traits.C; j++) {
                    idx = localMat->chunkStart[chunk]+i*mat->localPart->traits.C+j;
                    memset(&((char *)(localMat->val))[idx*mat->elSize],0,mat->elSize);
                    localMat->col[idx] = 0;
                }
            }
        }
        
        #pragma omp parallel for schedule(runtime) private (i,j,idx)
        for(chunk = 0; chunk < SPM_NCHUNKS(mat->remotePart); chunk++) {
            for (i=0; i<remoteMat->chunkLenPadded[chunk]; i++) {
                for (j=0; j<mat->remotePart->traits.C; j++) {
                    idx = remoteMat->chunkStart[chunk]+i*mat->remotePart->traits.C+j;
                    memset(&((char *)(remoteMat->val))[idx*mat->elSize],0,mat->elSize);
                    remoteMat->col[idx] = 0;
                }
            }
        }
        
        current_l = 0;
        current_r = 0;
        ghost_lidx *col_l, *col_r;
        ghost_malloc((void **)&col_l,sizeof(ghost_lidx)*mat->traits.C);
        ghost_malloc((void **)&col_r,sizeof(ghost_lidx)*mat->traits.C);
        
        for(chunk = 0; chunk < SPM_NCHUNKS(mat); chunk++) {
            
            for (j=0; j<mat->traits.C; j++) {
                col_l[j] = 0;
                col_r[j] = 0;
            }
            
            for (i=0; i<mat->chunkLen[chunk]; i++) {
                for (j=0; j<mat->traits.C; j++) {
                    row = chunk*mat->traits.C+j;
                    idx = mat->chunkStart[chunk]+i*mat->traits.C+j;
                    
                    if (i<mat->rowLen[row]) {
                        if (mat->col[idx] < mat->context->row_map->ldim[me]) {
                            if (col_l[j] < localMat->rowLen[row]) {
                                ghost_lidx lidx = localMat->chunkStart[chunk]+col_l[j]*mat->localPart->traits.C+j;
                                localMat->col[lidx] = mat->col[idx];
                                memcpy(&localMat->val[lidx*mat->elSize],&mat->val[idx*mat->elSize],mat->elSize);
                                current_l++;
                            }
                            col_l[j]++;
                        }
                        else{
                            if (col_r[j] < remoteMat->rowLen[row]) {
                                ghost_lidx ridx = remoteMat->chunkStart[chunk]+col_r[j]*mat->remotePart->traits.C+j;
                                remoteMat->col[ridx] = mat->col[idx];
                                memcpy(&remoteMat->val[ridx*mat->elSize],&mat->val[idx*mat->elSize],mat->elSize);
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
            ghost_sparsemat_upload(mat->localPart);
            ghost_sparsemat_upload(mat->remotePart);
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

ghost_error ghost_sparsemat_to_bin(ghost_sparsemat *mat, char *matrixPath)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_IO);
    UNUSED(mat);
    UNUSED(matrixPath);
    
    ERROR_LOG("SELL matrix to binary CRS file not implemented");
    
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_IO);
    return GHOST_ERR_NOT_IMPLEMENTED;
}

#ifdef GHOST_HAVE_CUDA
static ghost_error ghost_sparsemat_upload(ghost_sparsemat* mat) 
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    if (!(mat->traits.flags & GHOST_SPARSEMAT_HOST)) {
        DEBUG_LOG(1,"Creating matrix on CUDA device");
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&mat->cu_rowLen,(SPM_NROWS(mat))*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&mat->cu_rowLenPadded,(SPM_NROWS(mat))*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&mat->cu_col,(mat->nEnts)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&mat->cu_val,(mat->nEnts)*mat->elSize));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&mat->cu_chunkStart,(SPM_NROWSPAD(mat)/mat->traits.C+1)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_malloc((void **)&mat->cu_chunkLen,(SPM_NROWSPAD(mat)/mat->traits.C)*sizeof(ghost_lidx)));
        
        GHOST_CALL_RETURN(ghost_cu_upload(mat->cu_rowLen, mat->rowLen, SPM_NROWS(mat)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_upload(mat->cu_rowLenPadded, mat->rowLenPadded, SPM_NROWS(mat)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_upload(mat->cu_col, mat->col, mat->nEnts*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_upload(mat->cu_val, mat->val, mat->nEnts*mat->elSize));
        GHOST_CALL_RETURN(ghost_cu_upload(mat->cu_chunkStart, mat->chunkStart, (SPM_NROWSPAD(mat)/mat->traits.C+1)*sizeof(ghost_lidx)));
        GHOST_CALL_RETURN(ghost_cu_upload(mat->cu_chunkLen, mat->chunkLen, (SPM_NROWSPAD(mat)/mat->traits.C)*sizeof(ghost_lidx)));
    }
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return GHOST_SUCCESS;
}
#endif

