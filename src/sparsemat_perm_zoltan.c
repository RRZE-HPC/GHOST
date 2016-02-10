#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"

#ifdef GHOST_HAVE_ZOLTAN
#include <zoltan.h>
#endif

typedef struct
{
    ghost_gidx *col;
    ghost_gidx *rpt;
    ghost_gidx nnz;
    ghost_gidx nrows;
    ghost_gidx gncols;
    ghost_gidx entoffs;
    ghost_gidx rowoffs;
    ghost_gidx uniquecols;
    ghost_gidx *colofvert;

} zoltan_info;


static int get_number_of_vertices(void *data, int *ierr)
{
    zoltan_info *info = (zoltan_info *)data;
    *ierr = ZOLTAN_OK;
    return info->uniquecols;
}

static void get_vertex_list(void *data, int sizeGID, int sizeLID,
            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                  int wgt_dim, float *obj_wgts, int *ierr)
{
    zoltan_info *info = (zoltan_info *)data;
    *ierr = ZOLTAN_OK;
    ghost_lidx i;

    for (i=0; i<info->uniquecols; i++){
        globalID[i] = info->colofvert[i];
    }
}

static void get_hypergraph_size(void *data, int *num_lists, int *num_nonzeroes,
                                int *format, int *ierr)
{
    zoltan_info *info = (zoltan_info *)data;
    *ierr = ZOLTAN_OK;

    *num_lists = info->nrows;
    *num_nonzeroes = info->nnz;
    *format = ZOLTAN_COMPRESSED_EDGE;

    return;
}

static void get_hypergraph(void *data, int sizeGID, int num_edges, int num_nonzeroes,
                           int format, ZOLTAN_ID_PTR edgeGID, int *vtxPtr,
                           ZOLTAN_ID_PTR vtxGID, int *ierr)
{
    zoltan_info *info = (zoltan_info *)data;
    ghost_lidx i;

    *ierr = ZOLTAN_OK;

    for (i=0; i < num_edges; i++){
        edgeGID[i] = info->rowoffs + i;
        vtxPtr[i] = info->rpt[i];
        WARNING_LOG("edgeGID[%d] = %d",i,edgeGID[i]);
        WARNING_LOG("vtxPtr[%d] = %d",i,vtxPtr[i]);
    }

    for (i=0; i < num_nonzeroes; i++){
        vtxGID[i] = info->col[i];
        WARNING_LOG("vtxGID[%d] = %d",i,vtxGID[i]);
    }

    return;
}

static int cmp_gidx(const void *p1, const void *p2)
{
    return *(ghost_gidx *)p1 - *(ghost_gidx *)p2;
}

ghost_error ghost_sparsemat_perm_zoltan(ghost_sparsemat *mat, void *matrixSource, ghost_sparsemat_src srcType)
{
#if !defined(GHOST_HAVE_ZOLTAN) || !defined(GHOST_HAVE_MPI)
    UNUSED(mat);
    UNUSED(matrixSource);
    UNUSED(srcType);
    WARNING_LOG("Zoltan or MPI not available. Will not create matrix permutation!");
    return GHOST_SUCCESS;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    ghost_error ret = GHOST_SUCCESS;
    ghost_lidx i;
    zoltan_info info;
    int me, nprocs;
    struct Zoltan_Struct *zz;
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    
    if (mat->context->perm_global) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }
    if (srcType != GHOST_SPARSEMAT_SRC_FUNC) {
        ERROR_LOG("Only function sparse matrix source allowed!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }
    
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_global,sizeof(ghost_permutation)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_global->perm,sizeof(ghost_gidx)*mat->context->lnrows[me]),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->perm_global->invPerm,sizeof(ghost_gidx)*mat->context->lnrows[me]),err,ret);
    memset(mat->context->perm_global->perm,0,sizeof(ghost_gidx)*mat->context->lnrows[me]);
    memset(mat->context->perm_global->invPerm,0,sizeof(ghost_gidx)*mat->context->lnrows[me]);
    mat->context->perm_global->scope = GHOST_PERMUTATION_GLOBAL;
    mat->context->perm_global->len = mat->context->lnrows[me];
           
    ghost_malloc((void **)&(info.rpt),(mat->context->lnrows[me]+1)*sizeof(ghost_gidx));

    ghost_sparsemat_src_rowfunc *src = (ghost_sparsemat_src_rowfunc *)matrixSource;
    char * tmpval = NULL;
    ghost_gidx * tmpcol = NULL;

    ghost_gidx nnz = 0;
    ghost_lidx rowlen;
    info.rpt[0] = 0;
#pragma omp parallel private (tmpval,tmpcol,i,rowlen) reduction(+:nnz)
    {
        ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
        ghost_malloc((void **)&tmpcol,src->maxrowlen*sizeof(ghost_gidx));

#pragma omp for
        for (i=0; i<mat->context->lnrows[me]; i++) {
            src->func(mat->context->lfRow[me]+i,&rowlen,tmpcol,tmpval,NULL);
            nnz += rowlen;
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }

    info.nnz = nnz;
    ghost_malloc((void **)&(info.col),(info.nnz)*sizeof(ghost_gidx));
    ghost_malloc((void **)&(info.colofvert),(info.nnz)*sizeof(ghost_gidx));

#pragma omp parallel private (tmpval,tmpcol,i,rowlen)
    {
        ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
#pragma omp for ordered
        for (i=0; i<mat->context->lnrows[me]; i++) {
#pragma omp ordered
            {
                src->func(mat->context->lfRow[me]+i,&rowlen,&(info.col[info.rpt[i]]),tmpval,NULL);
                info.rpt[i+1] = info.rpt[i] + rowlen;
            }
        }
        free(tmpval); tmpval = NULL;
    }
    info.nrows = mat->context->lnrows[me];

    ghost_gidx *sortcol;
    ghost_malloc((void **)&sortcol,info.nnz*sizeof(ghost_gidx));
    memcpy(sortcol,info.col,info.nnz*sizeof(ghost_gidx));
    qsort(sortcol,info.nnz,sizeof(ghost_gidx),cmp_gidx);

    info.uniquecols = 1;
    info.colofvert[0] = sortcol[0];
    for (i=1; i<info.nnz; i++) {
        if (sortcol[i] != sortcol[i-1]) {
            info.colofvert[info.uniquecols] = sortcol[i];
            info.uniquecols++;
        }
    }
    ERROR_LOG("unique cols: %d",info.uniquecols);
    for (i=0; i<info.uniquecols; i++) {
        INFO_LOG("colofvert[%d] = %d",i,info.colofvert[i]);
    }

    ghost_gidx neigh_nnz; // nnz of previous rank
    ghost_gidx accu_nnz; // so-far accumulated nnz

    info.rowoffs = mat->context->lfRow[me];
    info.gncols = mat->context->gncols;
    info.entoffs = 0;
    accu_nnz = info.nnz;
   
    if (nprocs > 1) { 
        if (me == 0) {
            MPI_CALL_GOTO(MPI_Send(&info.nnz,1,ghost_mpi_dt_gidx,1,me,mat->context->mpicomm),err,ret);
            MPI_CALL_GOTO(MPI_Recv(&neigh_nnz,1,ghost_mpi_dt_gidx,nprocs-1,nprocs-1,mat->context->mpicomm,MPI_STATUS_IGNORE),err,ret);
        }

        for (i=0; i<nprocs; i++) {
            if (i == me && i != 0) {
                MPI_CALL_GOTO(MPI_Recv(&neigh_nnz,1,ghost_mpi_dt_gidx,i-1,i-1,mat->context->mpicomm,MPI_STATUS_IGNORE),err,ret);
                info.entoffs = neigh_nnz;
                accu_nnz = neigh_nnz + info.nnz;
                MPI_CALL_GOTO(MPI_Send(&accu_nnz,1,ghost_mpi_dt_gidx,(i+1)%nprocs,i,mat->context->mpicomm),err,ret);
            }
        }
    }

    for (i=0; i<mat->context->lnrows[me]; i++) {
        mat->context->perm_global->perm[i] = mat->context->lfRow[me]+i;
        mat->context->perm_global->invPerm[i] = mat->context->lfRow[me]+i;
    }
    
    zz = Zoltan_Create(mat->context->mpicomm);

    /* General parameters */

    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0"),err,ret);
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "LB_METHOD", "HYPERGRAPH"),err,ret);   /* partitioning method */
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "HYPERGRAPH_PACKAGE", "PHG"),err,ret); /* version of method */
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1"),err,ret);/* global IDs are integers */
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1"),err,ret);/* local IDs are integers */
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL"),err,ret); /* export AND import lists */
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "0"),err,ret); /* use Zoltan default vertex weights */
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "EDGE_WEIGHT_DIM", "0"),err,ret);/* use Zoltan default hyperedge weights */

    /* PHG parameters  - see the Zoltan User's Guide for many more
    *   (The "REPARTITION" approach asks Zoltan to create a partitioning that is
    *    better but is not too far from the current partitioning, rather than partitioning 
    *    from scratch.  It may be faster but of lower quality that LB_APPROACH=PARTITION.)
    */

    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "LB_APPROACH", "PARTITION"),err,ret);
      
    ZOLTAN_CALL_GOTO(Zoltan_Set_Num_Obj_Fn(zz, get_number_of_vertices, &info),err,ret);
    ZOLTAN_CALL_GOTO(Zoltan_Set_Obj_List_Fn(zz, get_vertex_list, &info),err,ret);
    ZOLTAN_CALL_GOTO(Zoltan_Set_HG_Size_CS_Fn(zz, get_hypergraph_size, &info),err,ret);
    ZOLTAN_CALL_GOTO(Zoltan_Set_HG_CS_Fn(zz, get_hypergraph, &info),err,ret);

    ZOLTAN_CALL_GOTO(Zoltan_LB_Partition(zz, /* input (all remaining fields are output) */
        &changes,        /* 1 if partitioning was changed, 0 otherwise */ 
        &numGidEntries,  /* Number of integers used for a global ID */
        &numLidEntries,  /* Number of integers used for a local ID */
        &numImport,      /* Number of vertices to be sent to me */
        &importGlobalGids,  /* Global IDs of vertices to be sent to me */
        &importLocalGids,   /* Local IDs of vertices to be sent to me */
        &importProcs,    /* Process rank for source of each incoming vertex */
        &importToPart,   /* New partition for each incoming vertex */
        &numExport,      /* Number of vertices I must send to other processes*/
        &exportGlobalGids,  /* Global IDs of the vertices I must send */
        &exportLocalGids,   /* Local IDs of the vertices I must send */
        &exportProcs,    /* Process to which I send each of the vertices */
        &exportToPart),err,ret);  /* Partition to which each vertex will belong */

    INFO_LOG("numImport: %d",numImport);
    for (i=0; i<numImport; i++) {
        INFO_LOG("importGlobalGids[%d] = %d",i,importGlobalGids[i]);
        INFO_LOG("importProcs[%d] = %d",i,importProcs[i]);
    }
    INFO_LOG("numExport: %d",numExport);
    for (i=0; i<numExport; i++) {
        INFO_LOG("exportGlobalGids[%d] = %d",i,exportGlobalGids[i]);
    }

    goto out;
err:
    free(mat->context->perm_global->perm); mat->context->perm_global->perm = NULL;
    free(mat->context->perm_global->invPerm); mat->context->perm_global->invPerm = NULL;
#ifdef GHOST_HAVE_CUDA
    ghost_cu_free(mat->context->perm_global->cu_perm); mat->context->perm_global->cu_perm = NULL;
#endif
    free(mat->context->perm_global); mat->context->perm_global = NULL;
    
out:
    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
    Zoltan_Destroy(&zz);
    free(info.rpt);
    free(info.col);
    free(info.colofvert);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;

#endif
}

