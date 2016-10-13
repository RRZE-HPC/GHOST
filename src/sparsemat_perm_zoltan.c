#include "ghost/sparsemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include "ghost/bincrs.h"

#ifdef GHOST_HAVE_ZOLTAN
#include <zoltan.h>


typedef struct
{
    ghost_gidx *col;
    ghost_lidx *rpt;
    ghost_lidx nnz;
    ghost_lidx nrows;
    ghost_gidx rowoffs;

} zoltan_info;

static int get_number_of_vertices(void *data, int *ierr)
{
    zoltan_info *info = (zoltan_info *)data;
    *ierr = ZOLTAN_OK;
    return info->nrows;
}

typedef struct {
    int part;
    ghost_gidx row;
} part_info;

static int part_info_cmp(const void *a, const void *b)
{
    return ((part_info *)a)->part - ((part_info *)b)->part;
}

static void get_vertex_list(void *data, int sizeGID, int sizeLID,
            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                  int wgt_dim, float *obj_wgts, int *ierr)
{
    UNUSED(sizeGID);
    UNUSED(sizeLID);
    UNUSED(wgt_dim);
    UNUSED(obj_wgts);
    UNUSED(localID);

    zoltan_info *info = (zoltan_info *)data;
    *ierr = ZOLTAN_OK;
    ghost_lidx i;

    for (i=0; i<info->nrows; i++){
        globalID[i] = info->rowoffs+i;
        localID[i] = i;
    }
}

static void get_hypergraph_size(void *data, int *num_lists, int *num_nonzeroes,
                                int *format, int *ierr)
{
    zoltan_info *info = (zoltan_info *)data;
    *ierr = ZOLTAN_OK;

    *num_lists = info->nrows;
    *num_nonzeroes = info->nnz;
    *format = ZOLTAN_COMPRESSED_VERTEX;

    return;
}

static void get_hypergraph(void *data, int sizeGID, int num_vert, int num_nonzeroes,
                           int format, ZOLTAN_ID_PTR vtxGID, int *vtxPtr,
                           ZOLTAN_ID_PTR edgeGID, int *ierr)
{
    UNUSED(sizeGID);
    UNUSED(format);
    
    zoltan_info *info = (zoltan_info *)data;
    ghost_lidx i;

    *ierr = ZOLTAN_OK;

    for (i=0; i < num_vert; i++){
        vtxGID[i] = info->rowoffs + i;
        vtxPtr[i] = info->rpt[i];
    }

    for (i=0; i < num_nonzeroes; i++){
        edgeGID[i] = info->col[i];
    }

    return;
}
#endif

ghost_error ghost_sparsemat_perm_zoltan(ghost_context *ctx, ghost_sparsemat *mat)
{
#if !defined(GHOST_HAVE_ZOLTAN) || !defined(GHOST_HAVE_MPI)
    UNUSED(ctx);
    UNUSED(mat);
    WARNING_LOG("Zoltan or MPI not available. Will not create matrix permutation!");
    return GHOST_SUCCESS;
#else
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_SETUP);
    ghost_error ret = GHOST_SUCCESS;
    ghost_lidx i;
    zoltan_info info;
    int nprocs,me;
    struct Zoltan_Struct *zz;
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    
    if (ctx->row_map->glb_perm) {
        WARNING_LOG("Existing global permutations will be overwritten!");
    }
    
    GHOST_CALL_GOTO(ghost_rank(&me, ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->row_map->glb_perm,sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&ctx->row_map->glb_perm_inv,sizeof(ghost_gidx)*ctx->row_map->dim),err,ret);
    ctx->col_map->glb_perm = ctx->row_map->glb_perm;
    ctx->col_map->glb_perm_inv = ctx->row_map->glb_perm_inv;

    memset(ctx->row_map->glb_perm,0,sizeof(ghost_gidx)*ctx->row_map->dim);
    memset(ctx->row_map->glb_perm_inv,0,sizeof(ghost_gidx)*ctx->row_map->dim);
           
    info.nnz = SPM_NNZ(mat);
    info.col = mat->col_orig;
    info.rpt = mat->chunkStart;
    info.nrows = ctx->row_map->dim;
    info.rowoffs = ctx->row_map->offs;
   
    zz = Zoltan_Create(ctx->mpicomm);

    INFO_LOG("before zoltan");
    /* General parameters */

    IF_DEBUG(1) {
        ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "DEBUG_LEVEL", "1"),err,ret);
    } else {
        ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0"),err,ret);
    }
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "LB_METHOD", "HYPERGRAPH"),err,ret);
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "HYPERGRAPH_PACKAGE", "PHG"),err,ret);
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "RETURN_LISTS", "PARTS"),err,ret);
    //ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL"),err,ret);
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "LB_APPROACH", "PARTITION"),err,ret);
    ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "CHECK_HYPERGRAPH", "1"),err,ret);
   // ZOLTAN_CALL_GOTO(Zoltan_Set_Param(zz, "IMBALANCE_TOL", "1.0"),err,ret);
      
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

    INFO_LOG("after zoltan");
    for (i=0; i<ctx->row_map->ldim[me]; i++) {
//        ctx->row_map->glb_perm[i] = ctx->row_map->goffs[me]+i;
    }

    part_info *partinfo;
    ghost_malloc((void **)&partinfo,sizeof(part_info)*SPM_NROWS(mat));
    for (i=0; i<numExport; i++) {
        partinfo[i].part = exportToPart[i];
        partinfo[i].row = ctx->row_map->goffs[me]+i;
    }
    part_info *global_partinfo;
    ghost_malloc((void **)&global_partinfo,sizeof(part_info)*ctx->row_map->gdim);
    
    const int nitems=2;
    int          blocklengths[2] = {1,1};
    MPI_Datatype types[2] = {MPI_INT, ghost_mpi_dt_gidx};
    MPI_Datatype mpi_partinfo_type;
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(part_info, part);
    offsets[1] = offsetof(part_info, row);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_partinfo_type);
    MPI_Type_commit(&mpi_partinfo_type);

    PERFWARNING_LOG("The sorting of export lists is currently serial! This can cause problems in terms of performance and memory!");
    INFO_LOG("before gather");
    if (me != 0) {
        MPI_Send(partinfo,ctx->row_map->ldim[me],mpi_partinfo_type,0,me,ctx->mpicomm);
    }
    if (me == 0) {
        for (i=1; i<nprocs; i++) {
            MPI_Recv(&global_partinfo[ctx->row_map->goffs[i]],ctx->row_map->ldim[i],mpi_partinfo_type,i,i,ctx->mpicomm,MPI_STATUS_IGNORE);
        }
        memcpy(global_partinfo,partinfo,SPM_NROWS(mat)*sizeof(part_info));
    }
    INFO_LOG("after gather");
    qsort(global_partinfo,ctx->row_map->gdim,sizeof(part_info),part_info_cmp);
    
    INFO_LOG("after sort");
    if (me == 0) {
        for (i=1; i<nprocs; i++) {
            MPI_Send(&global_partinfo[ctx->row_map->goffs[i]],ctx->row_map->ldim[i],mpi_partinfo_type,i,i,ctx->mpicomm);
        }
        memcpy(partinfo,global_partinfo,SPM_NROWS(mat)*sizeof(part_info));
    }
    if (me != 0) {
        MPI_Recv(partinfo,ctx->row_map->ldim[me],mpi_partinfo_type,0,me,ctx->mpicomm,MPI_STATUS_IGNORE);
    }
    INFO_LOG("after scatter");

    MPI_Type_free(&mpi_partinfo_type);

#if 0

    qsort(partinfo,SPM_NROWS(mat),sizeof(part_info),part_info_cmp);
    for (i=0; i<numExport; i++) {
        printf("rank %d sorted partinfo[%d] = {%d,%d}\n",me,i,partinfo[i].part,partinfo[i].row);
        printf("send from %d to %d, tag %d\n",me,partinfo[i].part,me);
        if (partinfo[i].part != me) {
        MPI_Send(&partinfo[i].row,1,ghost_mpi_dt_gidx,partinfo[i].part,0,ctx->mpicomm);
   //     printf("recv to %d from %d, tag %d\n",me,partinfo[i].part,partinfo[i].part);
        MPI_Recv(&partinfo[i].row,1,ghost_mpi_dt_gidx,MPI_ANY_SOURCE,0,ctx->mpicomm,MPI_STATUS_IGNORE);
        }
    }
#endif
    for (i=0; i<SPM_NROWS(mat); i++) {
        ctx->row_map->glb_perm_inv[i] = partinfo[i].row;
    }
    

    ghost_global_perm_inv(ctx->row_map->glb_perm,ctx->row_map->glb_perm_inv,ctx);
    //ghost_global_perm_inv(ctx->row_map->glb_perm_inv,ctx->row_map->glb_perm,ctx);
    
    goto out;
err:
    free(ctx->row_map->glb_perm); ctx->row_map->glb_perm = NULL;
    free(ctx->row_map->glb_perm_inv); ctx->row_map->glb_perm_inv = NULL;
    
out:
    free(partinfo);
    free(global_partinfo);
    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
    Zoltan_Destroy(&zz);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;

#endif
}

