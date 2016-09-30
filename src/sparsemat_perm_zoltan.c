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
    
    if (mat->context->row_map->glb_perm) {
        WARNING_LOG("Existing permutations will be overwritten!");
    }
    if (srcType != GHOST_SPARSEMAT_SRC_FUNC) {
        ERROR_LOG("Only function sparse matrix source allowed!");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }
    
    GHOST_CALL_GOTO(ghost_rank(&me, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nprocs, mat->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->row_map->glb_perm,sizeof(ghost_gidx)*mat->context->row_map->lnrows[me]),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&mat->context->row_map->glb_perm_inv,sizeof(ghost_gidx)*mat->context->row_map->lnrows[me]),err,ret);
    mat->context->col_map->glb_perm = mat->context->row_map->glb_perm;
    mat->context->col_map->glb_perm_inv = mat->context->row_map->glb_perm_inv;

    memset(mat->context->row_map->glb_perm,0,sizeof(ghost_gidx)*mat->context->row_map->lnrows[me]);
    memset(mat->context->row_map->glb_perm_inv,0,sizeof(ghost_gidx)*mat->context->row_map->lnrows[me]);
           
    ghost_malloc((void **)&(info.rpt),(mat->context->row_map->lnrows[me]+1)*sizeof(ghost_gidx));

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
        for (i=0; i<mat->context->row_map->lnrows[me]; i++) {
            src->func(mat->context->row_map->goffs[me]+i,&rowlen,tmpcol,tmpval,NULL);
            nnz += rowlen;
        }
        free(tmpval); tmpval = NULL;
        free(tmpcol); tmpcol = NULL;
    }

    info.nnz = nnz;
    ghost_malloc((void **)&(info.col),(info.nnz)*sizeof(ghost_gidx));

#pragma omp parallel private (tmpval,tmpcol,i,rowlen)
    {
        ghost_malloc((void **)&tmpval,src->maxrowlen*mat->elSize);
#pragma omp for ordered
        for (i=0; i<mat->context->row_map->lnrows[me]; i++) {
#pragma omp ordered
            {
                src->func(mat->context->row_map->goffs[me]+i,&rowlen,&(info.col[info.rpt[i]]),tmpval,NULL);
                info.rpt[i+1] = info.rpt[i] + rowlen;
            }
        }
        free(tmpval); tmpval = NULL;
    }
    info.nrows = mat->context->row_map->lnrows[me];
    info.rowoffs = mat->context->row_map->goffs[me];
   
    zz = Zoltan_Create(mat->context->mpicomm);

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
    for (i=0; i<mat->context->row_map->lnrows[me]; i++) {
//        mat->context->row_map->glb_perm[i] = mat->context->row_map->goffs[me]+i;
    }

    part_info *partinfo;
    ghost_malloc((void **)&partinfo,sizeof(part_info)*SPM_NROWS(mat));
    for (i=0; i<numExport; i++) {
        partinfo[i].part = exportToPart[i];
        partinfo[i].row = mat->context->row_map->goffs[me]+i;
    }
    part_info *global_partinfo;
    ghost_malloc((void **)&global_partinfo,sizeof(part_info)*mat->context->row_map->gnrows);
    
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
        MPI_Send(partinfo,mat->context->row_map->lnrows[me],mpi_partinfo_type,0,me,mat->context->mpicomm);
    }
    if (me == 0) {
        for (i=1; i<nprocs; i++) {
            MPI_Recv(&global_partinfo[mat->context->row_map->goffs[i]],mat->context->row_map->lnrows[i],mpi_partinfo_type,i,i,mat->context->mpicomm,MPI_STATUS_IGNORE);
        }
        memcpy(global_partinfo,partinfo,SPM_NROWS(mat)*sizeof(part_info));
    }
    INFO_LOG("after gather");
    qsort(global_partinfo,mat->context->row_map->gnrows,sizeof(part_info),part_info_cmp);
    
    INFO_LOG("after sort");
    if (me == 0) {
        for (i=1; i<nprocs; i++) {
            MPI_Send(&global_partinfo[mat->context->row_map->goffs[i]],mat->context->row_map->lnrows[i],mpi_partinfo_type,i,i,mat->context->mpicomm);
        }
        memcpy(partinfo,global_partinfo,SPM_NROWS(mat)*sizeof(part_info));
        free(global_partinfo);
    }
    if (me != 0) {
        MPI_Recv(partinfo,mat->context->row_map->lnrows[me],mpi_partinfo_type,0,me,mat->context->mpicomm,MPI_STATUS_IGNORE);
    }
    INFO_LOG("after scatter");

    MPI_Type_free(&mpi_partinfo_type);

#if 0

    qsort(partinfo,SPM_NROWS(mat),sizeof(part_info),part_info_cmp);
    for (i=0; i<numExport; i++) {
        printf("rank %d sorted partinfo[%d] = {%d,%d}\n",me,i,partinfo[i].part,partinfo[i].row);
        printf("send from %d to %d, tag %d\n",me,partinfo[i].part,me);
        if (partinfo[i].part != me) {
        MPI_Send(&partinfo[i].row,1,ghost_mpi_dt_gidx,partinfo[i].part,0,mat->context->mpicomm);
   //     printf("recv to %d from %d, tag %d\n",me,partinfo[i].part,partinfo[i].part);
        MPI_Recv(&partinfo[i].row,1,ghost_mpi_dt_gidx,MPI_ANY_SOURCE,0,mat->context->mpicomm,MPI_STATUS_IGNORE);
        }
    }
#endif
    for (i=0; i<SPM_NROWS(mat); i++) {
        mat->context->row_map->glb_perm_inv[i] = partinfo[i].row;
    }
    

    ghost_global_perm_inv(mat->context->row_map->glb_perm,mat->context->row_map->glb_perm_inv,mat->context);
    //ghost_global_perm_inv(mat->context->row_map->glb_perm_inv,mat->context->row_map->glb_perm,mat->context);
    
    goto out;
err:
    free(mat->context->row_map->glb_perm); mat->context->row_map->glb_perm = NULL;
    free(mat->context->row_map->glb_perm_inv); mat->context->row_map->glb_perm_inv = NULL;
    
out:
    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
    Zoltan_Destroy(&zz);
    free(info.rpt);
    free(info.col);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_SETUP);
    return ret;

#endif
}

