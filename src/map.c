#include "ghost/map.h"
#include "ghost/locality.h"
#include "ghost/util.h"
#include "ghost/error.h"

ghost_error ghost_map_create(ghost_map **map, ghost_gidx gdim, ghost_mpi_comm comm)
{
    ghost_error ret = GHOST_SUCCESS;
    int nranks;

    GHOST_CALL_GOTO(ghost_malloc((void **)map,sizeof(ghost_map)),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nranks, comm),err,ret);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&((*map)->goffs),sizeof(ghost_gidx)*nranks),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&((*map)->ldim),sizeof(ghost_lidx)*nranks),err,ret);

    (*map)->gdim = gdim;
    (*map)->loc_perm = NULL;
    (*map)->loc_perm_inv = NULL;
    (*map)->glb_perm = NULL;
    (*map)->glb_perm_inv = NULL;
    (*map)->cu_loc_perm = NULL;
    (*map)->dim = 0;
    (*map)->dimpad = 0;

    goto out;
err:

out: 

    return ret;
}
    
ghost_error ghost_map_create_distribution(ghost_map *map, ghost_sparsemat_src_rowfunc *matsrc, ghost_mpi_comm mpicomm, double weight, ghost_map_dist_type distType)
{
    int me,nranks,i;
    ghost_error ret = GHOST_SUCCESS;
    ghost_gidx row;
    
    GHOST_CALL_GOTO(ghost_nrank(&nranks, mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&me,mpicomm),err,ret);
    
    if (distType == GHOST_MAP_DIST_NNZ)
    { // read rpt and fill lfrow, lnrows, lfent, lnents
        ghost_gidx *rpt;
        ghost_gidx gnnz;

        PERFWARNING_LOG("Dividing the matrix by number of non-zeros is not scalable as rank 0 reads in _all_ row lengths of the matrix!");
        WARNING_LOG("Will not take into account possible matrix re-ordering when dividing the matrix by number of non-zeros!");


        if (me == 0) {
            GHOST_CALL_GOTO(ghost_malloc((void **)&rpt,sizeof(ghost_gidx)*(map->gdim+1)),err,ret);
#pragma omp parallel for schedule(runtime)
            for( row = 0; row < map->gdim+1; row++ ) {
                rpt[row] = 0;
            }
            char *tmpval = NULL;
            ghost_gidx *tmpcol = NULL;

            GHOST_CALL_GOTO(ghost_malloc((void **)&tmpval,matsrc->maxrowlen*GHOST_DT_MAX_SIZE),err,ret);
            GHOST_CALL_GOTO(ghost_malloc((void **)&tmpcol,matsrc->maxrowlen*sizeof(ghost_gidx)),err,ret);
            rpt[0] = 0;
            ghost_lidx rowlen;
            for(row = 0; row < map->gdim; row++) {
                matsrc->func(row,&rowlen,tmpcol,tmpval,matsrc->arg);
                rpt[row+1] = rpt[row]+rowlen;
            }
            free(tmpval); tmpval = NULL;
            free(tmpcol); tmpcol = NULL;

            gnnz = rpt[map->gdim];
            ghost_lidx target_nnz;
            target_nnz = (gnnz/nranks)+1; /* sonst bleiben welche uebrig! */

            map->goffs[0]  = 0;
            ghost_lidx j = 1;

            for (row=0;row<map->gdim;row++){
                if (rpt[row] >= j*target_nnz){
                    map->goffs[j] = row;
                    j = j+1;
                }
            }
            for (i=0; i<nranks-1; i++){
                map->ldim[i] = map->goffs[i+1] - map->goffs[i] ;
                //(*context)->lnEnts[i] = (*context)->lfEnt[i+1] - (*context)->lfEnt[i] ;
            }

            map->ldim[nranks-1] = map->gdim - map->goffs[nranks-1] ;
            //(*context)->lnEnts[nranks-1] = gnnz - (*context)->lfEnt[nranks-1];

            //fclose(filed);
        }
        MPI_CALL_GOTO(MPI_Bcast(map->goffs,  nranks, ghost_mpi_dt_gidx, 0, mpicomm),err,ret);
        //MPI_CALL_GOTO(MPI_Bcast((*context)->lfEnt,  nranks, ghost_mpi_dt_gidx, 0, mpicomm),err,ret);
        MPI_CALL_GOTO(MPI_Bcast(map->ldim, nranks, ghost_mpi_dt_lidx, 0, mpicomm),err,ret);
        //MPI_CALL_GOTO(MPI_Bcast((*context)->lnEnts, nranks, ghost_mpi_dt_lidx, 0, mpicomm),err,ret);
        //MPI_CALL_GOTO(MPI_Allreduce(&((*context)->lnEnts[me]),&((*context)->gnnz),1,ghost_mpi_dt_gidx,MPI_SUM,mpicomm),err,ret);

    } else if (distType == GHOST_MAP_DIST_NROWS)
    { // don't read rpt, only fill lfrow, lnrows, rest will be done after some matrix from*() function
        ghost_lidx *target_rows = NULL;
        double allweights;
        MPI_CALL_GOTO(MPI_Allreduce(&weight,&allweights,1,MPI_DOUBLE,MPI_SUM,mpicomm),err,ret)

        ghost_lidx my_target_rows = (ghost_lidx)(map->gdim*((double)weight/(double)allweights));
        if (my_target_rows == 0) {
            WARNING_LOG("This rank will have zero rows assigned!");
        }

        GHOST_CALL_GOTO(ghost_malloc((void **)&target_rows,nranks*sizeof(ghost_lidx)),err,ret);

        MPI_CALL_GOTO(MPI_Allgather(&my_target_rows,1,ghost_mpi_dt_lidx,target_rows,1,ghost_mpi_dt_lidx,mpicomm),err,ret);
                   
        map->goffs[0] = 0;

        for (i=1; i<nranks; i++){
            map->goffs[i] = map->goffs[i-1]+target_rows[i-1];
        }
        for (i=0; i<nranks-1; i++){
            ghost_gidx lnrows = map->goffs[i+1] - map->goffs[i];
            if (lnrows > (ghost_gidx)GHOST_LIDX_MAX) {
                ERROR_LOG("Re-compile with 64-bit local indices!");
                return GHOST_ERR_UNKNOWN;
            }
            map->ldim[i] = (ghost_lidx)lnrows;
        }
        ghost_gidx lnrows = map->gdim - map->goffs[nranks-1];
        if (lnrows > (ghost_gidx)GHOST_LIDX_MAX) {
            ERROR_LOG("The local number of rows (%"PRGIDX") exceeds the maximum range. Re-compile with 64-bit local indices!",lnrows);
            return GHOST_ERR_DATATYPE;
        }
        map->ldim[nranks-1] = (ghost_lidx)lnrows;
        
        //MPI_CALL_GOTO(MPI_Bcast(map->goffs,  nranks, ghost_mpi_dt_gidx, 0, mpicomm),err,ret);
        //MPI_CALL_GOTO(MPI_Bcast(map->ldim, nranks, ghost_mpi_dt_lidx, 0, mpicomm),err,ret);
        //(*context)->lnEnts[0] = -1;
        //(*context)->lfEnt[0] = -1;
        //(*context)->gnnz = -1;

        free(target_rows); target_rows = NULL;
    }
    map->dim = map->ldim[me];
    map->offs = map->goffs[me];

    goto out;
err:

out:
    return ret;
}

void ghost_map_destroy(ghost_map *map)
{
        free(map->cu_loc_perm); map->cu_loc_perm = NULL;
        free(map->goffs); map->goffs = NULL;
        free(map->ldim); map->ldim = NULL;

}
