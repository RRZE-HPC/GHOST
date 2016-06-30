#include "ghost/context.h"
#include "ghost/locality.h"
#include "ghost/util.h"

typedef struct {
    ghost_gidx idx, pidx;
} ghost_permutation_ent_t;
                    
static int perm_ent_cmp(const void *a, const void *b)
{
    return ((ghost_permutation_ent_t *)a)->pidx - ((ghost_permutation_ent_t *)b)->pidx;
}

ghost_error ghost_global_perm_inv(ghost_gidx *toPerm, ghost_gidx *fromPerm, ghost_context *context)
{
    ghost_mpi_datatype ghost_mpi_dt_perm;
    MPI_CALL_RETURN(MPI_Type_contiguous(2,ghost_mpi_dt_gidx,&ghost_mpi_dt_perm));
    MPI_CALL_RETURN(MPI_Type_commit(&ghost_mpi_dt_perm));

    ghost_lidx i;
    int proc, me, nprocs;
    ghost_rank(&me,context->mpicomm);
    ghost_nrank(&nprocs,context->mpicomm);

    ghost_permutation_ent_t *permclone;
    ghost_malloc((void **)&permclone,sizeof(ghost_permutation_ent_t)*context->lnrows[me]);

#pragma omp parallel for
    for (i=0; i<context->lnrows[me]; i++) {
        permclone[i].idx = context->lfRow[me]+i;
        permclone[i].pidx = fromPerm[i];
    }
    qsort(permclone,context->lnrows[me],sizeof(ghost_permutation_ent_t),perm_ent_cmp);
    // permclone is now sorted by ascending pidx

    ghost_lidx offs = 0;
    for (proc = 0; proc<nprocs; proc++) {
        int displ[nprocs];
        int nel[nprocs];
        int recvdispl[nprocs];
        memset(displ,0,sizeof(displ));
        memset(nel,0,sizeof(nel));

        // find 1st pidx in sorted permclone which lies in process proc
        while((offs < context->lnrows[me]) && (permclone[offs].pidx < context->lfRow[proc])) {
            offs++;
        }
        displ[me] = offs;
        
        // find last pidx in sorted permclone which lies in process proc
        while((offs < context->lnrows[me]) && (permclone[offs].pidx < context->lfRow[proc]+context->lnrows[proc])) {
            offs++;
        }
        nel[me] = offs-displ[me];

        // proc needs to know how many elements to receive from each process
        if (proc == me) { 
            MPI_Reduce(MPI_IN_PLACE,nel,nprocs,MPI_INT,MPI_MAX,proc,context->mpicomm);
        } else {
            MPI_Reduce(nel,NULL,nprocs,MPI_INT,MPI_MAX,proc,context->mpicomm);
        }

        // assemble receive displacements
        if (proc == me) {
            recvdispl[0] = 0;
            for (i=1; i<nprocs; i++) {
                recvdispl[i] = recvdispl[i-1] + nel[i-1];
            }
            
        }

        // prepare receive buffer
        ghost_permutation_ent_t *recvbuf = NULL;
        if (proc == me) {
            ghost_malloc((void **)&recvbuf,context->lnrows[me]*sizeof(ghost_permutation_ent_t));
        }

        // gather local invPerm
        MPI_Gatherv(&permclone[displ[me]],nel[me],ghost_mpi_dt_perm,recvbuf,nel,recvdispl,ghost_mpi_dt_perm,proc,context->mpicomm);
        
        if (proc == me) {
            // sort the indices and put them into the invPerm array
            qsort(recvbuf,context->lnrows[me],sizeof(ghost_permutation_ent_t),perm_ent_cmp);
            for (i=0; i<context->lnrows[me]; i++) {
                toPerm[i] = recvbuf[i].idx;
            }
        }

        if (proc == me) {
            free(recvbuf);
        }
    }

    free(permclone);
        
    MPI_CALL_RETURN(MPI_Type_free(&ghost_mpi_dt_perm));

    return GHOST_SUCCESS;
}
