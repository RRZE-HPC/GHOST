#include "ghost/densemat.h"
#include "ghost/util.h"
#include "ghost/locality.h"

ghost_error_t ghost_densemat_averagehalo(ghost_densemat_t *vec)
{
    INFO_LOG("averaging halos");
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;
    
    int rank, nrank, i, j, acc_dues = 0;
    char *work = NULL;
    MPI_Request *req = NULL;

    if (vec->context == NULL) {
        WARNING_LOG("Trying to average the halos of a densemat which has no context!");
        goto out;
    }

    GHOST_CALL_GOTO(ghost_rank(&rank,vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nrank,vec->context->mpicomm),err,ret);
   
    for (i=0; i<nrank; i++) {
       acc_dues += vec->context->dues[i];
    }
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&work, acc_dues*vec->elSize),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&req, 2*nrank*sizeof(MPI_Request)),err,ret);
    
    for (i=0; i<2*nrank; i++) {
        req[i] = MPI_REQUEST_NULL;
    }

    for (i=0; i<nrank; i++) {
        MPI_CALL_GOTO(MPI_Isend(&vec->val[0][vec->context->hput_pos[i]*vec->elSize],vec->context->wishes[i],vec->row_mpidt,i,rank,vec->context->mpicomm,&req[i]),err,ret);
    }

    char *curwork = work;
    for (i=0; i<nrank; i++) {
        MPI_CALL_GOTO(MPI_Irecv(curwork,vec->context->dues[i],vec->row_mpidt,i,i,vec->context->mpicomm,&req[nrank+i]),err,ret);
        curwork += vec->context->dues[i]*vec->elSize;
    }

    MPI_CALL_GOTO(MPI_Waitall(2*nrank,req,MPI_STATUSES_IGNORE),err,ret);
    
    ghost_lidx_t *curdue;
    double *sum;
    int *nrankspresent;
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&curdue, nrank*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&sum, vec->context->lnrows[rank]*sizeof(double)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&nrankspresent, vec->context->lnrows[rank]*sizeof(int)),err,ret);

    for (i=0; i<nrank; i++) {
        curdue[i] = 0;
    }
    
    for (i=0; i<vec->context->lnrows[rank]; i++) {
        sum[i] = ((double *)vec->val[0])[i];
        nrankspresent[i] = 1;
    }
    
    ghost_lidx_t currow;
    curwork = work;
    for (i=0; i<nrank; i++) {
        if (i == rank) {
            continue;
        }
        for (currow=0; currow<vec->context->lnrows[rank] && curdue[i] < vec->context->dues[i]; currow++) {
            if (vec->context->duelist[i][curdue[i]] == currow) {
                sum[currow] += ((double *)curwork)[curdue[i]];
                curdue[i]++;
                nrankspresent[currow]++;
            }
        }
        curwork += vec->context->dues[i]*vec->elSize;
    }
        
    for (currow=0; currow<vec->context->lnrows[rank]; currow++) {
        ((double *)vec->val[0])[currow] = sum[currow]/nrankspresent[currow];
    }
        
    goto out;
err:

out:
    free(curdue);
    free(sum);
    free(nrankspresent);
    free(work);
    free(req);
    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
}
