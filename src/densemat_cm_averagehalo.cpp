#include "ghost/densemat_cm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include <complex>

template<typename T>
static ghost_error_t ghost_densemat_cm_averagehalo_tmpl(ghost_densemat_t *vec)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;

    int rank, nrank, i, acc_dues = 0;
    T *work = NULL, *curwork = NULL;
    MPI_Request *req = NULL;
    ghost_lidx_t *curdue = NULL;
    T *sum = NULL;
    int *nrankspresent = NULL;
    
    
    if (vec->traits.ncols > 1) {
        ERROR_LOG("Multi-vec case not yet implemented");
        ret = GHOST_ERR_NOT_IMPLEMENTED;
        goto err;
    }

    if (vec->context == NULL) {
        WARNING_LOG("Trying to average the halos of a densemat which has no context!");
        goto out;
    }

    GHOST_CALL_GOTO(ghost_rank(&rank,vec->context->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nrank,vec->context->mpicomm),err,ret);
   
    for (i=0; i<nrank; i++) {
       acc_dues += vec->context->dues[i];
    }
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&work, acc_dues*sizeof(T)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&req, 2*nrank*sizeof(MPI_Request)),err,ret);
    
    for (i=0; i<2*nrank; i++) {
        req[i] = MPI_REQUEST_NULL;
    }

    for (i=0; i<nrank; i++) {
        MPI_CALL_GOTO(MPI_Isend(&((T *)vec->val)[vec->context->hput_pos[i]],vec->context->wishes[i]*vec->traits.ncols,vec->mpidt,i,rank,vec->context->mpicomm,&req[i]),err,ret);
    }

    curwork = work;
    for (i=0; i<nrank; i++) {
        MPI_CALL_GOTO(MPI_Irecv(curwork,vec->context->dues[i]*vec->traits.ncols,vec->mpidt,i,i,vec->context->mpicomm,&req[nrank+i]),err,ret);
        curwork += vec->context->dues[i];
    }

    MPI_CALL_GOTO(MPI_Waitall(2*nrank,req,MPI_STATUSES_IGNORE),err,ret);
    
    GHOST_CALL_GOTO(ghost_malloc((void **)&curdue, nrank*sizeof(ghost_lidx_t)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&sum, vec->context->lnrows[rank]*sizeof(T)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&nrankspresent, vec->context->lnrows[rank]*sizeof(int)),err,ret);

    for (i=0; i<nrank; i++) {
        curdue[i] = 0;
    }
    
    for (i=0; i<vec->context->lnrows[rank]; i++) {
        sum[i] = ((T *)vec->val)[i];
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
                sum[currow] += curwork[curdue[i]];
                curdue[i]++;
                nrankspresent[currow]++;
            }
        }
        curwork += vec->context->dues[i];
    }
        
    for (currow=0; currow<vec->context->lnrows[rank]; currow++) {
        ((T *)vec->val)[currow] = sum[currow]/(T)nrankspresent[currow];
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
#else
    UNUSED(vec);
    ERROR_LOG("MPI is required!");
    return GHOST_ERR_MPI;
#endif
}

ghost_error_t ghost_densemat_cm_averagehalo_selector(ghost_densemat_t *vec)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error_t ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,std::complex,ret,ghost_densemat_cm_averagehalo_tmpl,vec);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
}
