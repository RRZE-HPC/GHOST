#include "ghost/densemat_rm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include <complex>

#define ROWMAJOR
#include "ghost/densemat_iter_macros.h"

//TODO fix this- for some reasons broken

    template<typename T>
static ghost_error ghost_densemat_rm_averagehalo_tmpl(ghost_densemat *vec, ghost_context *ctx)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;

    int rank, nrank, i, j,  ctr, start, d, acc_dues = 0;
    T *work = NULL, *curwork = NULL;
    MPI_Request *req = NULL;
    T *sum = NULL;


    /*    if (vec->traits.ncols > 1) {
          ERROR_LOG("Multi-vec case not yet implemented");
          ret = GHOST_ERR_NOT_IMPLEMENTED;
          goto err;
          }
          */

    if (ctx == NULL) {
        WARNING_LOG("Trying to average the halos of a densemat which has no context!");
        goto out;
    }

    GHOST_CALL_GOTO(ghost_rank(&rank,ctx->mpicomm),err,ret);
    GHOST_CALL_GOTO(ghost_nrank(&nrank,ctx->mpicomm),err,ret);

    if(nrank > 1) {  
        for (i=0; i<nrank; i++) {
            acc_dues += ctx->dues[i];
        }

        GHOST_CALL_GOTO(ghost_malloc((void **)&work, vec->traits.ncols*acc_dues*sizeof(T)),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&req, 2*nrank*sizeof(MPI_Request)),err,ret);

        for (i=0; i<2*nrank; i++) {
            req[i] = MPI_REQUEST_NULL;
        }

        /* if(ctx->perm_local && ctx->flags & GHOST_PERM_NO_DISTINCTION){
           for (int to_PE=0; to_PE<nrank; to_PE++) {
           T* packed_data;
           GHOST_CALL_GOTO(ghost_malloc((void **)&packed_data, ctx->wishes[to_PE]*vec->traits.ncols*vec->elSize),err,ret);

        //   printf("packed data\n");	
        for (int i=0; i<ctx->wishes[to_PE]; i++){
        for (int c=0; c<vec->traits.ncols; c++) {
        memcpy(&packed_data[(c*ctx->wishes[to_PE]+i)],DENSEMAT_VALPTR(vec,ctx->perm_local->colPerm[ctx->hput_pos[to_PE]+i],c),vec->elSize);
        //			printf("%f\n",packed_data[(c*ctx->wishes[to_PE]+i)]);

        }
        }
        //TODO blocking and delete packed_data        
        MPI_CALL_GOTO(MPI_Isend(packed_data,ctx->wishes[to_PE]*vec->traits.ncols,vec->mpidt,to_PE,rank,ctx->mpicomm,&req[to_PE]),err,ret);

        }

        } else {*/
        for (i=0; i<nrank; i++) {
            MPI_CALL_GOTO(MPI_Isend(&((T *)vec->val)[ctx->hput_pos[i]*vec->traits.ncols],ctx->wishes[i]*vec->traits.ncols,vec->mpidt,i,rank,ctx->mpicomm,&req[i]),err,ret);
        }
        // }

        curwork = work;
        for (i=0; i<nrank; i++) {
            MPI_CALL_GOTO(MPI_Irecv(curwork,ctx->dues[i]*vec->traits.ncols,vec->mpidt,i,i,ctx->mpicomm,&req[nrank+i]),err,ret);
            curwork += ctx->dues[i];
        }

        MPI_CALL_GOTO(MPI_Waitall(2*nrank,req,MPI_STATUSES_IGNORE),err,ret);
        GHOST_CALL_GOTO(ghost_malloc((void **)&sum,vec->traits.ncols*ctx->nElemAvg*sizeof(T)),err,ret);

#pragma omp parallel for schedule(runtime) private(j,ctr)   //which one to parallelise(inner or outer) depends on the matrix
        for(i=0 ; i<ctx->nChunkAvg; ++i) {
            // #pragma omp parallel for schedule(runtime) private(j,k,ctr)
            for(j=ctx->avg_ptr[2*i]; j<ctx->avg_ptr[2*i+1]; ++j) {       
                 ctr = ctx->mapAvg[j];
                if(ctx->col_map->loc_perm_inv && ctx->col_map->loc_perm_inv[j]<ctx->row_map->ldim[rank]) {
                    if(ctx->entsInCol[ctx->col_map->loc_perm_inv[j]] != 0) {//this is necessary since there can be cases where only halos are present
                        for(int k=0; k<vec->traits.ncols; ++k) {
                            sum[ctr*vec->traits.ncols+k] = ((T *)vec->val)[j*vec->traits.ncols+k];
                        }
                    } else {
                        for(int k=0; k<vec->traits.ncols; ++k) {
                            sum[ctr*vec->traits.ncols+k] = 0;
                        }
                    }
                } else {
                    if(ctx->entsInCol[j] != 0) {//this is necessary since there can be cases where only halos are present
                        for(int k=0; k<vec->traits.ncols; ++k) {
                            sum[ctr*vec->traits.ncols+k] = ((T *)vec->val)[j*vec->traits.ncols+k];
                        }
                    } else {
                        for(int k=0; k<vec->traits.ncols; ++k) {
                            sum[ctr*vec->traits.ncols+k] = 0;
                        }
                    }
                } 
            }
        }

        curwork = work;


        start = 0;
        for (i=0; i<nrank; i++) {
            start += (i==0?0:ctx->dues[i-1]);
#pragma omp parallel for schedule(static) private(d,j,ctr)
            for (d=0 ;d < ctx->dues[i]; d++) {
                ctr = start + d;
                for(j=0 ; j<vec->traits.ncols; ++j) {
                    // printf("idx = %d\tsum=%f\n",(ctx->mappedDuelist[ctr])*vec->traits.ncols+j,sum[(ctx->mappedDuelist[ctr])*vec->traits.ncols+j]);
                    sum[(ctx->mappedDuelist[ctr])*vec->traits.ncols+j] += curwork[d*vec->traits.ncols+j];   
                }
            } 
            curwork += ctx->dues[i]*vec->traits.ncols;
        }

#pragma omp parallel for schedule(runtime) private(j,ctr) 
        for(i=0 ; i<ctx->nChunkAvg; ++i) {
            //  #pragma omp parallel for schedule(runtime) private(j,k,ctr)
            for(j=ctx->avg_ptr[2*i]; j<ctx->avg_ptr[2*i+1]; ++j) {
                ctr = ctx->mapAvg[j];
                for(int k=0; k<vec->traits.ncols; ++k) {
                    ((T *)vec->val)[j*vec->traits.ncols+k] = sum[ctr*vec->traits.ncols+k]/(T)ctx->nrankspresent[ctr];
                }
            }
        }


        goto out;
err:

out:
        free(sum);
        free(work);
        free(req);
        GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
        return ret;

    } else {
        return ret;
    }

#else
    UNUSED(vec);
    ERROR_LOG("MPI is required!");
    return GHOST_ERR_MPI;
#endif
}

ghost_error ghost_densemat_rm_averagehalo_selector(ghost_densemat *vec, ghost_context *ctx)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,std::complex,ret,ghost_densemat_rm_averagehalo_tmpl,vec,ctx);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
}
