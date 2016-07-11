#include "ghost/densemat_cm.h"
#include "ghost/util.h"
#include "ghost/locality.h"
#include <complex>

#define COLMAJOR
#include "ghost/densemat_iter_macros.h"

template<typename T>
static ghost_error ghost_densemat_cm_averagehalo_tmpl(ghost_densemat *vec)
{
#ifdef GHOST_HAVE_MPI
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;

    int rank, nrank, i, d, acc_dues = 0;
    T *work = NULL, *curwork = NULL;
    MPI_Request *req = NULL;
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

   /* if(vec->context->perm_local && vec->context->flags & GHOST_PERM_NO_DISTINCTION){
    	for (int to_PE=0; to_PE<nrank; to_PE++) {
		T* packed_data;
		GHOST_CALL_GOTO(ghost_malloc((void **)&packed_data, vec->context->wishes[to_PE]*vec->traits.ncols*vec->elSize),err,ret);

    //   printf("packed data\n");	
		for (int i=0; i<vec->context->wishes[to_PE]; i++){
                    for (int c=0; c<vec->traits.ncols; c++) {
                       memcpy(&packed_data[(c*vec->context->wishes[to_PE]+i)],DENSEMAT_VALPTR(vec,vec->context->perm_local->colPerm[vec->context->hput_pos[to_PE]+i],c),vec->elSize);
//			printf("%f\n",packed_data[(c*vec->context->wishes[to_PE]+i)]);

                    }
                }
               //TODO blocking and delete packed_data        
               MPI_CALL_GOTO(MPI_Isend(packed_data,vec->context->wishes[to_PE]*vec->traits.ncols,vec->mpidt,to_PE,rank,vec->context->mpicomm,&req[to_PE]),err,ret);
        
   	}

    } else {*/
        for (i=0; i<nrank; i++) {
                MPI_CALL_GOTO(MPI_Isend(&((T *)vec->val)[vec->context->hput_pos[i]],vec->context->wishes[i]*vec->traits.ncols,vec->mpidt,i,rank,vec->context->mpicomm,&req[i]),err,ret);
        }
   // }

    curwork = work;
    for (i=0; i<nrank; i++) {
         MPI_CALL_GOTO(MPI_Irecv(curwork,vec->context->dues[i]*vec->traits.ncols,vec->mpidt,i,i,vec->context->mpicomm,&req[nrank+i]),err,ret);
        curwork += vec->context->dues[i];
    }
    

    MPI_CALL_GOTO(MPI_Waitall(2*nrank,req,MPI_STATUSES_IGNORE),err,ret);
   


     GHOST_CALL_GOTO(ghost_malloc((void **)&sum, vec->traits.nrows*sizeof(T)),err,ret);
     GHOST_CALL_GOTO(ghost_malloc((void **)&nrankspresent, vec->traits.nrows*sizeof(int)),err,ret);

#pragma omp parallel for schedule(static) private(i)
    for (i=0; i<vec->traits.nrows; i++) {
	if(vec->context->flags & GHOST_PERM_NO_DISTINCTION){
        	nrankspresent[i] = vec->context->entsInCol[vec->context->perm_local->colInvPerm[i]]?1:0; //this has also to be permuted since it was
													 //for unpermuted columns that we calculate
		if(nrankspresent[i]==1)
	        	sum[i] = ((T *)vec->val)[i];
		else
			sum[i] = 0;
	} else {
		nrankspresent[i] = vec->context->entsInCol[i]?1:0;		
		if(nrankspresent[i]==1)
		        sum[i] = ((T *)vec->val)[i];
		else
			sum[i] = 0;
	}

     }
    
     ghost_lidx currow;
     curwork = work;

    for (i=0; i<nrank; i++) {
	if(vec->context->perm_local) {
#pragma omp parallel for schedule(static) private(d)
	        for (d=0 ;d < vec->context->dues[i]; d++) {
        	    sum[vec->context->perm_local->colPerm[vec->context->duelist[i][d]]] += curwork[d];
             	    nrankspresent[vec->context->perm_local->colPerm[vec->context->duelist[i][d]]]++; 
		}
	} else {
#pragma omp parallel for schedule(static) private(d)
	        for (d=0 ;d < vec->context->dues[i]; d++) {
        	    sum[vec->context->duelist[i][d]] += curwork[d];
            	    nrankspresent[vec->context->duelist[i][d]]++; 
		}
	}
           curwork += vec->context->dues[i];
     }

#pragma omp parallel for schedule(static) private(currow)
    for (currow=0; currow<vec->traits.nrows; currow++) {
      if(nrankspresent[currow]!=0) {
 	       ((T *)vec->val)[currow] = sum[currow]/(T)nrankspresent[currow];
	}
     }

/* } else { 
    GHOST_CALL_GOTO(ghost_malloc((void **)&sum, vec->context->lnrows[rank]*sizeof(T)),err,ret);
    GHOST_CALL_GOTO(ghost_malloc((void **)&nrankspresent, vec->context->lnrows[rank]*sizeof(int)),err,ret);

    
    for (i=0; i<vec->context->lnrows[rank]; i++) {
        sum[i] = ((T *)vec->val)[i];
    	nrankspresent[i] = vec->context->entsInCol[i]?1:0;	
    }
    
    ghost_lidx currow;
    curwork = work;
    for (i=0; i<nrank; i++) {
        for (d=0 ;d < vec->context->dues[i]; d++) {
	   if(vec->context->perm_local) {
	           sum[vec->context->perm_local->colPerm[vec->context->duelist[i][d]]] +=  curwork[d];
	           nrankspresent[vec->context->perm_local->colPerm[vec->context->duelist[i][d]]]++;
	   } else {
          	   sum[vec->context->duelist[i][d]] += curwork[d];
           	   nrankspresent[vec->context->duelist[i][d]]++;
	   }        
        }
        curwork += vec->context->dues[i];
    }

      for (i=0; i<vec->context->lnrows[rank]; i++) {
        printf("<%d> ranks of row[%d] = %d\n",rank,i,nrankspresent[i]);
    }

        
    for (currow=0; currow<vec->context->lnrows[rank]; currow++) { 
        ((T *)vec->val)[currow] = sum[currow]/(T)nrankspresent[currow];
    }
   }
*/

    goto out;
err:

out:
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

ghost_error ghost_densemat_cm_averagehalo_selector(ghost_densemat *vec)
{
    GHOST_FUNC_ENTER(GHOST_FUNCTYPE_COMMUNICATION);
    ghost_error ret = GHOST_SUCCESS;

    SELECT_TMPL_1DATATYPE(vec->traits.datatype,std::complex,ret,ghost_densemat_cm_averagehalo_tmpl,vec);

    GHOST_FUNC_EXIT(GHOST_FUNCTYPE_COMMUNICATION);
    return ret;
}
