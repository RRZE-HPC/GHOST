#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/affinity.h"
#include "ghost/vec.h"
#include "ghost/util.h"
#include "ghost/constants.h"

#include <mpi.h>
#include <stdio.h>
#include <string.h>

#if GHOST_HAVE_OPENMP
#include <omp.h>
#endif

// if called with context==NULL: clean up variables
void hybrid_kernel_I(ghost_context_t *context, ghost_vec_t* res, ghost_mat_t* mat, ghost_vec_t* invec, int spmvmOptions)
{

    /*****************************************************************************
     ********                  Kernel ir -- cs -- wa -- ca                ********   
     ********          Kommunikation mittels MPI_ISend, MPI_IRecv         ********
     ********                serielles Umkopieren und Senden              ********
     ****************************************************************************/

    ghost_mnnz_t max_dues;
    int nprocs;

    int me; 
    int i, from_PE, to_PE;
    int msgcount;
    ghost_vidx_t c;

    char *work = NULL;
    MPI_Request *request = NULL;
    MPI_Status  *status = NULL;

    size_t sizeofRHS;

    if (context == NULL)
      return;

    me = ghost_getRank(context->mpicomm);
    nprocs = ghost_getNumberOfRanks(context->mpicomm);
    sizeofRHS = ghost_sizeofDataType(invec->traits->datatype);

    max_dues = 0;
    for (i=0;i<nprocs;i++)
        if (context->dues[i]>max_dues) 
            max_dues = context->dues[i];

    GHOST_INSTR_START(spMVM_vectormode_comm);
    invec->downloadNonHalo(invec);
    work = (char *)ghost_malloc(invec->traits->nvecs*max_dues*nprocs * ghost_sizeofDataType(invec->traits->datatype));

    request = (MPI_Request*) ghost_malloc(invec->traits->nvecs*2*nprocs*sizeof(MPI_Request));
    status  = (MPI_Status*)  ghost_malloc(invec->traits->nvecs*2*nprocs*sizeof(MPI_Status));

#ifdef __INTEL_COMPILER
    kmp_set_blocktime(1);
#endif

    msgcount=0;
    for (i=0;i<invec->traits->nvecs*2*nprocs;i++) { 
        request[i] = MPI_REQUEST_NULL;
    }

    for (from_PE=0; from_PE<nprocs; from_PE++){
        if (context->wishes[from_PE]>0){
            for (c=0; c<invec->traits->nvecs; c++) {
                MPI_safecall(MPI_Irecv(VECVAL(invec,invec->val,c,context->hput_pos[from_PE]), context->wishes[from_PE]*sizeofRHS,MPI_CHAR, from_PE, from_PE, context->mpicomm,&request[msgcount] ));
                msgcount++;
            }
        }
    }

    GHOST_INSTR_START(spMVM_vectormode_copybuffers)
#pragma omp parallel private(to_PE,i,c)
    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
        for (c=0; c<invec->traits->nvecs; c++) {
#pragma omp for 
            for (i=0; i<context->dues[to_PE]; i++){
                memcpy(work + c*nprocs*max_dues*sizeofRHS + (to_PE*max_dues+i)*sizeofRHS,VECVAL(invec,invec->val,c,context->duelist[to_PE][i]),sizeofRHS);
            }
        }
    }
    GHOST_INSTR_STOP(spMVM_vectormode_copybuffers)


    for (to_PE=0 ; to_PE<nprocs ; to_PE++){
        if (context->dues[to_PE]>0){
            for (c=0; c<invec->traits->nvecs; c++) {
                MPI_safecall(MPI_Isend( work + c*nprocs*max_dues*sizeofRHS + to_PE*max_dues*sizeofRHS, context->dues[to_PE]*sizeofRHS, MPI_CHAR, to_PE, me, context->mpicomm, &request[msgcount] ));
                msgcount++;
            }
        }
    }

    GHOST_INSTR_START(spMVM_vectormode_waitall)
    MPI_safecall(MPI_Waitall(msgcount, request, status));
    GHOST_INSTR_STOP(spMVM_vectormode_waitall)

    invec->uploadHalo(invec);
    GHOST_INSTR_STOP(spMVM_vectormode_comm);
    
    GHOST_INSTR_START(spMVM_vectormode_comp);
    mat->spmv(mat,res,invec,spmvmOptions);    
    GHOST_INSTR_STOP(spMVM_vectormode_comp);

    free(work);
    free(request);
    free(status);
}

