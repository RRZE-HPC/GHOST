#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/timing.h"
#include "ghost/rand.h"
#include "ghost/machine.h"
#include "ghost/omp.h"
#include "ghost/locality.h"
#include "ghost/core.h"


static unsigned int* ghost_rand_states=NULL;
static int nrand = 0;
static unsigned int cu_seed = 0;

ghost_error_t ghost_rand_create()
{
    ghost_error_t ret = GHOST_SUCCESS;
    int i;
    int rank;
    int time;
    double dtime;

    GHOST_CALL_GOTO(ghost_machine_npu(&nrand, GHOST_NUMANODE_ANY),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&rank, MPI_COMM_WORLD),err,ret);

    if (!ghost_rand_states) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&ghost_rand_states,nrand*sizeof(unsigned int)),err,ret);
    }

    for (i=0; i<nrand; i++) {
        GHOST_CALL_GOTO(ghost_timing_wcmilli(&dtime),err,ret);
        time = (int)(((int64_t)(dtime)) & 0xFFFFFFLL);

        unsigned int seed=(unsigned int)ghost_hash(
                time,
                rank,
                i);
        ghost_rand_states[i] = seed;
    }

    GHOST_CALL_GOTO(ghost_timing_wcmilli(&dtime),err,ret);
    time = (int)(((int64_t)(dtime)) & 0xFFFFFFLL);
    cu_seed = (unsigned int)ghost_hash((int)time,clock(),(int)getpid());

    goto out;
err:
    ERROR_LOG("Free rand states");
    free(ghost_rand_states);

out:

    return ret;
}

ghost_error_t ghost_rand_get(unsigned int **s)
{
    if (!s) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    *s = &ghost_rand_states[ghost_omp_threadnum()];

    return GHOST_SUCCESS;
}

ghost_error_t ghost_rand_seed(ghost_rand_seed_t which, unsigned int seed)
{
    ghost_error_t ret = GHOST_SUCCESS;
    double dtime;
    int i, rank, time;
    ghost_type_t type;

    GHOST_CALL_GOTO(ghost_type_get(&type),err,ret);
    GHOST_CALL_GOTO(ghost_machine_npu(&nrand, GHOST_NUMANODE_ANY),err,ret);
    
    if (which & GHOST_RAND_SEED_RANK) {
        rank = (int)seed;
    } else {
        GHOST_CALL_GOTO(ghost_rank(&rank, MPI_COMM_WORLD),err,ret);
    }

    if (which & GHOST_RAND_SEED_TIME) {
        GHOST_CALL_GOTO(ghost_timing_wcmilli(&dtime),err,ret);
        time = (int)(((int64_t)(dtime)) & 0xFFFFFFLL);
    } else {
        time = (int)seed;
    }

    if (!ghost_rand_states) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&ghost_rand_states,nrand*sizeof(unsigned int)),err,ret);
    }
    
    if ((type == GHOST_TYPE_CUDA) && (which & GHOST_RAND_SEED_PU)) {
        WARNING_LOG("This function does not really do what you would expect for CUDA random numbers, the random seed will differ between threads!");
        cu_seed = ghost_hash(rank,time,0);
    }

    for (i=0; i<nrand; i++) {

        if (which & GHOST_RAND_SEED_PU) {
            ghost_rand_states[i] = (unsigned int)ghost_hash(
                time,
                rank,
                seed);
        } else {
            ghost_rand_states[i] = (unsigned int)ghost_hash(
                time,
                rank,
                i);
        }
    }


    goto out;
err:
    ERROR_LOG("Free rand states");
    free(ghost_rand_states);

out:

    return ret;

}

void ghost_rand_destroy()
{

    free(ghost_rand_states);
    ghost_rand_states=NULL;

}

int ghost_rand_cu_seed_get()
{
    return cu_seed;
}
