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

ghost_error_t ghost_rand_seed(unsigned int global_seed)
{
    ghost_error_t ret = GHOST_SUCCESS;
    int i;
    int rank;

    GHOST_CALL_GOTO(ghost_machine_npu(&nrand, GHOST_NUMANODE_ANY),err,ret);
    GHOST_CALL_GOTO(ghost_rank(&rank, MPI_COMM_WORLD),err,ret);

    if (!ghost_rand_states) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&ghost_rand_states,nrand*sizeof(unsigned int)),err,ret);
    }

    for (i=0; i<nrand; i++) {
        //double dtime;
        //GHOST_CALL_GOTO(ghost_timing_wcmilli(&dtime),err,ret);
        //int time = (int)(((int64_t)(dtime)) & 0xFFFFFFLL);

        unsigned int seed=(unsigned int)ghost_hash(
                //time,
                global_seed,
                rank,
                i);
        ghost_rand_states[i] = seed;
    }

    cu_seed = global_seed;

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

ghost_error_t ghost_rand_set1(unsigned int seed) 
{
    int i;
    ghost_type_t type;

    GHOST_CALL_RETURN(ghost_type_get(&type));
    if (type == GHOST_TYPE_CUDA) {
        WARNING_LOG("This function does not really do what you would expect for CUDA random numbers, the random seed will differ between threads!");
        cu_seed = seed;
    }

    for (i=0; i<nrand; i++) {
        ghost_rand_states[i] = seed;
    }

    return GHOST_SUCCESS;
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
