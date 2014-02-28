#include "ghost/config.h"
#include "ghost/types.h"
#include "ghost/util.h"
#include "ghost/timing.h"
#include "ghost/rand.h"
#include "ghost/machine.h"
#include "ghost/locality.h"


static unsigned int* ghost_rand_states=NULL;

ghost_error_t ghost_rand_create()
{
    ghost_error_t ret = GHOST_SUCCESS;
    int i;
    int nthreads;
    int rank;

    GHOST_CALL_GOTO(ghost_machine_nPus(&nthreads, GHOST_NUMANODE_ANY),err,ret);
    GHOST_CALL_GOTO(ghost_getRank(MPI_COMM_WORLD,&rank),err,ret);

    if (!ghost_rand_states) {
        GHOST_CALL_GOTO(ghost_malloc((void **)&ghost_rand_states,nthreads*sizeof(unsigned int)),err,ret);
    }

    for (i=0; i<nthreads; i++) {
        double time;
        GHOST_CALL_GOTO(ghost_wctimeMilli(&time),err,ret);

        unsigned int seed=(unsigned int)ghost_hash(
                (int)time,
                rank,
                i);
        ghost_rand_states[i] = seed;
    }

    goto out;
err:
    ERROR_LOG("Free rand states");
    free(ghost_rand_states);

out:

    return ret;
}

ghost_error_t ghost_rand_get(unsigned int *s)
{
    int pu;
    GHOST_CALL_RETURN(ghost_getCore(&pu));
    if (!s) {
        ERROR_LOG("NULL pointer");
        return GHOST_ERR_INVALID_ARG;
    }
    *s = ghost_rand_states[pu];

    return GHOST_SUCCESS;
}

void ghost_rand_destroy()
{

    free(ghost_rand_states);
    ghost_rand_states=NULL;

}
