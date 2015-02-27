#include "ghost/config.h"
#include "ghost/omp.h"
#include "ghost/util.h"

#ifdef GHOST_HAVE_OPENMP
#include <omp.h>
#endif

void ghost_omp_nthread_set(int nthreads)
{
#ifdef GHOST_HAVE_OPENMP
    omp_set_num_threads(nthreads);
#else
    UNUSED(nthreads);
#endif
}

int ghost_omp_nthread()
{
#ifdef GHOST_HAVE_OPENMP
    return omp_get_num_threads();
#else 
    return 1;
#endif
}

int ghost_omp_threadnum()
{
#ifdef GHOST_HAVE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

int ghost_omp_get_nested()
{
#ifdef GHOST_HAVE_OPENMP
    return omp_get_nested();
#else
    return 0;
#endif
}

int ghost_omp_in_parallel()
{
#ifdef GHOST_HAVE_OPENMP
    return omp_in_parallel();
#else
    return 0;
#endif
}


void ghost_omp_set_nested(int nested)
{
#ifdef GHOST_HAVE_OPENMP
    omp_set_nested(nested);
#else
    UNUSED(nested);
#endif
}
