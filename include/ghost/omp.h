#ifndef GHOST_OMP_H
#define GHOST_OMP_H

#ifdef __cplusplus
extern "C" {
#endif
    
    void ghost_ompSetNumThreads(int nthreads);
    int ghost_ompGetThreadNum();
    int ghost_ompGetNumThreads();

#ifdef __cplusplus
}
#endif

#endif
