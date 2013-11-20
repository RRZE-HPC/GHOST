#ifndef __GHOST_CONTEXT_H__
#define __GHOST_CONTEXT_H__

#include <ghost_config.h>
#include <ghost_types.h>

#ifdef __cplusplus
extern "C" {
#endif

ghost_context_t * ghost_createContext(int64_t, int64_t, int, char *, MPI_Comm, double weight);
void              ghost_freeContext(ghost_context_t *);

#ifdef __cplusplus
} //extern "C"
#endif

#endif
