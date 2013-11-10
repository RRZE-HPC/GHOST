#ifndef __GHOST_CONTEXT_H__
#define __GHOST_CONTEXT_H__

#include <ghost_types.h>

ghost_context_t * ghost_createContext(int64_t, int64_t, int, char *, MPI_Comm, double weight);
void              ghost_freeContext(ghost_context_t *);

#endif
