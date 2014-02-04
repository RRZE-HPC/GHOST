#ifndef GHOST_INSTR_H
#define GHOST_INSTR_H

#include "util.h"

#if GHOST_HAVE_INSTR_TIMING

#include "log.h"

#define GHOST_INSTR_START(tag) double __start_##tag = ghost_wctime();
#define GHOST_INSTR_STOP(tag) LOG(TIMING,ANSI_COLOR_BLUE, "%s: %e secs" ANSI_COLOR_RESET,\
#tag,ghost_wctime()-__start_##tag);

#elif GHOST_HAVE_INSTR_LIKWID

#include <likwid.h>
#define GHOST_INSTR_START(tag) {\
_Pragma("omp parallel")\
    LIKWID_MARKER_START(#tag);}
#define GHOST_INSTR_STOP(tag) {\
_Pragma("omp parallel")\
    LIKWID_MARKER_STOP(#tag);}

#else

#define GHOST_INSTR_START(tag)
#define GHOST_INSTR_STOP(tag)

#endif

#endif
